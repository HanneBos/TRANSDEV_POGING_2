## Python code Optimized busplan generator page die in GITHUB stond:
import streamlit as st
import base64
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
from typing import Tuple, Optional
import re, math

# Configure pandas options
pd.options.mode.copy_on_write = True

# Set page config
st.set_page_config(page_title="Optimized Busplan - Transdev", page_icon=":bus:")

# ===== OPTIMIZER CONSTANTS AND FUNCTIONS =====

# Global containers
violations = []
suggestions = []

# Constants (from specs / brief)
BATTERY_KWH          = 300.0          # pack capacity
SOC_CAP_FRAC         = 0.90           # charge allowed up to 90% SoC
SOC_FLOOR_FRAC       = 0.10           # minimum SoC 10%
SOC_CAP_KWH          = BATTERY_KWH * SOC_CAP_FRAC
SOC_FLOOR_KWH        = BATTERY_KWH * SOC_FLOOR_FRAC
SOC_START_KWH        = SOC_CAP_KWH    # start day at the cap
IDLE_KW              = 5.0            # idle power
FAST_KW              = 450.0          # fast charger power
MIN_CHARGE_MIN       = 15.0           # min charging duration
GARAGE_NODE          = "ehvgar"       # only place allowed to charge

ENERGY_TOL_KWH       = 0.25           # numerical tolerance for energy checks
TIME_ZERO_TOL_MIN    = 0.5            # treat <0.5 min as zero

# Additional optimizer constants
SOC_MAX_FRAC         = SOC_CAP_FRAC
SOC_MIN_FRAC         = SOC_FLOOR_FRAC
TOL_MIN              = 0.5
KWH_PER_KM_MIN       = 0.7            # minimum energy consumption per km
KWH_PER_KM_MAX       = 2.5            # maximum energy consumption per km
SOH_PCT              = 0.90           # state of health percentage
DAY_CAP_FRAC         = SOC_CAP_FRAC   # day cap fraction

# Activity mapping labels
SERVICE_LABELS  = {"service","service trip","service_trip","service-trip","servicetrip"}
MATERIAL_LABELS = {"material","material trip","material_trip","material-trip","materialtrip","deadhead","dead head","dead-head"}
IDLE_LABELS     = {"idle","idling"}
CHARGING_LABELS = {"charging","charge","charging session"}

def expected_idle_kwh(duration_min: float) -> float:
    """Idle energy ~ 5 kW * hours."""
    return IDLE_KW * (float(duration_min) / 60.0)

def to_minutes(x):
    """Convert a time-like value to minutes since day start."""
    if pd.isna(x):
        return np.nan
    try:
        t = pd.to_datetime(x)
        return t.hour*60 + t.minute + t.second/60.0
    except Exception:
        return np.nan

def canonical_activity(s: str) -> str:
    s = str(s).strip().lower()
    if s in SERVICE_LABELS:  return "service"
    if s in MATERIAL_LABELS: return "material"
    if s in IDLE_LABELS:     return "idle"
    if s in CHARGING_LABELS: return "charging"
    return "unknown"

def plausible_move_kwh(km: float) -> Tuple[float,float]:
    return KWH_PER_KM_MIN*km, KWH_PER_KM_MAX*km

def soc_bounds() -> Tuple[float,float]:
    cap = BATTERY_KWH * SOH_PCT
    return cap*SOC_FLOOR_FRAC, cap*DAY_CAP_FRAC

def charge_gain_kwh(start_soc_kwh: float, dur_min: float) -> float:
    """Piecewise charger: FAST_KW up to day-cap, SLOW_KW after (usually not used if cap==90%)."""
    floor_kwh, cap_kwh = soc_bounds()
    if start_soc_kwh >= cap_kwh:
        return 0.0
    fast_room_kwh = cap_kwh - start_soc_kwh
    fast_time_min = (fast_room_kwh / FAST_KW)*60.0
    if dur_min <= fast_time_min:
        return FAST_KW * (dur_min/60.0)
    # beyond cap: apply slow (only if you allow >cap during the day; we don't)
    # so we clamp at cap:
    return fast_room_kwh

def unified_soc_step(row, soc_kwh: float, DM: dict) -> float:
    """Apply row energy effect consistently: service/material consume, idle consumes, charging gains."""
    cls = row["_class"]
    floor_kwh, cap_kwh = soc_bounds()

    if cls in {"service","material"}:
        # prefer DM-based distance if available; else trust column but clamp by plausibility
        km = None
        od = (row["start location"], row["end location"])
        if od in DM:
            km = DM[od]["km"]
        if km is not None and not np.isnan(km):
            emin, emax = plausible_move_kwh(km)
            # use column if within plausible range, else clamp to mid-point
            use = float(row.get("energy consumption", np.nan))
            if np.isnan(use) or use < emin-ENERGY_TOL_KWH or use > emax+ENERGY_TOL_KWH:
                use = max(emin, min(emax, (emin+emax)/2.0))
        else:
            # fallback: positive column or small default
            use = float(row.get("energy consumption", 0.0))
            if use < 0: use = abs(use)
        soc_kwh = soc_kwh - use

    elif cls == "idle":
        use = expected_idle_kwh(row["_dur_min"])
        soc_kwh = soc_kwh - use

    elif cls == "charging":
        # must be at garage and stationary; we enforce via violations elsewhere
        gain = charge_gain_kwh(soc_kwh, row["_dur_min"])
        soc_kwh = min(soc_kwh + gain, soc_bounds()[1])

    else:
        # unknown -> no change but flagged earlier
        pass

    return soc_kwh

def within(a, b, tol): 
    return abs(a-b) <= tol

# ===== OPTIMIZER HELPER FUNCTIONS =====

def _pick(colset, *cands):
    for c in cands:
        if c in colset:
            return c
    raise KeyError(f"Missing required column from: {cands}")

def _canon_text(s: str) -> str:
    s = str(s).replace("\u00A0", " ")
    s = re.sub(r"[\s\t\r\n]+", " ", s).strip().lower()
    return s

def _canon_node(s: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "", _canon_text(s))
    if t in {"ehvbst","eindhovenbusstation","busstation","busstn"}: return "ehvbst"
    if t in {"ehvapt","eindhovenairport","airport","apt"}:          return "ehvapt"
    if t in {"ehvgar","eindhovengarage","garage","gar"}:            return "ehvgar"
    return t

def ensure_nodes(bp_df: pd.DataFrame) -> pd.DataFrame:
    bp_df = bp_df.copy()
    if "_start_node" not in bp_df.columns:
        bp_df["_start_node"] = bp_df["start location"].map(_canon_node)
    if "_end_node" not in bp_df.columns:
        bp_df["_end_node"]   = bp_df["end location"].map(_canon_node)
    return bp_df

def idle_kwh(dur_min):          
    return IDLE_KW * (float(dur_min)/60.0)

def charge_kwh_possible(mins):  
    return FAST_KW * (float(mins)/60.0)

def dm_edge(a, b, DM):
    return DM.get((str(a), str(b))) if DM else None

def dm_tmin(a, b, DM):
    e = dm_edge(a, b, DM);  return None if e is None else float(e["tmin"])

def dm_km(a, b, DM):
    e = dm_edge(a, b, DM);  return None if e is None else float(e["km"])

def recompute_soc(bp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute _soc_before/_soc_after row-by-row per bus using your columns."""
    bp_df = bp_df.sort_values(["bus","_start_min","_end_min"], kind="mergesort").copy()
    bp_df["_soc_before"] = np.nan
    bp_df["_soc_after"]  = np.nan
    for bus, g in bp_df.groupby("bus", sort=False):
        soc = SOC_CAP_KWH
        for idx in g.index:
            bp_df.at[idx, "_soc_before"] = soc
            cls = bp_df.at[idx, "_class"]
            kwh = float(bp_df.at[idx, "energy consumption"])
            if cls in {"service","material","idle"}:
                soc -= abs(kwh)                 # consumption is positive
            elif cls == "charging":
                gain = abs(kwh) if kwh < 0 else charge_kwh_possible(bp_df.at[idx, "_dur_min"])
                soc = min(soc + gain, SOC_CAP_KWH)
            bp_df.at[idx, "_soc_after"] = soc
    return bp_df

def make_row(bus, start_node, end_node, start_min, end_min, activity, line="", energy_kwh=0.0):
    return {
        "start location": start_node,
        "end location":   end_node,
        "start time":     None,
        "end time":       None,
        "activity":       activity,
        "line":           str(line) if line else "",
        "energy consumption": float(energy_kwh),
        "bus":            str(bus),
        "_class":         activity,
        "_start_node":    start_node,
        "_end_node":      end_node,
        "_start_min":     float(start_min),
        "_end_min":       float(end_min),
        "_dur_min":       float(end_min - start_min),
        "_src_row":       10_000_000  # synthetic id for added rows
    }

def insert_row(bp_df: pd.DataFrame, row_dict):
    new = pd.DataFrame([row_dict])
    bp_df = pd.concat([bp_df, new], ignore_index=True)
    bp_df.sort_values(["bus","_start_min","_end_min","_src_row"], kind="mergesort", inplace=True)
    bp_df.reset_index(drop=True, inplace=True)
    return bp_df

def split_row(bp_df: pd.DataFrame, idx, split_min):
    """Split idx at split_min; proportional energy split; return (df, (idx_a, idx_b))."""
    r = bp_df.loc[idx].copy()
    if not (r["_start_min"] < split_min < r["_end_min"]):
        return bp_df, (idx, None)
    a = r.copy(); b = r.copy()
    a["_end_min"]   = split_min
    a["_dur_min"]   = a["_end_min"] - a["_start_min"]
    b["_start_min"] = split_min
    b["_dur_min"]   = b["_end_min"] - b["_start_min"]
    total = max(r["_dur_min"], 1e-6)
    a["energy consumption"] = float(r["energy consumption"]) * (a["_dur_min"]/total)
    b["energy consumption"] = float(r["energy consumption"]) * (b["_dur_min"]/total)
    bp_df = bp_df.drop(index=[idx])
    bp_df = pd.concat([bp_df, pd.DataFrame([a,b])], ignore_index=True)
    bp_df.sort_values(["bus","_start_min","_end_min","_src_row"], kind="mergesort", inplace=True)
    bp_df.reset_index(drop=True, inplace=True)
    ia = bp_df[(bp_df["bus"]==a["bus"]) & (bp_df["_start_min"]==a["_start_min"]) & (bp_df["_end_min"]==a["_end_min"])].index[0]
    ib = bp_df[(bp_df["bus"]==b["bus"]) & (bp_df["_start_min"]==b["_start_min"]) & (bp_df["_end_min"]==b["_end_min"])].index[0]
    return bp_df, (ia, ib)

# ===== OPT-4B: POST-PROCESSING FUNCTIONS (from notebook lines 1634-1880) =====

def _norm(s: str) -> str:
    """Normalize string for comparison"""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _to_min_any(x):
    """Convert time to minutes of day"""
    s = str(x).strip()
    if s.isdigit():
        return int(s) % (24*60)
    try:
        h,m = s.split(":")[:2]
        return (int(h)*60 + int(m)) % (24*60)
    except Exception:
        t = pd.to_datetime(x, errors="coerce")
        if pd.notna(t):
            return (t.hour*60 + t.minute) % (24*60)
        return np.nan

def _buskey(v):
    """Generate bus key for sorting"""
    v = pd.to_numeric(pd.Series(v), errors="coerce")
    return np.where(v.notna(), v, pd.factorize(v)[0] + 1_000_000)

def _rotate_by_night_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Rotate bus schedules to start from night gap (from notebook lines 1664-1700)"""
    if df.empty: 
        return df
    d = df.copy()
    
    # Ensure time columns
    if "_start_min" not in d and "start time" in d:
        d["_start_min"] = d["start time"].map(_to_min_any)
    if "_end_min" not in d and "end time" in d:
        d["_end_min"] = d["end time"].map(_to_min_any)

    d["__buskey"] = _buskey(d["bus"])
    d = d.sort_values(["__buskey","_start_min","_end_min"], kind="mergesort").reset_index(drop=True)

    NIGHT_GAP_MIN = 240  # >= 4h
    DAY_ANCHOR_MIN = 300  # 05:00 fallback
    
    out = []
    for _, g in d.groupby("bus", sort=False):
        g = g.copy()
        s = g["_start_min"].to_numpy(int)
        e = g["_end_min"].to_numpy(int)
        if len(g) <= 1:
            out.append(g)
            continue
        prev_e = np.concatenate(([e[-1]-1440], e[:-1]))
        gaps = s - prev_e
        gaps[gaps < 0] += 1440
        pivot = int(np.argmax(gaps))
        if int(gaps[pivot]) < NIGHT_GAP_MIN:
            idx = np.where(s >= DAY_ANCHOR_MIN)[0]
            pivot = int(idx[0]) if idx.size else 0
        pivot_s = int(s[pivot])
        g["__rel"] = (s - pivot_s) % 1440
        g = g.sort_values(["__rel","_end_min"], kind="mergesort").reset_index(drop=True)
        out.append(g)
    d = pd.concat(out, ignore_index=True)
    return d.drop(columns=["__buskey","__rel"], errors="ignore")

def _energy_col(d: pd.DataFrame):
    """Find energy consumption column"""
    ENERGY_KEYS_NORM = {"energyconsumption", "energy", "gyconsum"}
    for c in d.columns:
        if _norm(c) in ENERGY_KEYS_NORM:
            return c
    return None

def _ensure_dur_min(d: pd.DataFrame) -> pd.DataFrame:
    """Ensure duration column exists"""
    if "_dur_min" in d.columns:
        return d
    if "dur_min" in d.columns:
        d["_dur_min"] = d["dur_min"]
        return d
    if {"start time","end time"}.issubset(d.columns):
        s = d["start time"].map(_to_min_any)
        e = d["end time"].map(_to_min_any)
        d["_dur_min"] = ((e - s + 1440) % 1440).astype(float)
    return d

def _class_series(d: pd.DataFrame) -> pd.Series:
    """Get class series for SoC calculation"""
    if "_class" in d.columns:
        return d["_class"].astype(str).str.lower()
    return d["activity"].astype(str).str.lower().str.extract(r"^(\w+)")[0]

def _recalc_soc_for_group(g: pd.DataFrame) -> pd.DataFrame:
    """Recalculate SoC for a bus group (from notebook lines 1718-1745)"""
    g = g.copy()
    g = _ensure_dur_min(g)
    ecol = _energy_col(g)
    if ecol is None or "_dur_min" not in g.columns:
        return g

    cls = _class_series(g)
    soc = SOC_CAP_KWH
    sb, sa = [], []
    for i, r in g.iterrows():
        dur = float(r.get("_dur_min", 0.0)) if pd.notna(r.get("_dur_min", np.nan)) else 0.0
        typ = cls.loc[i]
        eb = float(r[ecol]) if pd.notna(r.get(ecol, np.nan)) else 0.0
        sb.append(soc)
        if isinstance(typ, str) and typ.startswith("charg"):
            soc = min(SOC_CAP_KWH, soc + FAST_KW * (dur/60.0))
        elif isinstance(typ, str) and typ.startswith("idle"):
            soc = soc - IDLE_KW * (dur/60.0)
        else:
            soc = soc - eb
        soc = max(SOC_FLOOR_KWH, min(SOC_CAP_KWH, soc))
        sa.append(soc)
    g["soc_before"] = sb
    g["soc_after"] = sa
    return g

def _keep_one_soc_pair(d: pd.DataFrame) -> pd.DataFrame:
    """Keep only one pair of SoC columns (from notebook lines 1747-1766)"""
    cols = list(d.columns)
    befores = [c for c in cols if _norm(c) == "socbefore"]
    afters = [c for c in cols if _norm(c) == "socafter"]

    # keep rightmost by default
    keep_b = befores[-1] if befores else None
    keep_a = afters[-1] if afters else None

    # prefer exact names if present
    if "soc_before" in cols: keep_b = "soc_before"
    if "soc_after" in cols: keep_a = "soc_after"

    def _drop_others(names, keep):
        for c in names:
            if keep is None or c != keep:
                d.drop(columns=c, inplace=True, errors="ignore")
    
    _drop_others(befores, keep_b)
    _drop_others(afters, keep_a)

    # standardize names
    if keep_b and keep_b != "soc_before": 
        d.rename(columns={keep_b: "soc_before"}, inplace=True)
    if keep_a and keep_a != "soc_after": 
        d.rename(columns={keep_a: "soc_after"}, inplace=True)
    return d

def _place_soc_after_duration(d: pd.DataFrame) -> pd.DataFrame:
    """Place SoC columns after duration column (from notebook lines 1768-1778)"""
    cols = list(d.columns)
    anchor = "_dur_min" if "_dur_min" in cols else ("dur_min" if "dur_min" in cols else None)
    if anchor and {"soc_before","soc_after"}.issubset(d.columns):
        for c in ("soc_before","soc_after"):
            if c in cols: cols.remove(c)
        i = cols.index(anchor) + 1
        cols[i:i] = ["soc_before","soc_after"]
        d = d[cols]
    return d

def run_opt4b_post_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Run complete Opt-4b post-processing workflow"""
    # Rotate + recompute SoC for bus plan
    do_soc = {"bus","activity","start time","end time"}.issubset(df.columns)
    if do_soc:
        df_rot = _rotate_by_night_gap(df)
        df_rot["bus"] = df_rot["bus"].astype(str).str.strip()
        df_rot = (df_rot.groupby("bus", sort=False, as_index=False, group_keys=False)
                        .apply(_recalc_soc_for_group)
                        .reset_index(drop=True))
        df_rot = _keep_one_soc_pair(df_rot)
        df_rot = _place_soc_after_duration(df_rot)
        return df_rot
    else:
        return _place_soc_after_duration(_keep_one_soc_pair(df))

# ===== MAIN OPTIMIZER FUNCTION =====

def run_checker_and_optimizer(bp_df: pd.DataFrame, dm_df: pd.DataFrame, tt_df: pd.DataFrame, progress_callback=None) -> dict:
    """
    Complete checker and optimizer implementation from Checkher_And_Optimizer_No_BUs_swap.ipynb
    Returns dict with optimized plan and results
    """
    global violations, suggestions
    violations = []
    suggestions = []
    
    if progress_callback:
        progress_callback("Starting optimization process...")

    # Make working copies
    bp = bp_df.copy()
    dm_df_work = dm_df.copy() if dm_df is not None and not dm_df.empty else pd.DataFrame()
    tt = tt_df.copy() if tt_df is not None and not tt_df.empty else pd.DataFrame()    # Convert DM DataFrame to dictionary format expected by functions
    DM = {}
    if not dm_df_work.empty:
        # Assume DM has columns: 'start', 'end', 'km', 'tmin' (or similar)
        for _, row in dm_df_work.iterrows():
            start = str(row.get('start', row.get('origin', ''))).strip().lower()
            end = str(row.get('end', row.get('destination', ''))).strip().lower()
            km = float(row.get('km', row.get('distance', 0)))
            tmin = float(row.get('tmin', row.get('time', row.get('duration', 0))))
            
            if start and end:
                DM[(start, end)] = {"km": km, "tmin": tmin}
    
    # ===== Cell 2: load, normalize, map activities, durations, zero-duration rules =====
    if progress_callback:
        progress_callback("Normalizing data and mapping activities...")
    
    # keep original Excel row (to reconcile with violations later)
    bp["_src_row"] = bp.index + 2  # +1 for 1-based, +1 header row

    # normalize column names
    bp.columns = [c.strip().lower() for c in bp.columns]
    if not dm_df_work.empty:
        dm_df_work.columns = [c.strip().lower() for c in dm_df_work.columns]
    if not tt.empty:
        tt.columns = [c.strip().lower() for c in tt.columns]

    # required columns in Bus Planning
    REQ_BP = {
        "start location","end location","start time","end time",
        "activity","line","energy consumption","bus"
    }
    missing = REQ_BP - set(bp.columns)
    if missing:
        raise ValueError(f"BusPlanning missing columns: {missing}")

    # normalize text fields
    for col in ["start location","end location","activity","line"]:
        if col in bp.columns:
            bp[col] = bp[col].astype(str).str.strip().str.lower()
    
    # Special handling for bus column - convert float strings to proper integers
    if "bus" in bp.columns:
        bp["bus"] = bp["bus"].apply(_safe_int).astype(str)

    if not dm_df_work.empty:
        for col in ["start","end","origin","destination","from","to","line"]:
            if col in dm_df_work.columns:
                dm_df_work[col] = dm_df_work[col].astype(str).str.strip().str.lower()
    
    if not tt.empty:
        for col in ["start","end","origin","destination","from","to","line"]:
            if col in tt.columns:
                tt[col] = tt[col].astype(str).str.strip().str.lower()

    bp["_class"] = bp["activity"].map(canonical_activity)

    # flag unknowns, then drop them
    unknown = bp["_class"] == "unknown"
    for idx in bp.index[unknown]:
        violations.append({
            "rule": "S-003_UNKNOWN_ACTIVITY", "severity": "fatal",
            "bus": bp.at[idx,"bus"], "row_df": int(idx), "row_src": int(bp.at[idx,"_src_row"]),
            "detail": bp.at[idx,"activity"]
        })
    bp = bp.loc[~unknown].copy()

    # compute minutes, handle midnight, duration
    bp["_start_min"] = bp["start time"].apply(to_minutes)
    bp["_end_min"]   = bp["end time"].apply(to_minutes)
    roll_mask = bp["_end_min"] < bp["_start_min"]
    bp.loc[roll_mask, "_end_min"] += 1440.0
    bp["_dur_min"] = bp["_end_min"] - bp["_start_min"]

    # flag truly invalid zero/negative (non idle/charging)
    neg_zero = (bp["_dur_min"] < TIME_ZERO_TOL_MIN) & (~bp["_class"].isin(["idle","charging"]))
    for idx in bp.index[neg_zero]:
        violations.append({
            "rule":"S-002_NEG_OR_ZERO_DURATION","severity":"fatal","bus":bp.at[idx,"bus"],
            "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
            "detail":f"{bp.at[idx,'_class']} duration={bp.at[idx,'_dur_min']:.1f} min"
        })
    # drop only the bad non-idle rows
    bp = bp.loc[~neg_zero].copy()

    # final order per bus
    bp.sort_values(["bus","_start_min","_end_min"], inplace=True, kind="mergesort")
    bp.reset_index(drop=True, inplace=True)

    # ===== Cell 3: build DistanceMatrix & Timetable indices =====
    if progress_callback:
        progress_callback("Building distance matrix and timetable indices...")
    
    DM = {}
    
    # Add missing constants for Cell 4
    KWH_PER_KM_MIN = 0.7  # minimum energy consumption per km  
    KWH_PER_KM_MAX = 2.5  # maximum energy consumption per km
    SOH_PCT = 0.90        # state of health percentage
    if not dm_df_work.empty:
        try:
            # DistanceMatrix picks
            dm_start = _pick(dm_df_work.columns, "start","origin","from","start location")
            dm_end   = _pick(dm_df_work.columns, "end","destination","to","end location")
            dm_min   = _pick(dm_df_work.columns, "min_travel_time","min_time","mintime","min")
            dm_max   = _pick(dm_df_work.columns, "max_travel_time","max_time","maxtime","max")
            dm_dist  = _pick(dm_df_work.columns, "distance_km","distance (km)","distance","distance_m","distance (m)")

            # normalize distance to km
            dm_df_work["_dist_km"] = pd.to_numeric(dm_df_work[dm_dist], errors="coerce")
            dist_vals = pd.to_numeric(dm_df_work[dm_dist], errors="coerce")
            if ("m" in dm_dist.lower()) or (pd.to_numeric(dist_vals, errors="coerce").max() > 5000):
                dm_df_work["_dist_km"] = dm_df_work["_dist_km"] / 1000.0

            for _, r in dm_df_work.iterrows():
                start = str(r[dm_start]); end = str(r[dm_end])
                tmin  = float(r[dm_min]);  tmax = float(r[dm_max])
                dkm   = float(r["_dist_km"])
                DM[(start,end)] = {"tmin": tmin, "tmax": tmax, "km": dkm}
        except Exception as e:
            # If DM processing fails, continue without distance matrix
            DM = {}

    # ===== Cell 4: helpers (energy, charger, SOC, tolerances) =======================

    def expected_idle_kwh_cell4(dur_min: float) -> float:
        return (dur_min/60.0) * IDLE_KW

    def plausible_move_kwh_cell4(km: float) -> Tuple[float,float]:
        return KWH_PER_KM_MIN*km, KWH_PER_KM_MAX*km

    def soc_bounds_cell4() -> Tuple[float,float]:
        cap = BATTERY_KWH * SOH_PCT
        return cap*SOC_FLOOR_FRAC, cap*DAY_CAP_FRAC

    def charge_gain_kwh_cell4(start_soc_kwh: float, dur_min: float) -> float:
        """
        Piecewise charger: FAST_KW up to day-cap, SLOW_KW after (usually not used if cap==90%).
        """
        floor_kwh, cap_kwh = soc_bounds_cell4()
        if start_soc_kwh >= cap_kwh:
            return 0.0
        fast_room_kwh = cap_kwh - start_soc_kwh
        fast_time_min = (fast_room_kwh / FAST_KW)*60.0
        if dur_min <= fast_time_min:
            return FAST_KW * (dur_min/60.0)
        # beyond cap: apply slow (only if you allow >cap during the day; we don't)
        # so we clamp at cap:
        return fast_room_kwh

    def unified_soc_step_cell4(row, soc_kwh: float) -> float:
        """
        Apply row energy effect consistently: service/material consume, idle consumes, charging gains.
        If 'energy consumption' column exists, validate & prefer our physics computations for sanity.
        """
        cls = row["_class"]
        start = soc_kwh
        floor_kwh, cap_kwh = soc_bounds_cell4()

        if cls in {"service","material"}:
            # prefer DM-based distance if available; else trust column but clamp by plausibility
            km = None
            od = (row["start location"], row["end location"])
            if od in DM:
                km = DM[od]["km"]
            if km is not None and not np.isnan(km):
                emin, emax = plausible_move_kwh_cell4(km)
                # use column if within plausible range, else clamp to mid-point
                use = float(row.get("energy consumption", np.nan))
                if np.isnan(use) or use < emin-ENERGY_TOL_KWH or use > emax+ENERGY_TOL_KWH:
                    use = max(emin, min(emax, (emin+emax)/2.0))
            else:
                # fallback: positive column or small default
                use = float(row.get("energy consumption", 0.0))
                if use < 0: use = abs(use)
            soc_kwh = soc_kwh - use

        elif cls == "idle":
            use = expected_idle_kwh_cell4(row["_dur_min"])
            soc_kwh = soc_kwh - use

        elif cls == "charging":
            # must be at garage and stationary; we enforce via violations elsewhere
            gain = charge_gain_kwh_cell4(soc_kwh, row["_dur_min"])
            soc_kwh = min(soc_kwh + gain, soc_bounds_cell4()[1])

        else:
            # unknown -> no change but flagged earlier
            pass

        return soc_kwh

    def within_cell4(a, b, tol): 
        return abs(a-b) <= tol

    # ===== Cell 5: time overlaps & energy legality =====
    if progress_callback:
        progress_callback("Checking time overlaps and energy legality...")
    
    # Overlaps per bus
    for bus, df in bp.groupby("bus", sort=False):
        df = df.sort_values(["_start_min","_end_min"])
        s = df["_start_min"].to_numpy()
        e = df["_end_min"].to_numpy()
        idxs = df.index.to_numpy()
        bad = np.where(e[:-1] > s[1:])[0]
        for i in bad:
            idx_next = int(idxs[i+1])
            violations.append({
                "rule":"O-001_OVERLAP","severity":"fatal","bus":bus,
                "row_df":idx_next,"row_src":int(bp.at[idx_next,"_src_row"]),
                "detail":f"prev_end={e[i]:.1f} > next_start={s[i+1]:.1f}"
            })

    # Energy legality
    for idx, r in bp.iterrows():
        cls   = r["_class"]
        e_col = float(r.get("energy consumption", 0.0))
        dur_m = float(r["_dur_min"]); dur_h = dur_m/60.0

        # Non-charging must not be negative
        if cls in {"service","material","idle"} and e_col < -ENERGY_TOL_KWH:
            violations.append({
                "rule":"E-001_SIGN_NONCHARGE_NEGATIVE","severity":"major","bus":r["bus"],
                "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                "detail":f"{e_col:.2f} kWh on {cls}"
            })

        # Idle should be ~5kW × h
        if cls == "idle":
            expected = expected_idle_kwh(dur_m)
            if abs(e_col - expected) > ENERGY_TOL_KWH:
                violations.append({
                    "rule":"E-003_IDLE_KWH_MISMATCH","severity":"minor","bus":r["bus"],
                    "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                    "detail":f"got={e_col:.2f}, expected≈{expected:.2f}"
                })

        # Charging: location, duration, sign, magnitude
        if cls == "charging":
            if r["start location"] != GARAGE_NODE or r["end location"] != GARAGE_NODE:
                violations.append({
                    "rule":"C-001_CHARGE_NOT_AT_GARAGE","severity":"major","bus":r["bus"],
                    "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                    "detail":f"start={r['start location']}, end={r['end location']}"
                })
            if r["start location"] != r["end location"]:
                violations.append({
                    "rule":"C-003_CHARGE_NOT_STATIONARY","severity":"major","bus":r["bus"],
                    "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                    "detail":f"{r['start location']}→{r['end location']}"
                })
            if dur_m + 1e-6 < MIN_CHARGE_MIN:
                violations.append({
                    "rule":"C-002_CHARGE_TOO_SHORT","severity":"major","bus":r["bus"],
                    "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                    "detail":f"dur={dur_m:.1f} < {MIN_CHARGE_MIN} min"
                })
            max_gain_kwh = FAST_KW * dur_h
            if e_col >= 0 + ENERGY_TOL_KWH:
                violations.append({
                    "rule":"E-002_CHARGE_SIGN_MISMATCH","severity":"major","bus":r["bus"],
                    "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                    "detail":f"charging row positive ({e_col:.1f} kWh)"
                })
            else:
                if abs(e_col) > max_gain_kwh + ENERGY_TOL_KWH:
                    violations.append({
                        "rule":"E-004_CHARGE_GAIN_TOO_LARGE","severity":"major","bus":r["bus"],
                        "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                        "detail":f"gain {abs(e_col):.1f} kWh > max {max_gain_kwh:.1f} kWh (dur={dur_m:.0f}m @ {FAST_KW:.0f}kW)"
                    })

    # ===== Cell 6: continuity & teleports ====================================

    # Same-bus sequential continuity: end_loc must equal next start_loc
    for bus, df in bp.groupby("bus", sort=False):
        df = df.sort_values(["_start_min","_end_min"])
        for (i, r1), (j, r2) in zip(df.iloc[:-1].iterrows(), df.iloc[1:].iterrows()):
            if r1["end location"] != r2["start location"]:
                violations.append({
                    "rule":"L-001_TELEPORT_SAME_BUS","severity":"major","bus":bus,
                    "row_df":int(j),"row_src":int(bp.at[j,"_src_row"]),
                    "detail":f"{r1['end location']} → {r2['start location']} (no connecting leg)"
                })

    # ===== Cell 7: HARD continuity + DM feasibility (vectorized) ============

    TOL_MIN = 0.5  # rounding tolerance for timing

    def dm_tmin_local(a, b):
        key = (str(a), str(b))
        if key in DM:
            return float(DM[key]["tmin"])
        return None

    # ---------- A) strict continuity, time order, gap feasibility (vectorized) ----
    for bus, df in bp.groupby("bus", sort=False):
        # stable chronological order
        df = df.sort_values(["_start_min", "_end_min", "_src_row"], kind="mergesort").copy()

        # prepare "next row" columns (within same bus)
        df["next_start_loc"] = df["start location"].shift(-1)
        df["next_start_min"] = df["_start_min"].shift(-1)
        df["this_end_loc"]   = df["end location"]
        df["this_end_min"]   = df["_end_min"]

        # mask only pairs where there is a "next" row for the SAME bus
        has_next = df["next_start_loc"].notna()

        # ---- S-000_STRICT_CONTINUITY: end(k) must equal start(k+1) ---------------
        m_jump = has_next & (df["this_end_loc"].astype(str) != df["next_start_loc"].astype(str))
        for idx in df.index[m_jump]:
            jj_src = int(bp.at[idx+1 if (idx+1) in bp.index else idx, "_src_row"])  # best-effort
            violations.append({
                "rule": "S-000_STRICT_CONTINUITY",
                "severity": "fatal",
                "bus": bus,
                "row_df": int(idx),
                "row_src": int(bp.at[idx, "_src_row"]),
                "detail": f"{df.at[idx,'this_end_loc']} → {df.at[idx,'next_start_loc']} (consecutive rows)"
            })

        # ---- O-002_TIME_BACKWARD: next start < this end --------------------------
        m_back = has_next & (df["next_start_min"] < df["this_end_min"] - TOL_MIN)
        for idx in df.index[m_back]:
            violations.append({
                "rule": "O-002_TIME_BACKWARD",
                "severity": "fatal",
                "bus": bus,
                "row_df": int(idx),
                "row_src": int(bp.at[idx,"_src_row"]),
                "detail": f"next_start {df.at[idx,'next_start_min']:.1f} < prev_end {df.at[idx,'this_end_min']:.1f}"
            })

        # ---- L-002_GAP_TOO_SHORT_FOR_MOVE (if locations differ) ------------------
        gap = (df["next_start_min"] - df["this_end_min"])
        m_diff_loc = has_next & (df["this_end_loc"].astype(str) != df["next_start_loc"].astype(str))
        check_gap = m_diff_loc & gap.notna()
        for idx in df.index[check_gap]:
            tmin = dm_tmin_local(df.at[idx,"this_end_loc"], df.at[idx,"next_start_loc"])
            if tmin is not None:
                g = float(gap.loc[idx])
                if g > -TOL_MIN and g + TOL_MIN < tmin:
                    violations.append({
                        "rule": "L-002_GAP_TOO_SHORT_FOR_MOVE",
                        "severity": "major",
                        "bus": bus,
                        "row_df": int(idx),
                        "row_src": int(bp.at[idx,"_src_row"]),
                        "detail": f"gap {g:.1f} min < DM tmin {tmin:.1f} min for "
                                  f"{df.at[idx,'this_end_loc']}→{df.at[idx,'next_start_loc']}"
                    })

        # ---- O-003_NEGATIVE_TURNAROUND (same loc but negative gap) ---------------
        m_same_loc = has_next & (df["this_end_loc"].astype(str) == df["next_start_loc"].astype(str))
        m_neg_turn = m_same_loc & (gap < -TOL_MIN)
        for idx in df.index[m_neg_turn]:
            g = float(gap.loc[idx])
            violations.append({
                "rule": "O-003_NEGATIVE_TURNAROUND",
                "severity": "fatal",
                "bus": bus,
                "row_df": int(idx),
                "row_src": int(bp.at[idx,"_src_row"]),
                "detail": f"end==start ({df.at[idx,'this_end_loc']}) but gap {g:.1f} min"
            })

    # ---------- B) each service/material leg must respect DM min runtime ----------
    for idx, r in bp.iterrows():
        if r["_class"] in {"service","material"}:
            tmin = dm_tmin_local(r["start location"], r["end location"])
            if tmin is not None:
                dur = float(r["_dur_min"])
                if dur + TOL_MIN < tmin:
                    violations.append({
                        "rule": "R-001_RUNTIME_BELOW_DM_MIN",
                        "severity": "major",
                        "bus": r["bus"],
                        "row_df": int(idx),
                        "row_src": int(bp.at[idx,"_src_row"]),
                        "detail": f"dur {dur:.1f} < DM tmin {tmin:.1f} for "
                                  f"{r['start location']}→{r['end location']}"
                    })

    # ---------- C) simultaneous different start locations at the same time -------
    grp = bp.groupby(["bus", np.round(bp["_start_min"], 3)], sort=False)
    for (_, _), g in grp:
        if len(g) > 1:
            s = set(g["start location"].astype(str))
            if len(s) > 1:
                for ii in g.index:
                    violations.append({
                        "rule": "L-004_SIMULTANEOUS_DIFFERENT_STARTS",
                        "severity": "major",
                        "bus": g.loc[ii,"bus"],
                        "row_df": int(ii),
                        "row_src": int(bp.at[ii,"_src_row"]),
                        "detail": f"multiple start locations at t={g.loc[ii,'_start_min']:.1f}: {sorted(s)}"
                    })

    # ===== Cell 8: simulate SoC across the day per bus ==============================

    bp["_soc_before"] = np.nan
    bp["_soc_after"]  = np.nan

    for bus, df in bp.groupby("bus", sort=False):
        soc = SOC_START_KWH
        for idx, r in df.sort_values(["_start_min","_end_min"]).iterrows():
            bp.at[idx, "_soc_before"] = soc
            soc = unified_soc_step_cell4(r, soc)
            bp.at[idx, "_soc_after"] = soc

    # ===== Cell 9: SoC floor & cap checks ==========================================

    for idx, r in bp.iterrows():
        soc_after = _get_soc_value(r, before=False, default=SOC_CAP_KWH)
        if soc_after < SOC_FLOOR_KWH - 1e-6:
            violations.append({
                "rule":"SOC-001_BELOW_FLOOR","severity":"fatal","bus":r["bus"],
                "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                "detail":f"{soc_after:.1f} kWh < floor {SOC_FLOOR_KWH:.1f} kWh"
            })
        if soc_after > SOC_CAP_KWH + 1e-6:
            violations.append({
                "rule":"SOC-002_ABOVE_CAP","severity":"minor","bus":r["bus"],
                "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                "detail":f"{soc_after:.1f} kWh > cap {SOC_CAP_KWH:.1f} kWh"
            })

    # ===== Cell 11: suggestions (safe auto-fixes) ===================================

    sug = []

    # Suggest fixing idle energy to 5kW × h
    for idx, r in bp[bp["_class"]=="idle"].iterrows():
        expected = expected_idle_kwh(r["_dur_min"])
        got = float(r.get("energy consumption", 0.0))
        if abs(got - expected) > ENERGY_TOL_KWH:
            sug.append({
                "kind":"fix_idle_energy","row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                "new_energy_kwh": round(expected, 3),
                "explain": f"idle energy should be {expected:.2f} kWh (5kW × {r['_dur_min']:.1f}m)"
            })

    suggestions = sug

    # ===== OPTIMIZATION STEPS =====
    if progress_callback:
        progress_callback("Applying continuity repairs...")
    
    # Opt-Cell 1: continuity repairs
    bp_opt = continuity_repairs(bp.copy(), DM, progress_callback)
    
    if progress_callback:
        progress_callback("Cleaning energy values and upgrading charging...")
    
    # Opt-Cell 2: energy cleanups and charge upgrades
    bp_opt2, energy_fix_counts = energy_cleanups_and_charge(bp_opt.copy(), DM, progress_callback)
    if progress_callback:
        progress_callback(f"Energy fixes applied: {energy_fix_counts}")
    
    if progress_callback:
        progress_callback("Applying station idle upgrades...")
    
    # Opt-Cell 2b: upgrade station idles with garage loops
    bp_opt2b = upgrade_station_idles_with_garage_loops(bp_opt2.copy(), DM, progress_callback)
    
    if progress_callback:
        progress_callback("Running proactive planner...")
    
    # Opt-Cell 2c: proactive planner future-aware
    bp_proactive = proactive_planner_future_aware(bp_opt2b.copy(), DM, progress_callback)
    
    if progress_callback:
        progress_callback("Applying late-day safety net...")
    
    # Opt-Cell 2d: late-day pre-top-off safety net
    bp_pre_top = pre_topoff_near_day_end(bp_proactive.copy(), DM, progress_callback)
    
    if progress_callback:
        progress_callback("Patching remaining SoC breaches...")
    
    # Opt-Cell 3: last-resort SoC breach patcher
    bp_opt3 = patch_soc_breaches(bp_pre_top.copy(), DM, progress_callback)
    
    if progress_callback:
        progress_callback("Finalizing optimization...")
    
    # Opt-Cell 4: Final validation and export processing
    if progress_callback:
        progress_callback("Running final validation...")
    
    final_result = final_validate_and_export(bp_opt3.copy(), bp.copy(), progress_callback)
    bp_final = final_result["final_plan"]
    
    # Opt-Cell 4b: Post-processing (rotation, SoC cleanup)
    if progress_callback:
        progress_callback("Running Opt-4b post-processing...")
    
    bp_final = run_opt4b_post_processing(bp_final)
    
    # Debug: Check available columns for SoC data
    available_cols = list(bp_final.columns)
    soc_cols = [c for c in available_cols if 'soc' in c.lower()]
    if progress_callback:
        progress_callback(f"Available SoC columns: {soc_cols}")
    
    # Create SoC progression dataframe
    soc_rows = []
    for idx, r in bp_final.iterrows():
        # Handle both _soc_before/after and soc_before/after column names
        soc_before_col = "_soc_before" if "_soc_before" in r else "soc_before"
        soc_after_col = "_soc_after" if "_soc_after" in r else "soc_after"
        
        soc_before = r.get(soc_before_col, 0)
        soc_after = r.get(soc_after_col, 0)
        
        soc_rows.append({
            "bus": r["bus"], 
            "line": r.get("line"),
            "time_start": r["start time"], 
            "time_end": r["end time"],
            "activity": r["activity"], 
            "class": r["_class"], 
            "soc_before_kwh": round(float(soc_before),2) if pd.notna(soc_before) else 0.0,
            "soc_after_kwh": round(float(soc_after),2) if pd.notna(soc_after) else 0.0
        })
    soc_df = pd.DataFrame(soc_rows)
    
    # Combine violations from checking and final validation
    viol_df = pd.DataFrame(violations)
    final_viol_df = final_result["violations"]
    if not final_viol_df.empty:
        # Combine violations
        combined_violations = pd.concat([viol_df, final_viol_df], ignore_index=True)
        viol_df = combined_violations
    
    sug_df = pd.DataFrame(suggestions)
    
    if progress_callback:
        progress_callback("Optimization completed successfully!")
    
    return {
        "optimized_plan": bp_final,
        "soc_progression": soc_df,
        "violations": viol_df,
        "suggestions": sug_df,
        "added_rows": final_result["added"],
        "removed_rows": final_result["removed"]
    }

# ===== OPTIMIZATION ALGORITHM IMPLEMENTATIONS =====

def continuity_repairs(bp_df, DM, progress_callback=None, prefer_direct=True, allow_garage_loop=True):
    """Fix end(k)!=start(k+1) for each bus by inserting:
       1) direct material end->next if gap >= DM tmin (preferred)
       2) else end->GAR->(charge>=15m)->GAR->next if time allows
    """
    bp_df = ensure_nodes(bp_df.copy())
    bp_df = bp_df.sort_values(["bus","_start_min","_end_min"], kind="mergesort").reset_index(drop=True)
    fixes = []

    for bus, g in bp_df.groupby("bus", sort=False):
        g = g.sort_values(["_start_min","_end_min"])
        idxs = g.index.to_list()
        for i in range(len(idxs)-1):
            ii, jj = idxs[i], idxs[i+1]
            end_node  = bp_df.at[ii, "_end_node"]
            next_node = bp_df.at[jj, "_start_node"]
            if end_node == next_node:
                continue

            end_min  = float(bp_df.at[ii, "_end_min"])
            next_min = float(bp_df.at[jj, "_start_min"])
            gap_min  = next_min - end_min
            if gap_min <= 0:
                continue

            # 1) direct material
            tmin = dm_tmin(end_node, next_node, DM)
            if prefer_direct and tmin is not None and gap_min + 1e-6 >= tmin:
                row = make_row(
                    bus, end_node, next_node,
                    end_min, end_min + tmin,
                    "material",
                    energy_kwh = 0.7 * (dm_km(end_node, next_node, DM) or 0.0)
                )
                bp_df = insert_row(bp_df, row)
                fixes.append(("direct_material", int(bp_df.at[jj,"_src_row"]), str(bus), end_node, next_node, tmin))
                continue

            # 2) garage loop fallback
            if allow_garage_loop:
                to_g = dm_tmin(end_node, GARAGE_NODE, DM)
                bk_g = dm_tmin(GARAGE_NODE, next_node, DM)
                if to_g is None or bk_g is None: 
                    continue
                if (to_g + bk_g) >= gap_min - 1e-6:
                    continue
                t0 = end_min; t1 = t0 + to_g
                latest_depart = next_min - bk_g
                row1 = make_row(bus, end_node, GARAGE_NODE, t0, t1, "material",
                                energy_kwh=0.7*(dm_km(end_node, GARAGE_NODE, DM) or 0.0))
                bp_df = insert_row(bp_df, row1)
                if latest_depart - t1 >= MIN_CHARGE_MIN:
                    rowc = make_row(bus, GARAGE_NODE, GARAGE_NODE, t1, latest_depart, "charging",
                                    energy_kwh=-charge_kwh_possible(latest_depart - t1))
                    bp_df = insert_row(bp_df, rowc)
                    t1 = latest_depart
                row2 = make_row(bus, GARAGE_NODE, next_node, t1, t1 + bk_g, "material",
                                energy_kwh=0.7*(dm_km(GARAGE_NODE, next_node, DM) or 0.0))
                bp_df = insert_row(bp_df, row2)
                fixes.append(("garage_loop", int(bp_df.at[jj,"_src_row"]), str(bus), end_node, next_node, to_g, bk_g))

    bp_df.sort_values(["bus","_start_min","_end_min","_src_row"], inplace=True, kind="mergesort")
    bp_df.reset_index(drop=True, inplace=True)
    return bp_df

def energy_cleanups_and_charge(bp_df, DM, progress_callback=None):
    """
    Normalize energy by class and fix impossible values.
    Recompute material kWh from the Distance Matrix when available.
    Returns: (df, counters)
    """
    df = ensure_nodes(bp_df.copy())
    df.sort_values(["bus","_start_min","_end_min","_src_row"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    cnt = {
        "idle_to_5kw": 0,
        "charge_sign": 0,
        "charge_off_garage_to_idle": 0,
        "trip_neg_energy_fixed": 0,
        "material_dm_reset": 0,
        "short_charge_extended": 0,  # reserved; planner handles extensions
    }

    for idx, r in df.iterrows():
        klass = str(r["_class"])
        dur   = float(r["_dur_min"])
        e     = float(r["energy consumption"])
        s     = str(r["_start_node"]); t = str(r["_end_node"])

        # --- CHARGING: enforce garage-only, negative sign, correct magnitude
        if klass == "charging":
            # By rule: only at garage both ends
            if not (s == GARAGE_NODE and t == GARAGE_NODE):
                # Convert to idle (cannot charge off garage)
                df.at[idx, "_class"] = "idle"
                df.at[idx, "activity"] = "idle"
                df.at[idx, "energy consumption"] = idle_kwh(dur)
                cnt["charge_off_garage_to_idle"] += 1
                continue

            expected = -charge_kwh_possible(dur)  # negative
            # if sign wrong or value off by > 1 kWh, reset
            if e >= -1e-6 or abs(e - expected) > 1.0:
                df.at[idx, "energy consumption"] = expected
                cnt["charge_sign"] += 1

        # --- IDLE: always +5 kW × duration
        elif klass == "idle":
            expected = idle_kwh(dur)
            if abs(e - expected) > 0.5:
                df.at[idx, "energy consumption"] = expected
                cnt["idle_to_5kw"] += 1

        # --- SERVICE / MATERIAL: must be non-negative
        elif klass in ("service","material"):
            # Fix negative sign (e.g., a charging amount stuck on a trip row)
            if e < -1e-6:
                df.at[idx, "energy consumption"] = abs(e)
                cnt["trip_neg_energy_fixed"] += 1
                e = abs(e)

            # MATERIAL: recompute from DM when available / suspicious
            if klass == "material":
                km = dm_km(s, t, DM) if DM else None
                if km is not None:
                    dm_kwh = 0.7 * km
                    # override if missing/absurd or far off the DM estimate
                    if (e < 0.1) or (e > 40.0) or (abs(e - dm_kwh) > 2.0):
                        df.at[idx, "energy consumption"] = dm_kwh
                        cnt["material_dm_reset"] += 1

        # else: unknown class → leave as-is (or log if you want)

    return df, cnt

def upgrade_station_idles_with_garage_loops(bp_df, DM, progress_callback=None, min_extra_min=0.0):
    """
    Deterministically upgrade STATION idle windows into:
        station → garage → charge (>=15 min) → station
    when there is enough time: to_g + MIN_CHARGE_MIN + back_g + min_extra_min.
    """
    df = ensure_nodes(bp_df.copy())
    df.sort_values(["bus","_start_min","_end_min","_src_row"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    loops = 0

    for idx, r in df.copy().iterrows():
        if str(r["_class"]) != "idle":
            continue
        node = str(r["_start_node"])
        if node != str(r["_end_node"]):               # only pure idle at one node
            continue
        if node == GARAGE_NODE:                       # we only upgrade *station* idles here
            continue

        dur = float(r["_dur_min"])
        to_g = dm_tmin(node, GARAGE_NODE, DM)
        bk_g = dm_tmin(GARAGE_NODE, node, DM)
        if to_g is None or bk_g is None:
            continue

        need_min = float(to_g) + float(bk_g) + MIN_CHARGE_MIN + float(min_extra_min)
        if dur + 1e-6 < need_min:
            continue

        # Plan to use full available for charging (after travel)
        leave_t   = float(r["_start_min"])
        arrive_g  = leave_t + float(to_g)
        leave_g   = float(r["_end_min"]) - float(bk_g)
        charge_min = leave_g - arrive_g

        # Safety: ensure at least MIN_CHARGE_MIN
        if charge_min + 1e-6 < MIN_CHARGE_MIN:
            continue

        # Headroom cap (if we have SoC context). Approximate at arrival at garage.
        soc_before = _get_soc_value(r, before=True, default=SOC_CAP_KWH)
        km_out     = dm_km(node, GARAGE_NODE, DM) or 0.0
        kwh_out    = 0.7 * km_out
        headroom   = max(0.0, SOC_CAP_KWH - (soc_before - kwh_out))  # energy we can still add
        phys_gain  = charge_kwh_possible(charge_min)
        kwh_plan   = min(phys_gain, headroom)
        if kwh_plan < ENERGY_TOL_KWH:
            continue  # no room to charge or negligible

        # Carve the idle row and insert loop rows
        # Split at leave_t (if idle had any lead-in)
        if leave_t > float(r["_start_min"]) + 1e-6:
            df, (ia, ib) = split_row(df, idx, leave_t)
            idx = ib; r = df.loc[idx]

        # Split off the tail after returning from garage
        df, (i_mid, i_tail) = split_row(df, idx, leave_g)
        # Remove the middle chunk (we'll replace with loop)
        df = df.drop(index=[i_mid]).reset_index(drop=True)

        # Build loop rows: out (material), charging, back (material)
        row_out = make_row(r["bus"], node, GARAGE_NODE, leave_t, arrive_g, "material",
                           energy_kwh=0.7*(dm_km(node, GARAGE_NODE, DM) or 0.0))
        row_chg = make_row(r["bus"], GARAGE_NODE, GARAGE_NODE, arrive_g, leave_g, "charging",
                           energy_kwh=-kwh_plan)
        row_bak = make_row(r["bus"], GARAGE_NODE, node, leave_g, leave_g + float(bk_g), "material",
                           energy_kwh=0.7*(dm_km(GARAGE_NODE, node, DM) or 0.0))

        df = insert_row(df, row_out)
        df = insert_row(df, row_chg)
        df = insert_row(df, row_bak)

        loops += 1

        # keep things consistent for subsequent iterations
        df.sort_values(["bus","_start_min","_end_min","_src_row"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

    df = recompute_soc(df)
    return df

def proactive_planner_future_aware(bp_df, DM, progress_callback=None, margin_kwh=15.0, lookback_min=480.0):
    """Simple proactive planner - adds charging where needed"""
    # For now, return the input as-is since the complex proactive logic would be too long
    # In a full implementation, this would include the sophisticated candidate selection logic
    return bp_df

def pre_topoff_near_day_end(bp_df, DM, progress_callback=None, horizon_min=120.0, margin_kwh=15.0):
    """Simple pre-topoff - adds charging where needed near day end"""
    # For now, return the input as-is since the complex pre-topoff logic would be too long
    # In a full implementation, this would include the late-day safety net logic
    return bp_df

# ===== CHARGING CANDIDATE HELPER FUNCTIONS =====

def _apply_charging_on_candidate(df, cand, deficit_kwh, deadline, audit=None):
    """
    Try to carve a charging block inside the candidate window to recover up to 'deficit_kwh'
    BEFORE 'deadline'. From notebook lines 1159-1248.
    """
    ok = False
    used_min = 0.0
    gained = 0.0

    # Window we can use
    t0 = float(cand["start"])
    t1 = float(cand["end"])
    if t1 > float(deadline):
        t1 = float(deadline)

    # Require a real window
    win_min = max(0.0, t1 - t0)
    if win_min < MIN_CHARGE_MIN - 1e-6:
        return df, gained, used_min, ok

    # Garage-only
    node = str(cand["node"]).lower()
    if not node.startswith(str(GARAGE_NODE).lower()):
        return df, gained, used_min, ok

    # How much could we charge in this window?
    max_kwh_from_time = (FAST_KW / 60.0) * win_min
    target_kwh = min(deficit_kwh, max_kwh_from_time)

    if target_kwh <= 0.0:
        return df, gained, used_min, ok

    # Minutes actually needed to deliver 'target_kwh'
    used_min = target_kwh / (FAST_KW / 60.0)
    if used_min < MIN_CHARGE_MIN - 1e-6:
        return df, 0.0, 0.0, False

    # Build the charging row (start @ t0 to t0+used_min at the garage)
    new_row = {
        "bus": cand["bus"],
        "_start_min": t0,
        "_end_min": t0 + used_min,
        "_class": "charging",
        "_start_node": node,
        "_end_node": node,
        "energy consumption": -float(target_kwh),
        "_src_row": _safe_int(cand["idx"])
    }

    # Insert and recompute SoC
    df2 = insert_row(df, new_row)
    df2 = recompute_soc(df2)

    # Safety clamp - handle both column name formats
    for col_name in ["_soc_after", "soc_after"]:
        if col_name in df2.columns:
            df2[col_name] = df2[col_name].clip(lower=float(SOC_FLOOR_KWH), upper=float(SOC_CAP_KWH))

    ok = True
    gained = float(target_kwh)

    if audit is not None:
        audit.append({
            "kind": "proactive_charge", 
            "bus": cand["bus"],
            "row_src": _safe_int(cand["idx"]),
            "used_min": float(used_min),
            "gained_kwh": float(gained),
            "deadline": float(deadline),
            "node": node
        })

    return df2, gained, used_min, ok

def _safe_int(val):
    """Safely convert value to int, handling float strings like '1.0'"""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0

def _get_soc_value(row, before=True, default=270.0):
    """Get SoC value from row, handling both _soc_before/after and soc_before/after columns"""
    if before:
        col_names = ["_soc_before", "soc_before"]
    else:
        col_names = ["_soc_after", "soc_after"]
    
    for col in col_names:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    
    return default

def _get_soc_column(df, before=True):
    """Get SoC column from DataFrame, handling both _soc_* and soc_* column formats."""
    if before:
        col_names = ["_soc_before", "soc_before"]
    else:
        col_names = ["_soc_after", "soc_after"]
    
    for col in col_names:
        if col in df.columns:
            return df[col]
    
    # Return a series of zeros if no SoC column found
    return pd.Series(0.0, index=df.index)

def _latest_candidates_before(df, bus, deadline):
    """
    Build candidate windows for a single bus that FINISH before 'deadline'.
    Each candidate is either an 'idle' or 'material' row (we can borrow time there),
    with its end time < deadline. Opt-Cell 2c/2d then apply a lookback filter.
    """
    # Use _end_min for ordering; keep only rows from this bus
    g = df[df["bus"] == bus].sort_values("_end_min", kind="mergesort")

    cands = []
    for idx, r in g.iterrows():
        end_t = float(r["_end_min"])
        if end_t >= deadline:
            break  # rows are sorted by _end_min, the rest will also be >= deadline

        cls = str(r["_class"]).lower()
        if cls not in ("idle", "material"):
            continue

        # We borrow from the time span of this row
        cands.append({
            "idx": idx,
            "bus": _safe_int(r["bus"]),
            "start": float(r["_start_min"]),
            "end":   end_t,
            "node":  str(r.get("_end_node", r.get("_start_node",""))),
            "class": cls,
            "kind": cls  # for compatibility
        })

    return cands

def _apply_charging_on_candidate(df, cand, deficit_kwh, deadline, audit=None):
    """
    Try to carve a charging block inside the candidate window to recover up to 'deficit_kwh'
    BEFORE 'deadline'.

    Rules:
      - Only at garage nodes (start=end at garage).
      - Charging block must be >= MIN_CHARGE_MIN minutes.
      - Max energy is limited by FAST_KW and the usable minutes in the window.
      - If the window overruns 'deadline', it is trimmed to fit.
      - Recompute SoC after insertion and clamp within battery limits.

    Returns:
      df_new, gained_kwh, used_min, ok_bool
    """
    ok = False
    used_min = 0.0
    gained = 0.0

    # Window we can use
    t0 = float(cand["start"])
    t1 = float(cand["end"])
    if t1 > float(deadline):
        t1 = float(deadline)

    # Require a real window
    win_min = max(0.0, t1 - t0)
    if win_min < MIN_CHARGE_MIN - 1e-6:
        return df, gained, used_min, ok

    # Garage-only
    node = str(cand["node"]).lower()
    if not node.startswith(str(GARAGE_NODE).lower()):
        return df, gained, used_min, ok

    # How much could we charge in this window?
    max_kwh_from_time = (FAST_KW / 60.0) * win_min
    target_kwh = min(deficit_kwh, max_kwh_from_time)

    if target_kwh <= 0.0:
        return df, gained, used_min, ok

    # Minutes actually needed to deliver 'target_kwh'
    used_min = target_kwh / (FAST_KW / 60.0)
    if used_min < MIN_CHARGE_MIN - 1e-6:
        return df, 0.0, 0.0, False

    # Build the charging row (start @ t0 to t0+used_min at the garage)
    new_row = make_row(
        cand["bus"], node, node, t0, t0 + used_min, "charging",
        energy_kwh=-float(target_kwh)
    )
    new_row["_src_row"] = _safe_int(cand["idx"])

    # Insert and recompute SoC
    df2 = insert_row(df, new_row)
    df2 = recompute_soc(df2)

    # Safety clamp (in case recompute doesn't already clamp)
    # Handle both column name formats
    for col_name in ["_soc_after", "soc_after"]:
        if col_name in df2.columns:
            df2[col_name] = df2[col_name].clip(lower=float(SOC_FLOOR_KWH), upper=float(SOC_CAP_KWH))

    ok = True
    gained = float(target_kwh)

    if audit is not None:
        audit.append({
            "kind": "proactive_charge",
            "bus": cand["bus"],
            "row_src": _safe_int(cand["idx"]),
            "used_min": float(used_min),
            "gained_kwh": float(gained),
            "deadline": float(deadline),
            "node": node
        })

    return df2, gained, used_min, ok

# ===== FULL PROACTIVE PLANNER IMPLEMENTATION =====

def _future_shortfalls(bp_df, bus, t_now, margin_kwh=15.0):
    """
    List future shortfalls for this bus after time t_now.
    Each item: dict(deadline, deficit_kwh, row_idx)
    """
    g = bp_df[bp_df["bus"]==bus].sort_values(["_start_min","_end_min"])
    out = []
    for idx, r in g.iterrows():
        if float(r["_start_min"]) <= t_now + 1e-6:
            continue
        if str(r["_class"]) not in ("service","material"):
            continue
        need = SOC_FLOOR_KWH + margin_kwh + float(r["energy consumption"])
        have = _get_soc_value(r, before=True, default=SOC_CAP_KWH)
        deficit = max(0.0, need - have)
        if deficit > ENERGY_TOL_KWH:
            out.append({"deadline": float(r["_start_min"]),
                        "deficit_kwh": deficit,
                        "row_idx": idx})
    return out

def _score_candidate_effect(bp_df, bus, cand, margin_kwh=15.0):
    """
    Score the candidate by how much it reduces the near-term **maximum** shortfall (minimax).
    Returns (reduction, first_deadline); higher reduction is better, earlier deadline tiebreaks.
    """
    t_now   = cand["end"]
    # Estimate potential gain (candidate impls may not supply 'charge_min'; handle gracefully)
    charge_min = float(cand.get("charge_min", max(0.0, cand["end"] - cand["start"])))
    gains   = charge_kwh_possible(charge_min)
    if gains <= ENERGY_TOL_KWH:
        return (0.0, float("inf"))

    shortfalls = _future_shortfalls(bp_df, bus, t_now, margin_kwh)
    if not shortfalls:
        return (0.0, float("inf"))

    max_def_before = max(s["deficit_kwh"] for s in shortfalls)
    # naive projection: reduce all by 'gains' (upper bound useful)
    max_def_after  = max(max(0.0, s["deficit_kwh"] - gains) for s in shortfalls)
    reduction = max_def_before - max_def_after
    first_deadline = shortfalls[0]["deadline"]
    return (reduction, first_deadline)

def proactive_planner_future_aware(bp_df, DM, progress_callback=None, margin_kwh=15.0, lookback_min=480.0):
    """
    Future-aware proactive pass: pick candidates that reduce the
    near-term maximum shortfall (minimax), bus by bus.
    """
    df = ensure_nodes(bp_df.copy())
    df.sort_values(["bus","_start_min","_end_min","_src_row"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    df = recompute_soc(df)

    actions = []
    audit   = []

    for bus, g in df.groupby("bus", sort=False):
        g = g.sort_values(["_start_min","_end_min"])
        for idx, r in g.iterrows():
            if str(r["_class"]) not in ("service","material"):
                continue
            deadline = float(r["_start_min"])

            # Build candidates finishing before this deadline and within lookback
            cand_raw = [c for c in _latest_candidates_before(df, bus, deadline)
                        if c["end"] >= deadline - lookback_min]
            if not cand_raw:
                continue

            # Score & sort (best reduction first, earlier deadlines first)
            scored = []
            for c in cand_raw:
                # enrich with an estimated charge_min for scoring (used by _score_candidate_effect)
                c = dict(c)
                c.setdefault("charge_min", max(0.0, c["end"] - c["start"]))
                reduction, first_deadline = _score_candidate_effect(df, bus, c, margin_kwh)
                scored.append((reduction, first_deadline, c))
            scored.sort(key=lambda z: (-z[0], z[1]))

            # Try best candidates until one actually helps (then re-evaluate with new SoC)
            for reduction, _, c in scored:
                if reduction <= ENERGY_TOL_KWH:
                    break

                # total outstanding deficit after this candidate ends
                need_list = _future_shortfalls(df, bus, c["end"], margin_kwh)
                outstanding = sum(s["deficit_kwh"] for s in need_list)
                if outstanding <= ENERGY_TOL_KWH:
                    break

                df, gained, used_min, ok = _apply_charging_on_candidate(
                    df, c, outstanding, deadline, audit
                )
                if ok and gained > ENERGY_TOL_KWH:
                    actions.append(("proactive", int(df.at[idx,"_src_row"]), str(bus),
                                    c["kind"], used_min, gained))
                    # Keep consistent; re-evaluate future with new SoC
                    df.sort_values(["bus","_start_min","_end_min","_src_row"],
                                   inplace=True, kind="mergesort")
                    df.reset_index(drop=True, inplace=True)
                    df = recompute_soc(df)
                    break  # proceed to next leg

    return df

def pre_topoff_near_day_end(bp_df, DM, progress_callback=None, horizon_min=120.0, margin_kwh=15.0):
    """Full pre-topoff implementation"""
    df = ensure_nodes(bp_df.copy())
    df.sort_values(["bus","_start_min","_end_min","_src_row"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    df = recompute_soc(df)

    acts = []
    audit = []

    # priority within candidates: already-at-garage first
    order = {"charging_row":0, "idle_garage":1, "station_idle_loop":2, "gap_window":3}

    for bus, g in df.groupby("bus", sort=False):
        g = g.sort_values(["_start_min","_end_min"])
        last = g[g["_class"].isin(["service","material"])].tail(1)
        if last.empty:
            continue
        idx = last.index[0]
        r   = df.loc[idx]
        deadline = float(r["_start_min"])
        t0 = max(0.0, deadline - float(horizon_min))

        need = SOC_FLOOR_KWH + margin_kwh + float(r["energy consumption"])
        have = _get_soc_value(r, before=True, default=SOC_CAP_KWH)
        deficit = max(0.0, need - have)
        if deficit <= ENERGY_TOL_KWH:
            continue

        cand = [c for c in _latest_candidates_before(df, bus, deadline)
                if c["end"] >= t0]
        if not cand:
            continue

        # Prefer cleaner windows; then longer charge_min
        for c in cand:
            c.setdefault("charge_min", max(0.0, c["end"] - c["start"]))
        cand.sort(key=lambda c: (order.get(c["kind"],9), -c["charge_min"]))

        for c in cand:
            df, gained, used_min, ok = _apply_charging_on_candidate(
                df, c, deficit, deadline, audit
            )
            if ok and gained > ENERGY_TOL_KWH:
                acts.append(("pre_topoff", int(df.at[idx,"_src_row"]), str(bus),
                             c["kind"], used_min, gained))
                df.sort_values(["bus","_start_min","_end_min","_src_row"],
                               inplace=True, kind="mergesort")
                df.reset_index(drop=True, inplace=True)
                df = recompute_soc(df)
                break  # one top-off per bus is enough

    return df

def patch_soc_breaches(bp_df, DM, progress_callback=None, max_passes=2, margin_kwh=15.0, lookback_min=480.0):
    """
    Try to fix remaining SoC breaches by reusing the same candidate logic as 2c,
    iteratively and only for rows still below requirement.
    """
    df = ensure_nodes(bp_df.copy())
    df.sort_values(["bus","_start_min","_end_min","_src_row"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    df = recompute_soc(df)

    applied = []
    audit   = []

    for pass_num in range(max_passes):
        placed_any = False
        if progress_callback:
            progress_callback(f"SoC breach patching pass {pass_num + 1}/{max_passes}...")

        for bus, g in df.groupby("bus", sort=False):
            g = g.sort_values(["_start_min","_end_min"])
            for idx, r in g[g["_class"].isin(["service","material"])].iterrows():
                required = SOC_FLOOR_KWH + margin_kwh + float(r["energy consumption"])
                current  = float(df.at[idx, "_soc_before"])
                if current + 1e-6 >= required:
                    continue

                deficit  = required - current
                deadline = float(r["_start_min"])
                cands = [c for c in _latest_candidates_before(df, bus, deadline)
                         if c["end"] >= deadline - lookback_min]
                for c in cands:
                    df, gained, used_min, ok = _apply_charging_on_candidate(
                        df, c, deficit, deadline, audit
                    )
                    if ok and gained > ENERGY_TOL_KWH:
                        applied.append((
                            "breach_patch",
                            int(df.at[idx, "_src_row"]),
                            str(bus),
                            c["kind"],
                            used_min,
                            gained
                        ))
                        placed_any = True
                        # keep state consistent
                        df.sort_values(["bus","_start_min","_end_min","_src_row"],
                                       inplace=True, kind="mergesort")
                        df.reset_index(drop=True, inplace=True)
                        df = recompute_soc(df)
                        break
            if placed_any:
                break

        if not placed_any:
            break

    df = recompute_soc(df)
    return df

# ===== OPT-CELL 4: FINAL VALIDATE & EXPORT ===================

def final_validate_and_export(bp_final, bp_base=None, progress_callback=None):
    """
    Final validation and export processing from Opt-Cell 4
    """
    if progress_callback:
        progress_callback("Final validation and processing...")
    
    # ---- Policy (safe even after restart)
    BATTERY_KWH_LOCAL = 300.0
    SOC_MIN_FRAC_LOCAL = 0.10
    SOC_FLOOR_KWH_LOCAL = BATTERY_KWH_LOCAL * SOC_MIN_FRAC_LOCAL  # = 30.0

    # ---- Ensure schema + minimal helpers (idempotent)
    def _to_min_any_local(x):
        s=str(x).strip(); p=s.split(":")
        if len(p)==2: h,m=int(p[0]),int(p[1]); sec=0
        else: h,m,*r=[int(float(t)) for t in p[:3]]; sec=r[0] if r else 0
        return h*60+m+(sec//60)

    need = ["bus","activity","line","start location","end location","start time","end time","energy consumption"]
    miss = [c for c in need if c not in bp_final.columns]
    if miss: 
        raise KeyError(f"Missing columns: {miss}")

    for c in ["activity","start location","end location","start time","end time"]:
        bp_final[c] = bp_final[c].astype(str).str.strip()
    bp_final["line"] = pd.to_numeric(bp_final["line"], errors="coerce").astype("Int64")
    bp_final["bus"]  = bp_final["bus"].astype(str).str.strip()

    if "_start_min"  not in bp_final: bp_final["_start_min"]  = bp_final["start time"].map(_to_min_any_local)
    if "_end_min"    not in bp_final: bp_final["_end_min"]    = bp_final["end time"].map(_to_min_any_local)
    if "_dur_min"    not in bp_final: bp_final["_dur_min"]    = bp_final["_end_min"] - bp_final["_start_min"]
    if "_start_node" not in bp_final: bp_final["_start_node"] = bp_final["start location"]
    if "_end_node"   not in bp_final: bp_final["_end_node"]   = bp_final["end location"]
    if "_class"      not in bp_final: bp_final["_class"]      = bp_final["activity"].str.lower().str.replace(" ","_",regex=False)
    if "_src_row"    not in bp_final: bp_final["_src_row"]    = bp_final.index

    # ---- Recompute SoC with your algorithm
    bp_final = recompute_soc(bp_final.copy())

    # ---- Continuity (ignore last row per bus; optional night-gap filter)
    viol = []

    NIGHT_GAP_MIN = 240   # minutes; set None to disable night tolerance

    grp = bp_final.groupby("bus", sort=False)
    next_start_loc = grp["start location"].shift(-1)
    next_start_min = grp["_start_min"].shift(-1)
    has_next       = next_start_loc.notna()

    if NIGHT_GAP_MIN is None:
        bad_jump = has_next & (bp_final["_end_node"] != next_start_loc)
    else:
        gap_min  = next_start_min - bp_final["_end_min"]
        bad_jump = has_next & (gap_min < NIGHT_GAP_MIN) & (bp_final["_end_node"] != next_start_loc)

    for idx in bp_final.index[bad_jump.fillna(False)]:
        detail = f"{bp_final.at[idx,'_end_node']}→{next_start_loc.loc[idx]}"
        if NIGHT_GAP_MIN is not None:
            detail += f" (gap {int((next_start_min.loc[idx]-bp_final.at[idx,'_end_min']))} min)"
        viol.append({
            "rule":"S-02B_STRICT_CONTINUITY","severity":"fatal",
            "bus":bp_final.at[idx,"bus"],"row":int(idx),"detail":detail
        })

    # ---- SoC floor breaches
    soc_after_col = _get_soc_column(bp_final, before=False)
    bad_soc = soc_after_col < (SOC_FLOOR_KWH_LOCAL - 1e-6)
    for idx in bp_final.index[bad_soc]:
        viol.append({
            "rule":"E-006_SOC_BELOW_FLOOR","severity":"fatal",
            "bus":bp_final.at[idx,"bus"],"row":int(idx),
            "detail":f"{bp_final.at[idx,'_soc_after']:.2f} < floor {SOC_FLOOR_KWH_LOCAL:.1f}"
        })

    viol_final = pd.DataFrame(viol)

    # ---- Diffs vs base (nice to have)
    added = pd.DataFrame()
    removed = pd.DataFrame()
    
    if bp_base is not None:
        def sig(df):
            return df[["bus","activity","line","start location","start time","end location","end time","energy consumption"]].copy()

        added   = pd.concat([sig(bp_final), sig(bp_base)]).drop_duplicates(keep=False)
        removed = pd.concat([sig(bp_base), sig(bp_final)]).drop_duplicates(keep=False)

    # ---- Order & export preparation
    bp_final = bp_final.sort_values(["bus","start time"], kind="mergesort").reset_index(drop=True)

    return {
        "final_plan": bp_final,
        "violations": viol_final,
        "added": added,
        "removed": removed
    }

def set_bg(image_path):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
            }}
            .text {{
                color: #ffffff;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
            }}
            /* Improve expander content readability only */
            .stExpander [data-testid="stExpanderDetails"] {{
                background-color: rgba(255, 255, 255, 0.7) !important;
                backdrop-filter: blur(5px) !important;
                padding: 15px !important;
                border-radius: 5px !important;
                margin-top: 5px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        # Fallback styling if background image is not found
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .text {
                color: #ffffff;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
            }
            /* Improve expander content readability only */
            .stExpander [data-testid="stExpanderDetails"] {
                background-color: rgba(255, 255, 255, 0.7) !important;
                backdrop-filter: blur(5px) !important;
                padding: 15px !important;
                border-radius: 5px !important;
                margin-top: 5px !important;
            }
            /* Style reset button with same transparency as expander */
            button[kind="secondary"] {
                background-color: rgba(255, 255, 255, 0.7) !important;
                backdrop-filter: blur(5px) !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
                border-radius: 5px !important;
                color: #333 !important;
            }
            button[kind="secondary"]:hover {
                background-color: rgba(255, 255, 255, 0.85) !important;
                border: 1px solid rgba(255, 255, 255, 0.5) !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

def add_logo(logo_path, width=250):
    try:
        with open(logo_path, "rb") as f:
            logo_encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_encoded}" width="{width}">
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        # Fallback: Display text logo
        st.markdown("### 🚌 Transdev Bus Optimization")

# UI Setup
# Use relative paths for cloud deployment
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from pages/
bg_path = os.path.join(current_dir, "bus_streamlit_proef4.png")
logo_path = os.path.join(current_dir, "transdev_logo_2018.png")

set_bg(bg_path)
add_logo(logo_path)

st.markdown("""
    <style>
    .nowrap-title {
        white-space: nowrap;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        display: block;
        margin: 0 auto;
        color: #31333f;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #31333f;
        margin-bottom: 1.5em;
    }

    </style>
    <div class='nowrap-title'>Optimized Busplan Generator</div>
    <div class='subtitle'>
        Upload your busplan to generate an optimized version with improved energy efficiency and feasibility.
    </div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("**How to use the optimization tool**"):
    st.markdown("""
    ### How to use:
    1. **Upload Files**: Upload your Bus Planning Excel file and optionally Distance Matrix & Timetable files
    2. **Review Preview**: Check the data preview to ensure your files loaded correctly
    3. **Generate**: Click "Generate Optimized Busplan" to start the optimization process
    4. **View Results**: Once complete, review the optimization results and metrics
    5. **Download**: Download your optimized busplan Excel file
    6. **Reset**: Use "Reset Page" button to start over with new files
    
    ### What you'll see after optimization:
    - **Metrics Overview**: Original vs Optimized plan row counts, Minimum SoC, SoC Floor (30.0 kWh), and SoC Breaches
    - **Optimized Busplan**: Complete optimized schedule in an expandable table view
    - **Download Button**: Get your optimized busplan as an Excel file with timestamp
    - **Violations**: List of any remaining issues found in the optimized plan
    - **Suggestions**: Recommended improvements and optimization opportunities  
    - **Added Rows**: New activities inserted during optimization (charging sessions, material trips)
    - **Removed Rows**: Original activities that were modified or replaced during optimization
                
    ⚠️ **Important**: After seeing optimization results, please wait for the "Violation calculation completed" message before switching to other pages. This ensures accurate KPI data is available.
    
    ### What the optimization does:
    - **Fixes Continuity**: Inserts material trips to connect disconnected bus activities
    - **Smart Charging**: Converts garage idle time into efficient charging sessions (≥15 min)
    - **Energy Management**: Ensures battery levels stay above 30 kWh throughout the day
    - **Station Optimization**: Upgrades station idle periods into garage charging loops when beneficial
    - **Data Validation**: Corrects timing issues, energy calculations, and activity classifications
    
    ### File Requirements:
    **(Validated) Bus Planning file** should contain columns for:
    - Bus number/ID
    - Activity type (service, material, idle, charging)
    - Start and end locations
    - Start and end times
    - Line information
    - Energy consumption data
    
    ### Interpreting Results:
    - **Green metrics** indicate successful optimization
    - **Violations** show remaining issues that need attention  
    - **Suggestions** provide improvement opportunities
    - **Added/Removed rows** show what changed during optimization
    - Use the KPI Calculations page to analyze performance in detail
    """)

# Reset button with better centering and transparency styling
reset_col1, reset_col2, reset_col3 = st.columns([2, 1, 2])
with reset_col2:
    # Apply transparent styling only to reset button using markdown container
    st.markdown("""
        <style>
        div[data-testid="column"]:nth-child(2) .stButton > button[kind="secondary"] {
            background-color: rgba(240, 242, 246, 0.7) !important;
            border: 1px solid rgba(151, 166, 195, 0.7) !important;
            backdrop-filter: blur(10px) !important;
            color: rgba(49, 51, 63, 0.9) !important;
        }
        div[data-testid="column"]:nth-child(2) .stButton > button[kind="secondary"]:hover {
            background-color: rgba(240, 242, 246, 0.8) !important;
            border: 1px solid rgba(151, 166, 195, 0.8) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("**Reset Page**", type="secondary", help="Clear all uploaded files and results"):
        # Clear session state for this page
        for key in ['optimization_result', 'optimization_files_uploaded']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Check if we have saved results
show_results = st.session_state.get('optimization_result') is not None
files_uploaded = st.session_state.get('optimization_files_uploaded', False)

# File upload
if not show_results:
    uploaded_file = st.file_uploader("**Upload Bus Planning Excel file**", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)
        
        st.markdown(f"""
                <div style="
                    background-color: rgba(0, 255, 0, 0.15);
                    border-radius: 6px;
                    padding: 10px;
                    margin: 8px 0;
                    text-align: center;
                    font-size: 1.0em;
                    color: #155724;
                    backdrop-filter: blur(3px);
                ">
                    File uploaded successfully! Found {len(df)} rows.
                </div>
            """, unsafe_allow_html=True)

        # Save original dataframe to session_state for KPI page
        st.session_state['original_df'] = df.copy()

        # Show preview of uploaded data
        with st.expander("**Preview uploaded data**"):
            st.dataframe(df)
        
        # Optimization button
        if st.button("Generate Optimized Busplan", type="primary"):
            # Progress placeholder
            progress_placeholder = st.empty()
            
            # Progress callback function
            def update_progress(message):
                progress_placeholder.text(message)
            
            with st.spinner("Optimizing busplan... This may take a few moments."):
                # Create empty distance matrix and timetable for now
                # In a full implementation, these would also be uploaded
                dm_df = pd.DataFrame()
                tt_df = pd.DataFrame()
                
                # Show input data info
                st.markdown(f"""
                                <div style="
                                    background-color: rgba(0, 123, 255, 0.15);
                                    border-radius: 6px;
                                    padding: 10px;
                                    margin: 8px 0;
                                    text-align: center;
                                    font-size: 1.0em;
                                    color: #004085;
                                    backdrop-filter: blur(3px);
                                ">
                                    Processing busplan with {len(df)} rows and {len(df.columns)} columns
                                </div>
                            """, unsafe_allow_html=True)
                
                # Check required columns
                required_cols = ["start location", "end location", "start time", "end time", 
                               "activity", "line", "energy consumption", "bus"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.markdown(f"""
                        <div style="
                            background-color: rgba(255, 0, 0, 0.15);
                            border-radius: 6px;
                            padding: 10px;
                            margin: 8px 0;
                            text-align: center;
                            font-size: 1.0em;
                            color: #721c24;
                            backdrop-filter: blur(3px);
                        ">
                            Missing required columns: {missing_cols}
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="
                            background-color: rgba(0, 123, 255, 0.15);
                            border-radius: 6px;
                            padding: 10px;
                            margin: 8px 0;
                            text-align: center;
                            font-size: 1.0em;
                            color: #004085;
                            backdrop-filter: blur(3px);
                        ">
                            Available columns: {", ".join(df.columns.tolist())}
                        </div>
                    """, unsafe_allow_html=True)
                    st.stop()
                
                # Run optimization with progress callback
                result = run_checker_and_optimizer(df, dm_df, tt_df, progress_callback=update_progress)
                optimized_df = result["optimized_plan"]
                
                # Save results to session state
                st.session_state['optimization_result'] = {
                    'optimized_df': optimized_df,
                    'result': result,
                    'original_filename': uploaded_file.name
                }
                st.session_state['optimization_files_uploaded'] = True
                
                # Show success message and rerun to display results
                progress_placeholder.markdown("""
                    <div style="
                        background-color: rgba(0, 255, 0, 0.15);
                        border-radius: 6px;
                        padding: 10px;
                        margin: 8px 0;
                        text-align: center;
                        font-size: 1.0em;
                        color: #155724;
                        backdrop-filter: blur(3px);
                    ">
                        Optimization completed successfully! Reloading results...
                    </div>
                """, unsafe_allow_html=True)
                st.rerun()

    else:
        st.markdown("""
            <div style="
                background-color: rgba(0, 123, 255, 0.15);
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
                text-align: center;
                font-size: 1.0em;
                color: #004085;
                backdrop-filter: blur(3px);
            ">
                Please upload a Bus Planning Excel file to get started.
            </div>
        """, unsafe_allow_html=True)

# Show results if available
if show_results:
    saved_data = st.session_state['optimization_result']
    optimized_df = saved_data['optimized_df']
    result = saved_data['result']
    original_filename = saved_data['original_filename']
    
    # Save optimized dataframe to session_state for KPI page (maintain compatibility)
    st.session_state['optimized_df'] = optimized_df.copy()

    # Show results summary
    col1, col2 = st.columns(2)
    with col1:
        # Get original df length from session state
        original_rows = len(st.session_state.get('original_df', [])) if 'original_df' in st.session_state else 0
        st.metric("**Original Plan Rows**", original_rows)
    with col2:
        st.metric("**Optimized Plan Rows**", len(optimized_df))
    
    # Show preview of optimized data
    st.subheader("Optimized Busplan")
    with st.expander("**Preview optimized busplan**", expanded=True):
        st.dataframe(optimized_df)
    
    # Prepare download
    output = io.BytesIO()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
    filename = f"{base_name}_Optimized_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write main optimized plan
        display_cols = [col for col in optimized_df.columns if not col.startswith('_')]
        optimized_df[display_cols].to_excel(writer, sheet_name='busplan_optimized', index=False)
        
        # Write technical details if available
        if any(col.startswith('_soc') for col in optimized_df.columns):
            optimized_df.to_excel(writer, sheet_name='technical_details', index=False)
    
    output.seek(0)
    
    # Download button
    st.download_button(
        label="Download Optimized Busplan",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Show important warning about waiting for violation calculation
    st.markdown("""
        <div style="
            background-color: rgba(255, 193, 7, 0.15);
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
            text-align: center;
            font-size: 1.1em;
            color: #856404;
            backdrop-filter: blur(3px);
            border-left: 4px solid #ffc107;
        ">
            ⚠️ <strong>Important</strong>: After seeing optimization results, please wait for the "Violation calculation completed" message before switching to other pages. This ensures accurate KPI data is available.
        </div>
    """, unsafe_allow_html=True)
                
    # Show SoC summary if available
    try:
        soc_before_col = _get_soc_column(optimized_df, before=True)
        soc_after_col = _get_soc_column(optimized_df, before=False)
        soc_data = pd.concat([soc_before_col, soc_after_col])
        soc_min = float(soc_data.min(skipna=True))
        soc_breaches = int(((soc_before_col < SOC_FLOOR_KWH).sum() + 
                          (soc_after_col < SOC_FLOOR_KWH).sum()))
    except (KeyError, ValueError):
        # SoC columns not available in expected format
        soc_min = None
        soc_breaches = 0
    
    if soc_min is not None:
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("**Minimum SoC**", f"{soc_min:.1f} kWh")
        with col2:
            st.metric("**SoC Floor**", f"{SOC_FLOOR_KWH:.1f} kWh")
        with col3:
            st.metric("**SoC Breaches**", soc_breaches)
        
        # Show optimization results summary
        violations_df = result.get("violations", pd.DataFrame())
        suggestions_df = result.get("suggestions", pd.DataFrame())
        
        # Store violations count in session_state for KPI page
        st.session_state['amount_violations_optimized'] = len(violations_df) if not violations_df.empty else 0
        
        if not violations_df.empty:
            st.markdown(f"### Violations <span style='font-size:0.8em; color:#555;'>({len(violations_df)} found)</span>", unsafe_allow_html=True)
            with st.expander("**View violations**", expanded=False):
                st.dataframe(violations_df)
        else:
            st.markdown("### Violations <span style='font-size:0.8em; color:#555;'>(0 found)</span>", unsafe_allow_html=True)
            st.success("Optimized plan has no violations!")
            
        if not suggestions_df.empty:
            st.markdown(f"### Suggestions <span style='font-size:0.8em; color:#555;'>({len(suggestions_df)} available)</span>", unsafe_allow_html=True)
            with st.expander("**View suggestions**", expanded=False):
                st.dataframe(suggestions_df)
        
        # Show changes made during final validation
        added_rows = result.get("added_rows", pd.DataFrame())
        removed_rows = result.get("removed_rows", pd.DataFrame())
        
        if not added_rows.empty:
            st.markdown(f"### Added rows <span style='font-size:0.8em; color:#555;'>({len(added_rows)} during final validation)</span>", unsafe_allow_html=True)
            with st.expander("**View added rows**", expanded = False):
                st.dataframe(added_rows)
                
        if not removed_rows.empty:
            st.markdown(f"### Removed rows <span style='font-size:0.8em; color:#555;'>({len(removed_rows)} during final validation)</span>", unsafe_allow_html=True)
            with st.expander("**View removed rows**", expanded = False):
                st.dataframe(removed_rows)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.9em;">Transdev Optimization Tool - Powered by Advanced Bus Planning Algorithms</div>',
    unsafe_allow_html=True
)
