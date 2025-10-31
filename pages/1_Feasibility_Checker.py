import streamlit as st
import base64
import pandas as pd
import io
import os
import numpy as np
from typing import Tuple, Optional

# Configure pandas options
pd.options.mode.copy_on_write = True

# ===== FEASIBILITY CHECKER CONSTANTS AND FUNCTIONS =====

# Constants from Checker_only.ipynb
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

# Inferred missing constants
KWH_PER_KM_MIN       = 0.7            # minimum energy consumption per km
KWH_PER_KM_MAX       = 2.5            # maximum energy consumption per km
SOH_PCT              = 0.90           # state of health percentage (90% van originele capaciteit)
DAY_CAP_FRAC         = SOC_CAP_FRAC   # day cap fraction (same as SOC cap)

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

def run_feasibility_checker(bp_df: pd.DataFrame, dm_df: pd.DataFrame, tt_df: pd.DataFrame) -> dict:
    """
    Complete feasibility checker implementation from Checker_only.ipynb
    Returns dict with violations, soc data, and other results
    """
    
    # Initialize containers
    violations = []
    suggestions = []
    
    # Make working copies
    bp = bp_df.copy()
    dm = dm_df.copy()
    tt = tt_df.copy()
    
    # ===== Cell 2: load, normalize, map activities, durations, zero-duration rules =====
    
    # keep original Excel row (to reconcile with violations later)
    bp["_src_row"] = bp.index + 2  # +1 for 1-based, +1 header row

    # normalize column names
    bp.columns = [c.strip().lower() for c in bp.columns]
    dm.columns = [c.strip().lower() for c in dm.columns]
    tt.columns = [c.strip().lower() for c in tt.columns]

    # required columns in Bus Planning
    REQ_BP = {
        "start location","end location","start time","end time",
        "activity","line","energy consumption","bus"
    }
    missing = REQ_BP - set(bp.columns)
    if missing:
        st.markdown(f"""
            <div style="
                background-color: rgba(220, 53, 69, 0.15);
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
                text-align: center;
                font-size: 1.0em;
                color: #721c24;
                backdrop-filter: blur(3px);
            ">
                BusPlanning missing columns: {missing}
            </div>
        """, unsafe_allow_html=True)
        return {"violations": pd.DataFrame(), "soc": pd.DataFrame(), "bp_sorted": pd.DataFrame(), "rule_counts": pd.DataFrame()}

    # normalize text fields
    for col in ["start location","end location","activity","line","bus"]:
        if col in bp.columns:
            bp[col] = bp[col].astype(str).str.strip().str.lower()

    for col in ["start","end","origin","destination","from","to","line"]:
        if col in dm.columns:
            dm[col] = dm[col].astype(str).str.strip().str.lower()
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

    # optional warn for 0-min idle/charging — skip harmless connectors
    warn_idle = (bp["_dur_min"] < TIME_ZERO_TOL_MIN) & (bp["_class"].isin(["idle","charging"]))
    for idx in bp.index[warn_idle]:
        # next row for same bus that starts at the same boundary? then ignore
        nxt = bp[(bp["bus"]==bp.at[idx,"bus"]) & (bp["_start_min"]>=bp.at[idx,"_end_min"])].head(1)
        same_boundary = (len(nxt)==1) and abs(float(nxt["_start_min"].iloc[0]) - float(bp.at[idx,"_end_min"])) < TIME_ZERO_TOL_MIN
        if same_boundary:
            continue
        violations.append({
            "rule":"S-002_IDLE_ZERO_DURATION","severity":"warn","bus":bp.at[idx,"bus"],
            "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
            "detail":f"idle/charging duration={bp.at[idx,'_dur_min']:.1f} min"
        })

    # final order per bus
    bp.sort_values(["bus","_start_min","_end_min"], inplace=True, kind="mergesort")
    bp.reset_index(drop=True, inplace=True)

    # ===== Cell 3: build DistanceMatrix & Timetable indices =====
    
    def _pick(colset, *cands):
        for c in cands:
            if c in colset:
                return c
        raise KeyError(f"Missing required column from: {cands}")

    # DistanceMatrix picks
    try:
        dm_start = _pick(dm.columns, "start","origin","from","start location")
        dm_end   = _pick(dm.columns, "end","destination","to","end location")
        dm_min   = _pick(dm.columns, "min_travel_time","min_time","mintime","min")
        dm_max   = _pick(dm.columns, "max_travel_time","max_time","maxtime","max")
        dm_dist  = _pick(dm.columns, "distance_km","distance (km)","distance","distance_m","distance (m)")
    except KeyError as e:
        st.markdown(f"""
            <div style="
                background-color: rgba(220, 53, 69, 0.15);
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
                text-align: center;
                font-size: 1.0em;
                color: #721c24;
                backdrop-filter: blur(3px);
            ">
                DistanceMatrix missing required columns: {e}
            </div>
        """, unsafe_allow_html=True)
        return {"violations": pd.DataFrame(), "soc": pd.DataFrame(), "bp_sorted": pd.DataFrame(), "rule_counts": pd.DataFrame()}

    # normalize distance to km
    dm["_dist_km"] = pd.to_numeric(dm[dm_dist], errors="coerce")
    dist_vals = pd.to_numeric(dm[dm_dist], errors="coerce")
    if ("m" in dm_dist.lower()) or (pd.to_numeric(dist_vals, errors="coerce").max() > 5000):
        dm["_dist_km"] = dm["_dist_km"] / 1000.0

    DM = {}
    for _, r in dm.iterrows():
        start = str(r[dm_start]); end = str(r[dm_end])
        tmin  = float(r[dm_min]);  tmax = float(r[dm_max])
        dkm   = float(r["_dist_km"])
        DM[(start,end)] = {"tmin": tmin, "tmax": tmax, "km": dkm}

    # optional timetable index
    TT = {}
    req = {"origin","destination","departure_time","arrival_time","line"}
    if req.issubset(tt.columns):
        tt["_dep_min"] = tt["departure_time"].apply(to_minutes)
        tt["_arr_min"] = tt["arrival_time"].apply(to_minutes)
        roll = tt["_arr_min"] < tt["_dep_min"]
        tt.loc[roll, "_arr_min"] += 1440
        def _round_min(m): return int(round(float(m)))
        for _, r in tt.iterrows():
            key = (str(r["line"]), str(r["origin"]), str(r["destination"]), _round_min(r["_dep_min"]))
            TT[key] = {"arr": float(r["_arr_min"]), "runtime": float(r["_arr_min"] - r["_dep_min"])}

    # ===== Cell 5: time overlaps & energy legality =====
    
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

    # ===== Cell 6: continuity & teleports =====
    
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

    # ===== Cell 7: HARD continuity + DM feasibility (vectorized) =====
    
    TOL_MIN = 0.5  # rounding tolerance for timing

    def dm_tmin(a, b):
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
                "row_df": int(idx),                 # current row is k; the offender is the next, but we flag here
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
            tmin = dm_tmin(df.at[idx,"this_end_loc"], df.at[idx,"next_start_loc"])
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
            tmin = dm_tmin(r["start location"], r["end location"])
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

    # ===== Cell 8: simulate SoC across the day per bus =====
    
    bp["_soc_before"] = np.nan
    bp["_soc_after"]  = np.nan

    for bus, df in bp.groupby("bus", sort=False):
        soc = SOC_START_KWH
        for idx, r in df.sort_values(["_start_min","_end_min"]).iterrows():
            cls   = r["_class"]
            kwh   = float(r.get("energy consumption", 0.0))
            dur_h = float(r["_dur_min"]) / 60.0

            bp.at[idx, "_soc_before"] = soc

            if cls in {"service","material","idle"}:
                # consumption is positive kWh
                soc -= max(0.0, kwh)
            elif cls == "charging":
                # gain limited by power*time and by cap
                phys_gain = FAST_KW * dur_h
                gain_req  = abs(kwh) if kwh < 0 else 0.0
                gain      = min(phys_gain, gain_req)  # respect file & physics
                soc += gain
                soc = min(soc, SOC_CAP_KWH)           # cap at 90%
            # else: already filtered unknowns

            bp.at[idx, "_soc_after"] = soc

    # ===== Cell 9: SoC floor & cap checks =====
    
    for idx, r in bp.iterrows():
        if r["_soc_after"] < SOC_FLOOR_KWH - 1e-6:
            violations.append({
                "rule":"SOC-001_BELOW_FLOOR","severity":"fatal","bus":r["bus"],
                "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                "detail":f"{r['_soc_after']:.1f} kWh < floor {SOC_FLOOR_KWH:.1f} kWh"
            })
        if r["_soc_after"] > SOC_CAP_KWH + 1e-6:
            violations.append({
                "rule":"SOC-002_ABOVE_CAP","severity":"minor","bus":r["bus"],
                "row_df":int(idx),"row_src":int(bp.at[idx,"_src_row"]),
                "detail":f"{r['_soc_after']:.1f} kWh > cap {SOC_CAP_KWH:.1f} kWh"
            })

    # ===== Cell 11: suggestions (safe auto-fixes) =====
    
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

    # ===== Cell 12: report & export =====
    
    viol_df = pd.DataFrame(violations)
    sug_df  = pd.DataFrame(suggestions)

    # column order
    vcols = ["rule","severity","bus","row_src","row_df","detail"]
    viol_df = viol_df.reindex(columns=[c for c in vcols if c in viol_df.columns] +
                                       [c for c in viol_df.columns if c not in vcols])

    scols = ["kind","row_src","row_df","new_energy_kwh","explain"]
    sug_df  = sug_df.reindex(columns=[c for c in scols if c in sug_df.columns] +
                                       [c for c in sug_df.columns if c not in scols])

    # Create SoC progression dataframe
    soc_rows = []
    for idx, r in bp.iterrows():
        soc_rows.append({
            "bus": r["bus"], 
            "line": r.get("line"),
            "time_start": r["start time"], 
            "time_end": r["end time"],
            "activity": r["activity"], 
            "class": r["_class"], 
            "soc_before_kwh": round(float(r["_soc_before"]),2),
            "soc_after_kwh": round(float(r["_soc_after"]),2)
        })
    soc_df = pd.DataFrame(soc_rows)

    # Create rule counts summary
    if len(viol_df):
        counts = (viol_df.groupby(["rule","severity"])
                  .size().reset_index(name="count")
                  .sort_values(["severity","rule"], ascending=[False, True]))
    else:
        counts = pd.DataFrame(columns=["rule","severity","count"])

    return {
        "rule_counts": counts,
        "violations": viol_df,
        "soc": soc_df,
        "bp_sorted": bp,
        "suggestions": sug_df
    }

# Set page config
st.set_page_config(page_title="Feasibility Checker - Transdev", page_icon=":bus:")

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
            /* Improve expander content readability only */
            .stExpander [data-testid="stExpanderDetails"] {{
                background-color: rgba(255, 255, 255, 0.7) !important;
                backdrop-filter: blur(5px) !important;
                padding: 15px !important;
                border-radius: 5px !important;
                margin-top: 5px !important;
            }}
            /* Style reset button with same transparency as expander */
            button[kind="secondary"] {{
                background-color: rgba(255, 255, 255, 0.7) !important;
                backdrop-filter: blur(5px) !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
                border-radius: 5px !important;
                color: #333 !important;
            }}
            button[kind="secondary"]:hover {{
                background-color: rgba(255, 255, 255, 0.85) !important;
                border: 1px solid rgba(255, 255, 255, 0.5) !important;
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
        # Fallback text logo if image is not found
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #ff6b35; font-size: 2em; margin: 0;">🚌 TRANSDEV</h1>
                <p style="color: #31333f; margin: 0; font-style: italic;">the mobility company</p>
            </div>
            """,
            unsafe_allow_html=True
        )

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
    }
    </style>
    <div class='nowrap-title'>Busplan Feasibility Checker</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center; font-size:1.1em; margin-bottom:1em;">Upload all three required files to begin.</div>',
    unsafe_allow_html=True
)

# Instructions
with st.expander("**How to use the feasibility checker**"):
    st.markdown("""
    ### Instructions:
    1. **Upload Files**: Upload all three required Excel files (Bus Planning, Distance Matrix, and Timetable)
    2. **Review Results**: Check the feasibility summary and any violations found
    3. **Download Reports**: Download detailed reports for violations, SoC progression, or the validated schedule
    4. **Analyze Issues**: Use the violation report to understand what needs to be fixed
    
    ### What the feasibility checker does:
    - **Plan Validation**: Checks if your bus plan is operationally feasible
    - **Rule Enforcement**: Validates against timing, energy, and routing constraints
    - **SoC Analysis**: Tracks battery State of Charge throughout each bus journey (300kWh buses)
    - **Violation Detection**: Identifies specific problems like battery depletion or timing conflicts
    - **Data Normalization**: Standardizes and sorts your bus plan data for analysis
    - **Charging Validation**: Ensures charging only occurs at garage with proper duration (min 15 min)
    
    ### File Requirements:
    **Bus Planning file** should contain columns for:
    - Bus number/ID
    - Activity type (service, material, idle, charging)
    - Start and end locations
    - Start and end times
    - Line information
    - Energy consumption data
    
    **Distance Matrix file** should contain:
    - Location pairs with travel distances
    - Travel times between locations
    
    **Timetable file** should contain:
    - Service schedules and timing constraints
    - Route-specific information
    
    ### Understanding the results:
    - **Green "Feasible: YES"** = Your plan can be executed without issues
    - **Red "Feasible: NO"** = Problems found that need attention
    - **Feasibility summary** = Count of violations by type and severity
    - **Violations table** = Shows specific issues and their locations
    - **SoC progression** = Battery levels throughout the day per bus
    - **Suggested fixes** = Automated recommendations for common issues
    - **Validated schedule** = Your original plan, cleaned and sorted
    
    ### Violation severity levels:
    - **Fatal** = Critical errors that make the plan impossible to execute (overlapping times, battery depletion, teleportation)
    - **Major** = Serious issues that violate operational rules (charging outside garage, incorrect energy signs)
    - **Minor** = Problems that should be addressed but don't prevent execution (energy values slightly outside expected ranges)
    - **Warn** = Notifications about unusual situations that may need attention (zero-duration activities)
    """)

# Reset button with better centering and transparency styling
st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(2) .stButton > button[kind="secondary"] {
        background-color: rgba(240, 242, 246, 0.7) !important;
        border: 1px solid rgba(151, 166, 195, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        color: rgba(49, 51, 63, 0.8) !important;
    }
    div[data-testid="column"]:nth-child(2) .stButton > button[kind="secondary"]:hover {
        background-color: rgba(240, 242, 246, 0.9) !important;
        border: 1px solid rgba(151, 166, 195, 0.9) !important;
        color: rgba(49, 51, 63, 1.0) !important;
    }
    </style>
""", unsafe_allow_html=True)

reset_col1, reset_col2, reset_col3 = st.columns([2, 1, 2])
with reset_col2:
    if st.button("**Reset Page**", type="secondary", help="Clear all uploaded files and results"):
        # Clear session state for this page
        for key in ['feasibility_result', 'feasibility_files_uploaded']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Check if we have saved results
show_results = st.session_state.get('feasibility_result') is not None
files_uploaded = st.session_state.get('feasibility_files_uploaded', False)

if not show_results:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div style="font-size:1.2em;font-weight:bold;text-align:center;margin-bottom:0.1em;">Bus Planning file</div>', unsafe_allow_html=True)
        plan_file = st.file_uploader("", type=["xlsx"], key="plan")
    with col2:
        st.markdown('<div style="font-size:1.2em;font-weight:bold;text-align:center;margin-bottom:0.1em;">Distance Matrix file</div>', unsafe_allow_html=True)
        dist_file = st.file_uploader("", type=["xlsx"], key="dist")
    with col3:
        st.markdown('<div style="font-size:1.2em;font-weight:bold;text-align:center;margin-bottom:0.1em;">Timetable file</div>', unsafe_allow_html=True)
        tt_file = st.file_uploader("", type=["xlsx"], key="tt")

    if plan_file and dist_file and tt_file:
        with st.spinner("Running feasibility check..."):
            plan_df = pd.read_excel(plan_file)
            dist_df = pd.read_excel(dist_file)
            tt_df = pd.read_excel(tt_file)

            result = run_feasibility_checker(plan_df, dist_df, tt_df)
            
            # Save results to session state
            st.session_state['feasibility_result'] = result
            st.session_state['feasibility_files_uploaded'] = True
            
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
                Upload all required files to begin.
            </div>
        """, unsafe_allow_html=True)

if show_results:
    result = st.session_state['feasibility_result']

    rule_counts = result.get("rule_counts", pd.DataFrame())
    viol_df = result.get("violations", pd.DataFrame())
    soc_df = result.get("soc", pd.DataFrame())
    bp_sorted = result.get("bp_sorted", pd.DataFrame())
    sug_df = result.get("suggestions", pd.DataFrame())

    if viol_df.empty:
        st.markdown("""
            <div style="
                background-color: rgba(40, 167, 69, 0.15);
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
                text-align: center;
                font-size: 1.0em;
                color: #155724;
                backdrop-filter: blur(3px);
            ">
                **Feasible: YES (no violations)**
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="
                background-color: rgba(220, 53, 69, 0.15);
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
                text-align: center;
                font-size: 1.0em;
                color: #721c24;
                backdrop-filter: blur(3px);
            ">
                **Feasible: NO (see violations below)**
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Feasibility summary")
    if not rule_counts.empty:
        st.dataframe(rule_counts)
    else:
        st.markdown("""
            <div style="
                background-color: rgba(40, 167, 69, 0.15);
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
                text-align: center;
                font-size: 1.0em;
                color: #155724;
                backdrop-filter: blur(3px);
            ">
                No violations found! Plan is feasible.
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Violations")
    if not viol_df.empty:
        st.dataframe(viol_df)
        output_viol = io.BytesIO()
        with pd.ExcelWriter(output_viol, engine='openpyxl') as writer:
            viol_df.to_excel(writer, index=False)
        st.download_button(
            label="Download violations",
            data=output_viol.getvalue(),
            file_name="violations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.markdown("""
            <div style="
                background-color: rgba(40, 167, 69, 0.15);
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
                text-align: center;
                font-size: 1.0em;
                color: #155724;
                backdrop-filter: blur(3px);
            ">
                No violations found!
            </div>
        """, unsafe_allow_html=True)

    st.subheader("SOC per bus trip")
    if not soc_df.empty:
        st.dataframe(soc_df)
        output_soc = io.BytesIO()
        with pd.ExcelWriter(output_soc, engine='openpyxl') as writer:
            soc_df.to_excel(writer, index=False)
        st.download_button(
            label="Download SOC results",
            data=output_soc.getvalue(),
            file_name="soc_progression.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
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
                No SOC data found.
            </div>
        """, unsafe_allow_html=True)

    # Display suggestions if available
    if 'sug_df' in locals() and not sug_df.empty:
        st.subheader("Suggested fixes")
        st.dataframe(sug_df)
        output_sug = io.BytesIO()
        with pd.ExcelWriter(output_sug, engine='openpyxl') as writer:
            sug_df.to_excel(writer, index=False)
        st.download_button(
            label="Download suggestions",
            data=output_sug.getvalue(),
            file_name="suggestions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if not bp_sorted.empty:
        st.subheader("Preview: Validated schedule")
        st.dataframe(bp_sorted)
        output_plan = io.BytesIO()
        with pd.ExcelWriter(output_plan, engine='openpyxl') as writer:
            bp_sorted.to_excel(writer, index=False)
        st.download_button(
            label="Download validated schedule",
            data=output_plan.getvalue(),
            file_name="validated_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.9em;">Transdev Feasibility Analysis Tool - Ensuring Operational Excellence</div>',
    unsafe_allow_html=True
)

