# Busplan Optimizer - Exact code from Final_Checker_And_Optimizer.ipynb
# === Cell 1: setup, constants, helpers =======================================
from typing import Tuple
import pandas as pd
import numpy as np
from typing import Optional
import re, math
from pathlib import Path

pd.options.mode.copy_on_write = True

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

# Additional constants from optimization cells
SOC_MAX_FRAC    = 0.90
SOC_MIN_FRAC    = 0.10
CHARGER_KW      = 450.0                # fast pantograph
GARAGE_4MIN_DESTS = {"ehvbst","ehvapt"}  # station nodes that are 4 min from the garage
MAX_ITERS         = 12
MAX_SWAPS_PER_BUS = 3
TOL_MIN           = 2
PRIORITY_BUSES    = [20, 19]            # try 20 first
TRACTION_KWH_PER_KM = 1.2
DM_SERVICE        = None    # {(line,s,e):{distance_m,t_min_min}}
DM_MAT            = None    # {(s,e):{distance_m,t_min_min}}

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

# === Cell 2: load, normalize, map activities ===
SERVICE_LABELS  = {"service","service trip","service_trip","service-trip","servicetrip"}
MATERIAL_LABELS = {"material","material trip","material_trip","material-trip","materialtrip","deadhead","dead head","dead-head"}
IDLE_LABELS     = {"idle","idling"}
CHARGING_LABELS = {"charging","charge","charging session"}

def canonical_activity(s: str) -> str:
    s = str(s).strip().lower()
    if s in SERVICE_LABELS:  return "service"
    if s in MATERIAL_LABELS: return "material"
    if s in IDLE_LABELS:     return "idle"
    if s in CHARGING_LABELS: return "charging"
    return "unknown"

# === Opt-Cell 0: helpers (DM, normalization, SoC, row ops) ===================
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

# === Opt-Cell 4.5 — donor swapper functions ===
def _to_min_any(x):
    if pd.isna(x): return pd.NA
    s=str(x).strip()
    if s.isdigit(): return int(s)%(24*60)
    try:
        hh,mm=s.split(":")[:2]; return (int(hh)*60+int(mm))%(24*60)
    except Exception:
        t=pd.to_datetime(x,errors="coerce")
        if pd.isna(t): return pd.NA
        t=pd.Timestamp(t); return (t.hour*60+t.minute)%(24*60)

def _min_to_hhmm(m): 
    m=int(m)%(24*60); return f"{m//60:02d}:{m%60:02d}"

def _order_bus_then_time(df):
    d=df.copy()
    if "_start_min" not in d and "start time" in d: d["_start_min"]=d["start time"].map(_to_min_any)
    if "_end_min"   not in d and "end time"   in d: d["_end_min"]  =d["end time"].map(_to_min_any)
    if "bus" in d:
        bkey=pd.to_numeric(d["bus"],errors="coerce")
        if bkey.isna().any():
            bkey=d["bus"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        d["__buskey"]=bkey.fillna(1e9)
        d=d.sort_values(["__buskey","_start_min","_end_min"], kind="mergesort") \
            .drop(columns="__buskey", errors="ignore").reset_index(drop=True)
    else:
        d=d.sort_values(["_start_min","_end_min"], kind="mergesort").reset_index(drop=True)
    return d

def _canon_class(s):
    s=str(s).lower()
    if s.startswith("service"):  return "service"
    if s.startswith("material"): return "material"
    if s.startswith("charg"):    return "charging"
    if s.startswith("idle"):     return "idle"
    return s

def _norm(df):
    d=df.copy()
    need=["bus","activity","line","start location","end location","start time","end time"]
    miss=[c for c in need if c not in d.columns]
    if miss: raise RuntimeError(f"Opt-4.5 needs columns {miss}")
    for c in ["activity","start location","end location","start time","end time"]:
        d[c]=d[c].astype(str).str.strip()
    d["bus"]=d["bus"].astype(str).str.strip()
    d["line"]=pd.to_numeric(d["line"],errors="coerce").astype("Int64")
    if "_class"      not in d: d["_class"]=d["activity"].map(_canon_class)
    if "_start_node" not in d: d["_start_node"]=d["start location"]
    if "_end_node"   not in d: d["_end_node"]=d["end location"]
    if "_start_min"  not in d: d["_start_min"]=d["start time"].map(_to_min_any)
    if "_end_min"    not in d: d["_end_min"]  =d["end time"].map(_to_min_any)
    dur=d["_end_min"]-d["_start_min"]; d["_dur_min"]=dur.mask(dur<0, dur+24*60).astype("Int64")
    return _order_bus_then_time(d)

def _append_row(df, row):
    defaults={c:(0.0 if c=="energy consumption" else pd.NA) for c in df.columns}
    defaults.update(row)
    row_df=pd.DataFrame.from_records([defaults], columns=df.columns)
    for c in df.columns:
        try: row_df[c]=row_df[c].astype(df[c].dtype)
        except: pass
    return pd.concat([df,row_df], ignore_index=True)

def _tt_min(a, b):
    """Deadhead travel time (min). Enforce 4 min for garage <-> station pairs."""
    if a == b:
        return 0
    # (1) garage special-case FIRST (hard rule)
    if ((a == GARAGE_NODE and b in GARAGE_4MIN_DESTS) or
        (b == GARAGE_NODE and a in GARAGE_4MIN_DESTS)):
        return 4
    # (2) otherwise DM_MAT
    if DM_MAT is not None:
        rec=DM_MAT.get((a,b))
        if rec and "t_min_min" in rec: return int(rec["t_min_min"])
    # (3) safe fallback
    return 40

def _dist_km(a,b,line=None):
    if line is not None and DM_SERVICE is not None:
        rec=DM_SERVICE.get((line,a,b))
        if rec and "distance_m" in rec: return float(rec["distance_m"])/1000.0
    if DM_MAT is not None:
        rec=DM_MAT.get((a,b))
        if rec and "distance_m" in rec: return float(rec["distance_m"])/1000.0
    return 10.0

def _energy_move_kwh(a,b,line=None):
    return _dist_km(a,b,line=line) * TRACTION_KWH_PER_KM

def _first_breach(df):
    bad=[]
    if "_soc_after" in df.columns:  bad = df.index[df["_soc_after"]  < SOC_FLOOR_KWH].tolist()
    if not bad and "_soc_before" in df.columns: bad = df.index[df["_soc_before"] < SOC_FLOOR_KWH].tolist()
    return bad[0] if bad else None

def _prev_service_idx(df,bus,tmin):
    dfb=df[(df["bus"]==bus)&(df["_class"]=="service")&(df["_end_min"]<=tmin)]
    return int(dfb.index[-1]) if not dfb.empty else None

def _neighbor_prev(df,bus,start_min):
    dfb=df[df["bus"]==bus]
    prevs=dfb[dfb["_end_min"]<=start_min]
    pi=int(prevs.index[-1]) if not prevs.empty else None
    p_end  = int(dfb.loc[pi,"_end_min"])   if pi is not None else 0
    p_node = dfb.loc[pi,"_end_node"]       if pi is not None else GARAGE_NODE
    return (pi,p_end,p_node)

def _next_hard_event(df,bus,t_from_min):
    d=df[(df["bus"]==bus) & (df["_start_min"]>=t_from_min)]
    d=d[d["_class"].isin(["service","material"])].sort_values("_start_min")
    if d.empty: return None, 24*60, GARAGE_NODE
    i=int(d.index[0]); return i, int(d.loc[i,"_start_min"]), d.loc[i,"_start_node"]

def _next_service_event(df,bus,t_from_min):
    d=df[(df["bus"]==bus) & (df["_class"]=="service") & (df["_start_min"]>=t_from_min)].sort_values("_start_min")
    if d.empty: return None, 24*60, GARAGE_NODE
    i=int(d.index[0]); return i, int(d.loc[i,"_start_min"]), d.loc[i,"_start_node"]

def _fits_window(p_end,p_node, s_start,s_node, s_end,e_node, n_start,n_node):
    tt1=_tt_min(p_node,s_node); tt2=_tt_min(e_node,n_node)
    return (p_end + tt1 <= s_start) and (s_end + tt2 <= n_start)

def _insert_idle(df,bus,node,s_min,e_min):
    """Insert idle with 5 kW drain."""
    dur = int(max(0, e_min - s_min))
    if dur <= 0: return df
    idle_kwh = IDLE_KW * (dur/60.0)
    return _append_row(df,{
        "bus":bus,"activity":"idle","_class":"idle",
        "start location":node,"end location":node,
        "_start_node":node,"_end_node":node,
        "start time":_min_to_hhmm(s_min),"end time":_min_to_hhmm(e_min),
        "_start_min":s_min,"_end_min":e_min,"_dur_min":dur,
        "energy consumption": idle_kwh
    })

def _drop_rows_in_window(df,bus,w_start,w_end):
    mask=(df["bus"]==bus) & (df["_class"].isin(["idle","charging","material","service"])) & ~(
         (df["_end_min"]<=w_start)|(df["_start_min"]>=w_end))
    return df.loc[~mask].copy()

def _validate_time(df):
    d=_order_bus_then_time(df)
    for bus,grp in d.groupby("bus",sort=False):
        prev_end=None; prev_node=None
        for _,r in grp.iterrows():
            s,e=int(r["_start_min"]),int(r["_end_min"])
            if prev_end is not None and s < prev_end - 1e-9:
                return False, f"OVERLAP bus {bus}: prev_end={prev_end} > start={s}"
            if prev_end is not None:
                tt=_tt_min(prev_node, r["_start_node"])
                if prev_end + tt > s + TOL_MIN:
                    return False, f"TELEPORT bus {bus}: need {tt}min {prev_node}->{r['_start_node']} but gap {s-prev_end}"
            prev_end=e; prev_node=r["_end_node"]
    return True,"OK"

def _breaches(df):
    soc_out=recompute_soc(df.copy())
    d=soc_out[0] if isinstance(soc_out,tuple) else soc_out
    d=_order_bus_then_time(d)
    cnt=0
    if "_soc_after"  in d.columns: cnt += int((d["_soc_after"]  < SOC_FLOOR_KWH).sum())
    if "_soc_before" in d.columns: cnt += int((d["_soc_before"] < SOC_FLOOR_KWH).sum())
    return d, cnt

def _soc_before_at_next_service(d, bus, t_from):
    dd=d[(d["bus"]==bus) & (d["_class"]=="service") & (d["_start_min"]>=t_from)].sort_values("_start_min")
    if dd.empty: return float("inf")
    i=int(dd.index[0])
    try: return float(dd.loc[i, "_soc_before"])
    except: return float("inf")

def _donor_candidates(df, svc):
    s_start,s_end=int(svc["_start_min"]),int(svc["_end_min"])
    s_node,e_node=svc["_start_node"],svc["_end_node"]

    all_buses=list(df["bus"].unique())
    prios=[b for b in PRIORITY_BUSES if b in all_buses]
    rest=[b for b in sorted(all_buses, key=lambda x:(pd.to_numeric(str(x),errors="coerce"),str(x))) if b not in prios]
    buses=prios+rest

    cands=[]
    for bus in buses:
        if bus == svc["bus"]: continue
        (pi,p_end,p_node)=_neighbor_prev(df,bus,s_start)
        ni, n_start, n_node = _next_hard_event(df,bus,s_end)
        if _fits_window(p_end,p_node,s_start,s_node,s_end,e_node,n_start,n_node):
            tt1=_tt_min(p_node,s_node); tt2=_tt_min(e_node,n_node)
            slack=(s_start-(p_end+tt1)) + ((n_start-tt2)-s_end)
            score=3*tt1+3*tt2+max(0,-slack)
            if bus in PRIORITY_BUSES: score -= 1000
            cands.append((bus,pi,n_start,n_node,score))
    cands.sort(key=lambda x:x[4])
    return cands

def _svc_key(row):
    return (row.get("line",pd.NA),row["_start_node"],row["_end_node"],int(row["_start_min"]),int(row["_end_min"]))

# Main optimization function
def optimize_busplan(input_df, progress_callback=None, debug=False):
    """
    Main optimization function that takes a busplan dataframe and returns optimized version.
    
    Args:
        input_df: pandas DataFrame with busplan data
        progress_callback: optional function to call with progress updates
        debug: boolean to enable debug output
        
    Returns:
        pandas DataFrame with optimized busplan
    """
    # Initialize plan
    plan = _norm(input_df.copy())
    swaps_per_bus = {}
    tried = set()
    applied = 0
    
    # Main optimization loop
    for it in range(1, MAX_ITERS+1):
        plan, cur_bad = _breaches(plan)
        bad_idx = _first_breach(plan)
        
        if progress_callback:
            progress_callback(f"Iteration {it}: Found {cur_bad} SoC breaches")
        
        if bad_idx is None:
            if progress_callback:
                progress_callback(f"✔ plan feasible (no <{SOC_FLOOR_KWH:.0f} kWh) — iterations={it-1}, swaps={applied}")
            break

        low_bus = plan.loc[bad_idx,"bus"]
        t_idx = _prev_service_idx(plan, low_bus, int(plan.loc[bad_idx,"_start_min"]))
        if t_idx is None or plan.at[t_idx,"_class"]!="service":
            if progress_callback:
                progress_callback("breach bus has no prior service — stopping.")
            break

        svc = plan.loc[t_idx].copy()               # immutable service row
        donors = _donor_candidates(plan, svc)
        
        if debug and progress_callback:
            progress_callback(f"Breach on bus {low_bus} at {_min_to_hhmm(int(plan.loc[bad_idx,'_start_min']))}; donors: {[d[0] for d in donors]}")

        committed=False
        for donor_bus, prev_i, n_start, n_node, _ in donors:
            key=_svc_key(svc)
            if (key,donor_bus) in tried: continue
            if swaps_per_bus.get(donor_bus,0) >= MAX_SWAPS_PER_BUS:
                continue
            tried.add((key,donor_bus))

            cand=plan.copy()
            s_start,s_end = int(svc["_start_min"]),int(svc["_end_min"])
            s_node,e_node = svc["_start_node"],svc["_end_node"]
            s_line=svc.get("line",pd.NA)

            # donor prev boundary
            p_end  = int(cand.loc[prev_i,"_end_min"]) if prev_i is not None else 0
            p_node = cand.loc[prev_i,"_end_node"]      if prev_i is not None else GARAGE_NODE
            tt1=_tt_min(p_node,s_node); tt2=_tt_min(e_node,n_node)
            mat1_s,mat1_e = p_end, p_end+tt1
            mat2_s,mat2_e = s_end, s_end+tt2

            # donor window: clear and place material+idle+service+material+idle
            cand=_drop_rows_in_window(cand, donor_bus, mat1_s, mat2_e)
            if p_node!=s_node and tt1>0:
                cand=_append_row(cand, {"bus":donor_bus,"activity":"material","_class":"material",
                    "start location":p_node,"end location":s_node,"_start_node":p_node,"_end_node":s_node,
                    "start time":_min_to_hhmm(mat1_s),"end time":_min_to_hhmm(mat1_e),
                    "_start_min":mat1_s,"_end_min":mat1_e,"_dur_min":tt1,
                    "energy consumption":_energy_move_kwh(p_node,s_node)})
            cand=_insert_idle(cand, donor_bus, s_node, mat1_e, s_start)

            svc_energy=float(svc["energy consumption"]) if "energy consumption" in svc and pd.notna(svc["energy consumption"]) \
                        else _energy_move_kwh(s_node,e_node, line=s_line)
            cand=_append_row(cand, {"bus":donor_bus,"activity":"service","_class":"service","line":s_line,
                "start location":s_node,"end location":e_node,"_start_node":s_node,"_end_node":e_node,
                "start time":svc["start time"],"end time":svc["end time"],
                "_start_min":s_start,"_end_min":s_end,"_dur_min":int(svc["_dur_min"]),
                "energy consumption":svc_energy})

            if e_node!=n_node and tt2>0:
                cand=_append_row(cand, {"bus":donor_bus,"activity":"material","_class":"material",
                    "start location":e_node,"end location":n_node,"_start_node":e_node,"_end_node":n_node,
                    "start time":_min_to_hhmm(mat2_s),"end time":_min_to_hhmm(mat2_e),
                    "_start_min":mat2_s,"_end_min":mat2_e,"_dur_min":tt2,
                    "energy consumption":_energy_move_kwh(e_node,n_node)})
            cand=_insert_idle(cand, donor_bus, n_node, mat2_e, n_start)

            # remove the service from the low-SoC bus
            cand=cand.drop(index=t_idx)

            # === low-SoC bus: we can leave right after its PREVIOUS event =========
            low_prev_i, low_prev_end, low_prev_node = _neighbor_prev(cand, low_bus, s_start)
            svc_i_next, svc_start, svc_node = _next_service_event(cand, low_bus, s_end)

            # wipe everything between low_prev_end and next service
            cand = _drop_rows_in_window(cand, low_bus, low_prev_end, svc_start)

            # to garage
            tt_prev_gar = _tt_min(low_prev_node, GARAGE_NODE)
            gar_arr     = low_prev_end + tt_prev_gar
            if tt_prev_gar>0:
                cand=_append_row(cand, {"bus":low_bus,"activity":"material","_class":"material",
                    "start location":low_prev_node,"end location":GARAGE_NODE,
                    "_start_node":low_prev_node,"_end_node":GARAGE_NODE,
                    "start time":_min_to_hhmm(low_prev_end),"end time":_min_to_hhmm(gar_arr),
                    "_start_min":low_prev_end,"_end_min":gar_arr,"_dur_min":tt_prev_gar,
                    "energy consumption":_energy_move_kwh(low_prev_node,GARAGE_NODE)})

            # recompute SoC up to charge start to know how much we can add
            cand_pre, _ = _breaches(ensure_nodes(_order_bus_then_time(cand)))
            dsub = cand_pre[(cand_pre["bus"]==low_bus)&(cand_pre["_end_min"]<=gar_arr)].sort_values("_end_min")
            soc_at_charge_start = float(dsub.iloc[-1]["_soc_after"]) if not dsub.empty and "_soc_after" in dsub.columns else 0.0

            # return leg time and must-depart
            tt_back_to_svc = _tt_min(GARAGE_NODE, svc_node) if svc_i_next is not None else 0
            must_depart    = (svc_start - tt_back_to_svc) if svc_i_next is not None else 24*60
            window_min     = max(0, must_depart - gar_arr)

            # target energy to reach 90% cap (don't exceed time window; ensure >= min charge time)
            energy_to_cap_kwh = max(0.0, SOC_CAP_KWH - soc_at_charge_start)
            dur_need_min = math.ceil(energy_to_cap_kwh * 60.0 / CHARGER_KW) if CHARGER_KW>0 else MIN_CHARGE_MIN
            charge_dur   = max(MIN_CHARGE_MIN, min(window_min, dur_need_min))
            charge_end   = gar_arr + charge_dur
            charge_kwh   = -CHARGER_KW * (charge_dur/60.0)  # negative = energy gained

            # charging row
            cand=_append_row(cand, {"bus":low_bus,"activity":"charging","_class":"charging",
                "start location":GARAGE_NODE,"end location":GARAGE_NODE,
                "_start_node":GARAGE_NODE,"_end_node":GARAGE_NODE,
                "start time":_min_to_hhmm(gar_arr),"end time":_min_to_hhmm(charge_end),
                "_start_min":gar_arr,"_end_min":charge_end,"_dur_min":charge_dur,
                "energy consumption": charge_kwh})

            # back to next service start + idle to its start
            if svc_i_next is not None and GARAGE_NODE!=svc_node and tt_back_to_svc>0:
                back_s, back_e = charge_end, charge_end + tt_back_to_svc
                if back_e > svc_start + TOL_MIN:
                    continue
                cand=_append_row(cand, {"bus":low_bus,"activity":"material","_class":"material",
                    "start location":GARAGE_NODE,"end location":svc_node,
                    "_start_node":GARAGE_NODE,"_end_node":svc_node,
                    "start time":_min_to_hhmm(back_s),"end time":_min_to_hhmm(back_e),
                    "_start_min":back_s,"_end_min":back_e,"_dur_min":tt_back_to_svc,
                    "energy consumption":_energy_move_kwh(GARAGE_NODE,svc_node)})
                cand=_insert_idle(cand, low_bus, svc_node, back_e, svc_start)

            # validate timing
            cand=_order_bus_then_time(ensure_nodes(cand))
            ok,why=_validate_time(cand)
            if not ok:
                continue

            # recompute SoC + accept criteria
            cand, after_bad = _breaches(cand)
            donor_next_soc = _soc_before_at_next_service(cand, donor_bus, s_end)
            low_next_soc   = _soc_before_at_next_service(cand, low_bus, s_end)
            local_ok = (donor_next_soc >= SOC_FLOOR_KWH) and (low_next_soc >= SOC_FLOOR_KWH)

            if after_bad < cur_bad and local_ok:
                plan=cand
                swaps_per_bus[donor_bus]=swaps_per_bus.get(donor_bus,0)+1
                applied += 1
                if progress_callback:
                    progress_callback(f"swap #{applied} — donor {donor_bus} took line {s_line} "
                          f"{s_node}->{e_node} {_min_to_hhmm(s_start)}–{_min_to_hhmm(s_end)} | "
                          f"breaches {cur_bad}->{after_bad}")
                committed=True
                break

        if not committed:
            if progress_callback:
                progress_callback("no donor reduced total breaches while meeting local SoC guards; moving on.")
            continue

    return _order_bus_then_time(plan)