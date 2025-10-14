import pandas as pd
import numpy as np
from datetime import datetime

# === CONFIG ===
LAYOVER_MIN     = 0.0
MIN_CHARGE_MIN  = 15.0
BATTERY_KWH     = 300.0
SOH_FRAC        = 0.90
SOC_START_FRAC  = 0.90
SOC_FLOOR_FRAC  = 0.10
SOC_CAP_FRAC    = 0.90
KWH_PER_KM      = 1.2
IDLE_KW         = 5.0
FAST_KW         = 450.0
SLOW_KW         = 60.0
CHARGER_LOC     = "ehvgar"

def to_minutes(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.hour*60 + val.minute + val.second/60.0
    s = str(val).strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            t = datetime.strptime(s, fmt)
            return t.hour*60 + t.minute + t.second/60.0
        except: pass
    try:
        t2 = pd.to_datetime(val)
        return t2.hour*60 + t2.minute + t2.second/60.0
    except: return np.nan

def dm_lookup(dm, u, v, line=None):
    if pd.isna(u) or pd.isna(v):
        return pd.Series({"min_travel_time": np.nan, "max_travel_time": np.nan, "distance_m": np.nan})
    uu = str(u).strip().lower()
    vv = str(v).strip().lower()
    base = dm[(dm["start"].astype(str).str.strip().str.lower() == uu) &
              (dm["end"].astype(str).str.strip().str.lower()   == vv)]
    if len(base) == 0:
        return pd.Series({"min_travel_time": np.nan, "max_travel_time": np.nan, "distance_m": np.nan})
    if "line" in dm.columns:
        dm_line = pd.to_numeric(base["line"], errors="coerce")
    else:
        dm_line = pd.Series([np.nan]*len(base), index=base.index)
    plan_line = pd.to_numeric(line, errors="coerce") if line is not None else np.nan
    chosen = base
    used_specific = False
    if not pd.isna(plan_line):
        sub_line = base[dm_line == plan_line]
        if len(sub_line) > 0:
            chosen = sub_line
            used_specific = True
    if not used_specific:
        sub_generic = base[dm_line.isna()]
        if len(sub_generic) > 0:
            chosen = sub_generic
    out = {
        "min_travel_time": pd.to_numeric(chosen["min_travel_time"], errors="coerce").median(),
        "max_travel_time": pd.to_numeric(chosen["max_travel_time"], errors="coerce").median() if "max_travel_time" in chosen.columns else np.nan,
        "distance_m":      pd.to_numeric(chosen["distance_m"],      errors="coerce").median()
    }
    return pd.Series(out)

def add_violation(viol, rule, severity, row, **kw):
    viol.append({
        "rule": rule, "severity": severity,
        "bus": row.get("bus") if isinstance(row, dict) else row.get("bus"),
        "activity": row.get("activity") if isinstance(row, dict) else row.get("activity"),
        "line": row.get("line") if isinstance(row, dict) else row.get("line"),
        "start time": row.get("start time") if isinstance(row, dict) else row.get("start time"),
        "end time": row.get("end time") if isinstance(row, dict) else row.get("end time"),
        "start location": row.get("start location") if isinstance(row, dict) else row.get("start location"),
        "end location": row.get("end location") if isinstance(row, dict) else row.get("end location"),
        **kw
    })

def classify(a):
    s = str(a).lower()
    if "charg" in s or "laad" in s: return "charging"
    if any(k in s for k in ["dead","mat","non-rev","materiaal","materieel"]): return "deadhead"
    if any(k in s for k in ["service","trip","revenue","rit","dienst"]): return "service"
    return "idle"

def run_diagnostics_final(plan_df, dist_df, tt_df):
    # -------- schema sanity
    need = ["bus","activity","start time","end time","start location","end location"]
    missing = [c for c in need if c not in plan_df.columns]
    if missing:
        raise ValueError(f"Bus Planning.xlsx is missing columns: {missing}")

    # -------- parse times & durations (cross-midnight fix)
    plan_df["_start_min"] = plan_df["start time"].apply(to_minutes)
    plan_df["_end_min"]   = plan_df["end time"].apply(to_minutes)
    plan_df["_dur_min"]   = plan_df["_end_min"] - plan_df["_start_min"]
    neg = plan_df["_dur_min"] < 0
    plan_df.loc[neg, "_end_min"] += 24*60
    plan_df.loc[neg, "_dur_min"] = plan_df.loc[neg, "_end_min"] - plan_df["_start_min"]

    # -------- classify activities
    plan_df["_class"] = plan_df["activity"].apply(classify)

    # -------- DM data & energy
    plan_df[["min_travel_time","max_travel_time","distance_m"]] = plan_df.apply(
        lambda r: dm_lookup(dist_df, r.get("start location"), r.get("end location"), r.get("line")), axis=1
    )
    plan_df["_distance_km"] = pd.to_numeric(plan_df["distance_m"], errors="coerce")/1000.0
    energy_series = pd.to_numeric(plan_df.get("energy consumption"), errors="coerce")
    fallback      = plan_df["_distance_km"] * KWH_PER_KM
    plan_df["_energy_kwh"] = energy_series.fillna(fallback).fillna(0.0)

    # -------- timeline
    bp_sorted = plan_df.sort_values(["bus","_start_min","_end_min","activity"]).reset_index(drop=True)

    # -------- rules
    viol = []

    # A1) Negative/missing durations (zero-minute allowed)
    bad = plan_df[(plan_df["_dur_min"] < 0) | plan_df["_dur_min"].isna()]
    for _, r in bad.iterrows():
        add_violation(viol, "NEG_OR_ZERO_DURATION", "error", r, value=float(r.get("_dur_min", np.nan)))

    # A1) Overlaps
    for bus_id, grp in bp_sorted.groupby("bus", sort=False):
        grp = grp.reset_index(drop=True)
        prev_end = -1e9
        for _, r in grp.iterrows():
            if r["_start_min"] < prev_end - 1e-9:
                add_violation(viol, "OVERLAP", "error", r, overlap_min=float(prev_end - r["_start_min"]))
            prev_end = max(prev_end, r["_end_min"])

    # A2) Strict spatial continuity (end == next start), except charge→charge at CHARGER_LOC
    for bus_id, grp in bp_sorted.groupby("bus", sort=False):
        g = grp.reset_index(drop=True)
        for i in range(1, len(g)):
            prev_endloc  = str(g.loc[i-1, "end location"]).strip().lower()
            curr_startloc= str(g.loc[i,   "start location"]).strip().lower()
            prev_type    = str(g.loc[i-1, "_class"])
            curr_type    = str(g.loc[i,   "_class"])
            if (prev_type == "charging" and curr_type == "charging" and
                prev_endloc == str(CHARGER_LOC).lower() and curr_startloc == str(CHARGER_LOC).lower()):
                continue
            if prev_endloc != curr_startloc:
                add_violation(viol, "CONTINUITY_BREAK", "error", g.loc[i-1],
                              next_start=g.loc[i, "start time"],
                              from_to=f"{g.loc[i-1,'end location']} -> {g.loc[i,'start location']}")

    # A3/A4) Duration bands — only for MOVING service/deadhead rows (start≠end)
    for _, r in plan_df.iterrows():
        typ = str(r["_class"])
        if typ not in ("service", "deadhead"):
            continue  # ignore idle/charging
        u, v = r["start location"], r["end location"]
        if pd.isna(u) or pd.isna(v):
            continue
        # Skip same-stop segments (no movement → no DM needed)
        if str(u).strip().lower() == str(v).strip().lower():
            continue
        dmin = r["_dur_min"]
        if pd.isna(dmin):
            continue
        mm = dm_lookup(dist_df, u, v, r.get("line"))
        lo = mm["min_travel_time"]
        hi = mm["max_travel_time"]
        if pd.isna(lo) or pd.isna(hi):
            add_violation(viol, "MISSING_DM", "error", r, from_to=f"{u}->{v}")
            continue
        if typ == "service":
            # allow +1 min on upper bound for service
            if (dmin < lo - 1e-6) or (dmin > hi + 1 + 1e-6):
                add_violation(viol, "SERVICE_DURATION_OUT_OF_BAND", "error", r,
                              duration_min=float(dmin), band=f"[{lo}, {hi}+1]")
        else:  # deadhead/material
            if (dmin < lo - 1e-6) or (dmin > hi + 1e-6):
                add_violation(viol, "DEADHEAD_DURATION_OUT_OF_BAND", "error", r,
                              duration_min=float(dmin), band=f"[{lo}, {hi}]")

    # Charging & SOC
    USABLE    = BATTERY_KWH * SOH_FRAC
    SOC_START = USABLE * SOC_START_FRAC
    SOC_FLOOR = USABLE * SOC_FLOOR_FRAC
    SOC_CAP   = USABLE * SOC_CAP_FRAC

    soc_rows = []
    for bus_id, grp in bp_sorted.groupby("bus", sort=False):
        soc = SOC_START
        for _, r in grp.iterrows():
            cls = r["_class"]; dur = float(r["_dur_min"] or 0.0)
            if cls in ("service","deadhead","idle"):
                soc -= float(r["_energy_kwh"] or 0.0)
            if cls == "charging":
                if dur < MIN_CHARGE_MIN - 1e-6:
                    add_violation(viol, "SHORT_CHARGE", "error", r,
                                  duration_min=dur, min_required=MIN_CHARGE_MIN)
                need_to_cap = max(0.0, SOC_CAP - soc)
                fast_gain   = min(need_to_cap, FAST_KW * (dur/60.0))
                soc        += fast_gain
                leftover_h  = max(0.0, dur/60.0 - fast_gain/FAST_KW) if FAST_KW>0 else 0.0
                if leftover_h > 1e-9:
                    soc += SLOW_KW * leftover_h
                if soc > SOC_CAP + 1e-6:
                    add_violation(viol, "CAP_ABOVE_90", "warning", r,
                                  soc_end=round(soc,1), cap=round(SOC_CAP,1))
                    soc = min(soc, SOC_CAP)
            if soc < SOC_FLOOR - 1e-6:
                add_violation(viol, "SOC_FLOOR", "error", r,
                              soc_end=round(soc,1), floor=round(SOC_FLOOR,1))
            soc_rows.append({
                "bus": r["bus"], "line": r.get("line"),
                "time_start": r["start time"], "time_end": r["end time"],
                "activity": r["activity"], "class": cls, "soc_kwh": round(float(soc),2)
            })

    viol_df = pd.DataFrame(viol)
    soc_df  = pd.DataFrame(soc_rows)

    # Final decision
    if len(viol_df):
        counts = (viol_df.groupby(["rule","severity"])
                  .size().reset_index(name="count")
                  .sort_values(["severity","rule"], ascending=[False, True]))
    else:
        counts = pd.DataFrame(columns=["rule","severity","count"])

    # Je kunt hier extra outputs toevoegen als je meer wilt tonen in Streamlit
    return {
        "rule_counts": counts,
        "violations": viol_df,
        "soc": soc_df,
        "bp_sorted": bp_sorted,  # optioneel, voor preview van de planning
        # voeg meer toe indien gewenst
    }