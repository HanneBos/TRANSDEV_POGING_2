import streamlit as st
import base64
import pandas as pd
import io
import os
from datetime import timedelta
import altair as alt

# Set page config
st.set_page_config(page_title="KPI's - Transdev", page_icon=":bus:")

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
        st.markdown("### 🚌 Transdev KPI Dashboard")

def compute_total_activity_minutes(df: pd.DataFrame,
                                   activity_value: str = "idle",
                                   activity_col: str = "activity",
                                   start_col: str = "start time",
                                   end_col: str = "end time",
                                   substring_match: bool = False) -> int:
    """
    Sum durations (in minutes) for rows where activity matches activity_value.
    Assumes columns: 'start time', 'end time', 'activity'.
    """
    if df is None or activity_col not in df.columns:
        return 0
    act = df[activity_col].astype(str).str.lower().str.strip()
    key = activity_value.lower().strip()
    mask = act.str.contains(key, na=False) if substring_match else (act == key)
    sel = df[mask].copy()
    if sel.empty or start_col not in sel.columns or end_col not in sel.columns:
        return 0

    sel[start_col] = pd.to_datetime(sel[start_col].astype(str), errors="coerce", format="%H:%M:%S")
    sel[end_col] = pd.to_datetime(sel[end_col].astype(str), errors="coerce", format="%H:%M:%S")
    sel[start_col] = pd.to_datetime(sel[start_col].astype(str), errors="coerce")
    sel[end_col] = pd.to_datetime(sel[end_col].astype(str), errors="coerce")
    sel = sel.dropna(subset=[start_col, end_col])
    if sel.empty:
        return 0

    durations = (sel[end_col] - sel[start_col]).apply(
        lambda d: d if pd.notna(d) and d.total_seconds() >= 0 else (d + pd.Timedelta(days=1) if pd.notna(d) else pd.NaT)
    ).dropna()
    if durations.empty:
        return 0
    total_seconds = durations.dt.total_seconds().sum()
    return int(round(total_seconds / 60.0))

def compute_total_time_minutes(df: pd.DataFrame,
                               start_col: str = "start time",
                               end_col: str = "end time") -> int:
    """
    Sum durations (in minutes) across all rows with valid start/end.
    Assumes columns: 'start time', 'end time'.
    """
    if df is None or start_col not in df.columns or end_col not in df.columns:
        return 0
    tmp = df[[start_col, end_col]].copy()
    tmp[start_col] = pd.to_datetime(tmp[start_col].astype(str), errors="coerce", format="%H:%M:%S")
    tmp[end_col] = pd.to_datetime(tmp[end_col].astype(str), errors="coerce", format="%H:%M:%S")
    tmp[start_col] = pd.to_datetime(tmp[start_col].astype(str), errors="coerce")
    tmp[end_col] = pd.to_datetime(tmp[end_col].astype(str), errors="coerce")
    tmp = tmp.dropna(subset=[start_col, end_col])
    if tmp.empty:
        return 0
    durations = (tmp[end_col] - tmp[start_col]).apply(
        lambda d: d if pd.notna(d) and d.total_seconds() >= 0 else (d + pd.Timedelta(days=1) if pd.notna(d) else pd.NaT)
    ).dropna()
    if durations.empty:
        return 0
    total_seconds = durations.dt.total_seconds().sum()
    return int(round(total_seconds / 60.0))

def compute_total_energy_consumed_kwh(df: pd.DataFrame, col: str = "energy consumption") -> float | None:
    """
    Sum positive values in 'energy consumption' (kWh). Handles comma decimals and whitespace.
    Returns None if column missing or no numeric values.
    """
    if df is None or col not in df.columns:
        return None
    ser = df[col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False)
    nums = pd.to_numeric(ser, errors='coerce').dropna()
    if nums.empty:
        return None
    consumed = nums[nums > 0].sum(skipna=True)
    return float(consumed)

def compute_kpis(df: pd.DataFrame):
    """
    Returns dict with keys:
      - material_trips (int)
      - idle_minutes (int)
      - buses (int)
      - energy_consumed_kwh (float|None)
      - service_minutes (int)
      - total_minutes (int)
    """
    if df is None:
        return {'material_trips': None, 'idle_minutes': None, 'buses': None,
                'energy_consumed_kwh': None, 'service_minutes': None, 'total_minutes': None}
    material_trips = int((df['activity'].astype(str).str.lower().str.strip() == 'material trip').sum()) if 'activity' in df.columns else 0
    idle_minutes = compute_total_activity_minutes(df, activity_value='idle', activity_col='activity')
    service_minutes = compute_total_activity_minutes(df, activity_value='service', activity_col='activity', substring_match=True)
    total_minutes = compute_total_time_minutes(df)
    buses = int(df['bus'].nunique()) if 'bus' in df.columns else 0
    energy_consumed = compute_total_energy_consumed_kwh(df)
    return {'material_trips': material_trips,
            'idle_minutes': idle_minutes,
            'buses': buses,
            'energy_consumed_kwh': energy_consumed,
            'service_minutes': service_minutes,
            'total_minutes': total_minutes}

def fmt_val(v, suffix=""):
    return "N/A" if v is None else f"{v}{suffix}"

def compute_energy_time_series_cumulative(df: pd.DataFrame,
                                          time_col: str = "start time",
                                          energy_col: str = "energy consumption",
                                          freq: str = "H") -> pd.DataFrame | None:
    
    if df is None or time_col not in df.columns or energy_col not in df.columns:
        return None

    times = pd.to_datetime(df[time_col].astype(str), errors="coerce", format="%H:%M:%S")
    if times.isna().all():
        times = pd.to_datetime(df[time_col].astype(str), errors="coerce")

    ser_energy = df[energy_col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False)
    energy = pd.to_numeric(ser_energy, errors="coerce")

    mask = times.notna() & energy.notna()
    if not mask.any():
        return None

    tmp = pd.DataFrame({"time": times[mask], "energy": energy[mask]}).copy()
    tmp = tmp[tmp["energy"] > 0]
    if tmp.empty:
        return None

    if tmp["time"].dt.year.min() == 1900:
        base = pd.Timestamp("2020-01-01")
        tmp["time"] = tmp["time"].dt.time.apply(lambda t: pd.Timestamp.combine(base, t))

    tmp["bucket"] = tmp["time"].dt.floor(freq)
    agg = tmp.groupby("bucket", as_index=False)["energy"].sum().rename(columns={"bucket": "time", "energy": "energy_kwh"})
    agg = agg.sort_values("time").reset_index(drop=True)
    agg["cumulative_kwh"] = agg["energy_kwh"].cumsum()
    return agg

def compute_service_time_percentage_series(df: pd.DataFrame,
                                           start_col: str = "start time",
                                           end_col: str = "end time",
                                           activity_col: str = "activity",
                                           freq: str = "H") -> pd.DataFrame | None:

    if df is None or start_col not in df.columns or end_col not in df.columns or activity_col not in df.columns:
        return None

    s = pd.to_datetime(df[start_col].astype(str), errors="coerce", format="%H:%M:%S")
    e = pd.to_datetime(df[end_col].astype(str), errors="coerce", format="%H:%M:%S")
    # fallback generic parse
    s = pd.to_datetime(s.astype(str), errors="coerce")
    e = pd.to_datetime(e.astype(str), errors="coerce")

    mask = s.notna() & e.notna()
    if not mask.any():
        return None

    tmp = pd.DataFrame({
        "start": s[mask].reset_index(drop=True),
        "end": e[mask].reset_index(drop=True),
        "activity": df.loc[mask, activity_col].astype(str).reset_index(drop=True)
    }).copy()

    # durations, handle overnight spans
    def _dur_minutes(row):
        d = row["end"] - row["start"]
        if pd.isna(d):
            return 0.0
        secs = d.total_seconds()
        if secs < 0:
            secs += 24 * 3600
        return max(secs / 60.0, 0.0)

    tmp["duration_min"] = tmp.apply(_dur_minutes, axis=1)
    tmp = tmp[tmp["duration_min"] > 0].copy()
    if tmp.empty:
        return None

    # mark service rows (substring match)
    tmp["is_service"] = tmp["activity"].str.lower().str.contains("service", na=False)

    # normalize times with base date if necessary
    if tmp["start"].dt.year.min() == 1900:
        base = pd.Timestamp("2020-01-01")
        tmp["start"] = tmp["start"].dt.time.apply(lambda t: pd.Timestamp.combine(base, t))

    # bucket by start time
    tmp["bucket"] = tmp["start"].dt.floor(freq)

    agg = tmp.groupby("bucket", as_index=False).agg(
        service_min=("duration_min", lambda x: x[tmp.loc[x.index, "is_service"]].sum() if len(x) else 0.0),
        total_min=("duration_min", "sum")
    ).rename(columns={"bucket": "time"})

    if agg.empty:
        return None

    agg = agg.sort_values("time").reset_index(drop=True)
    agg["cum_service_min"] = agg["service_min"].cumsum()
    agg["cum_total_min"] = agg["total_min"].cumsum()
    # avoid division by zero
    agg["cumulative_service_pct"] = agg.apply(
        lambda r: (100.0 * r["cum_service_min"] / r["cum_total_min"]) if r["cum_total_min"] > 0 else 0.0,
        axis=1
    )
    return agg[["time", "service_min", "total_min", "cum_service_min", "cum_total_min", "cumulative_service_pct"]]


def compute_gantt_df(df: pd.DataFrame, plan_label: str = "Plan",
                     start_col: str = "start time", end_col: str = "end time",
                     bus_col: str = "bus", activity_col: str = "activity") -> pd.DataFrame | None:
    """
    Bouw een DataFrame geschikt voor een Gantt chart.
    Lost midnight bug op (23:50 -> 00:10) en behoudt correcte end-datum bij normalisatie.
    """
    if df is None:
        return None
    for c in (start_col, end_col, bus_col, activity_col):
        if c not in df.columns:
            return None

    start = pd.to_datetime(df[start_col].astype(str), errors="coerce", format="%H:%M:%S")
    end = pd.to_datetime(df[end_col].astype(str), errors="coerce", format="%H:%M:%S")

    # fallback parse
    start = pd.to_datetime(start.astype(str), errors="coerce")
    end = pd.to_datetime(end.astype(str), errors="coerce")

    mask = start.notna() & end.notna() & df[bus_col].notna()
    if not mask.any():
        return None

    tmp = pd.DataFrame({
        "start": start[mask].reset_index(drop=True),
        "end": end[mask].reset_index(drop=True),
        "bus": df.loc[mask, bus_col].astype(str).reset_index(drop=True),
        "activity": df.loc[mask, activity_col].astype(str).reset_index(drop=True)
    }).copy()

    # fix overnight spans: if end earlier than start, add one day to end (preserve date offset)
    tmp["end"] = tmp.apply(
        lambda r: r["end"] + pd.Timedelta(days=1) if r["end"] < r["start"] else r["end"],
        axis=1
    )

    # Normaliseer naar vaste datum zodat Altair goed kan plotten.
    # When converting time-only datetimes to a base date, preserve any +1-day offset.
    if tmp["start"].dt.year.min() == 1900:
        base = pd.Timestamp("2020-01-01")
        # extract time components
        tmp_start_times = tmp["start"].dt.time
        tmp_end_times = tmp["end"].dt.time
        tmp["start"] = tmp_start_times.apply(lambda t: pd.Timestamp.combine(base, t))
        tmp["end"] = tmp_end_times.apply(lambda t: pd.Timestamp.combine(base, t))
        # if end is strictly before start (overnight), add one day to end
        # (use '<' not '<=' so zero-length rows with start==end are not turned into 24h spans)
        tmp.loc[tmp["end"] < tmp["start"], "end"] += pd.Timedelta(days=1)


    tmp["Plan"] = plan_label
    return tmp.reset_index(drop=True)

def compute_efficiency(service_min: int | None, total_min: int | None) -> float | None:
    """
    Percentage of time that is 'service' time.
    service_min and total_min come from compute_kpis(...) as 'service_minutes' and 'total_minutes'.
    """
    if service_min is None or total_min is None or total_min == 0:
        return None
    return round(100.0 * service_min / total_min, 1)

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
        text-align: left;
        display: block;
        margin: 0 auto;
        color: #31333f;
    }
    .subtitle {
        text-align: left;
        font-size: 1.2em;
        color: #4a4a4a;
        margin-top: 0.5em;
        margin-bottom: 1.5em;
    }
    /* Improve expander content readability - same as other pages */
    .stExpander [data-testid="stExpanderDetails"] {
        background-color: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(5px) !important;
        padding: 15px !important;
        border-radius: 5px !important;
        margin-top: 5px !important;
    }
    </style>
    <div class='nowrap-title'>KPI's</div>
    <div class='subtitle'>KPI's are computed from session data. Upload & optimize a plan first if values show N/A.</div>
""", unsafe_allow_html=True)

with st.expander("**How to use the KPI Calculations tool**"):
        st.markdown("""
        ### How to use:

        1. **Run the Optimized Busplan Generator first**:  
             This page depends on data stored from the Optimized Busplan Generator.  
             If you haven't uploaded and generated an optimized busplan yet, the KPIs here will display as **N/A**.
        2. **Return to this page**: After generating your optimized plan, navigate here to explore and compare key performance metrics.
        3. **Review KPIs**: View side-by-side comparisons between the **Original** and **Optimized** busplans.

        ### What you'll see:
        - **KPI Overview**: Comparison of Original vs Optimized plans for key metrics such as:  
            - **Material Trips**  
            - **Idle Time**  
            - **Unique Buses**  
            - **Total Energy Consumed (kWh)**  
            - **Service Time (%)**  
            - **Violations Found**
        - **Busplan Summary**: Compact tables summarizing both the Original and Optimized schedules for quick reference.
        - **Visual Insights**: Graphs showing cumulative energy consumption and service time percentages over time.


        ### What the KPIs represent:
        - **Material Trips**: Trips where the bus runs without passengers.
        - **Idle Time**: Total idle duration across all buses.  
        - **Unique Buses**: Total number of distinct buses used in the plan.  
        - **Total Energy Consumed (kWh)**: Combined energy use during operations.  
        - **Service Time (%)**: Percentage of total schedule time where the bus is carrying passengers. 
        - **Violations Found**: Total issues detected in the plan

        ### Interpreting Results:
        - **Lower is better:** Material trips, idle time, energy consumption, violations found
        - **Higher is better:** Service time percentage
        - **Depends on needs:** Number of unique buses (fewer = more efficient, but may impact service)
        """)


orig_df = st.session_state.get('original_df', None)
opt_df = st.session_state.get('optimized_df', None)

orig_kpis = compute_kpis(orig_df)
opt_kpis = compute_kpis(opt_df)

st.session_state['kpi_material_trips_original'] = orig_kpis['material_trips']
st.session_state['kpi_idle_time_original'] = orig_kpis['idle_minutes']
st.session_state['kpi_buses_used_original'] = orig_kpis['buses']
st.session_state['kpi_energy_consumed_original'] = orig_kpis['energy_consumed_kwh']
st.session_state['kpi_service_minutes_original'] = orig_kpis['service_minutes']

st.session_state['kpi_material_trips_optimized'] = opt_kpis['material_trips']
st.session_state['kpi_idle_time_optimized'] = opt_kpis['idle_minutes']
st.session_state['kpi_buses_used_optimized'] = opt_kpis['buses']
st.session_state['kpi_energy_consumed_optimized'] = opt_kpis['energy_consumed_kwh']
st.session_state['kpi_service_minutes_optimized'] = opt_kpis['service_minutes']

orig_eff = compute_efficiency(orig_kpis['service_minutes'], orig_kpis['total_minutes'])
opt_eff = compute_efficiency(opt_kpis['service_minutes'], opt_kpis['total_minutes'])

def _values_equal(a, b, tol: float = 1e-6) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return str(a) == str(b)

_kpi_keys = ['material_trips', 'idle_minutes', 'buses', 'energy_consumed_kwh', 'service_minutes']
_all_same = all(_values_equal(orig_kpis.get(k), opt_kpis.get(k)) for k in _kpi_keys) and _values_equal(orig_eff, opt_eff)

if _all_same:
    st.info("No KPI differences found between Original and Optimized plans.")

st.markdown("---")

col_orig_header, col_opt_header = st.columns([1, 1])
with col_orig_header:
    st.subheader("Original")
with col_opt_header:
    st.subheader("Optimized")

kpi_rows = [
    ("Material Trips", fmt_val(orig_kpis['material_trips']), fmt_val(opt_kpis['material_trips'])),
    ("Idle time", fmt_val(orig_kpis['idle_minutes'], " min"), fmt_val(opt_kpis['idle_minutes'], " min")),
    ("Unique buses", fmt_val(orig_kpis['buses']), fmt_val(opt_kpis['buses'])),
    ("Total energy consumed (kWh)", fmt_val(orig_kpis['energy_consumed_kwh']), fmt_val(opt_kpis['energy_consumed_kwh'])),
    ("Service time (%)", fmt_val(orig_eff, "%"), fmt_val(opt_eff, "%")),
    ("Violations found", fmt_val(st.session_state.get('amount_violations_original', None)), fmt_val(st.session_state.get('amount_violations_optimized', None))),
]

for label, left, right in kpi_rows:
    c1, c2 = st.columns([1,1])
    with c1:
        st.metric(label=label, value=left)
    with c2:
        st.metric(label=label, value=right)

def _is_number(v):
    return isinstance(v, (int, float))

def _equal_values(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return str(a) == str(b)

orig_ts = compute_energy_time_series_cumulative(orig_df)
opt_ts = compute_energy_time_series_cumulative(opt_df)

st.markdown("---")
st.markdown("### Gantt chart bus plan <span style='font-size:0.8em; color:#555;'>(Original vs Optimized busplan)</span>", unsafe_allow_html=True)

g_orig = compute_gantt_df(orig_df, "Original")
g_opt = compute_gantt_df(opt_df, "Optimized")

if g_orig is None and g_opt is None:
    st.info("No bus schedule data available for Gantt chart (need start time, end time, bus, activity).")
else:
    # define desired plotting window: 04:00 -> 03:59 next day
    base = pd.Timestamp("2020-01-01")
    window_start = base + pd.Timedelta(hours=4)
    window_end = base + pd.Timedelta(days=1) + pd.Timedelta(hours=3, minutes=59)

    def _shift_rows_to_window(df):
        if df is None or df.empty:
            return None
        df2 = df.copy()
        # shift rows that start before window_start into the next day so axis runs 04:00->03:59
        mask = df2["start"] < window_start
        if mask.any():
            df2.loc[mask, ["start", "end"]] = df2.loc[mask, ["start", "end"]] + pd.Timedelta(days=1)
        return df2

    def _make_gantt_chart(df_plan, title):
        if df_plan is None or df_plan.empty:
            return None
        
        # Clean and normalize bus numbers to integers for consistent display
        df_plan = df_plan.copy()
        df_plan["bus"] = df_plan["bus"].astype(str).str.strip()
        
        # Convert bus numbers to integers where possible, keep as string otherwise
        def clean_bus_number(bus_str):
            try:
                # Try to convert to float first (handles "1.0"), then to int
                bus_float = float(bus_str)
                if bus_float.is_integer():
                    return str(int(bus_float))
                else:
                    return bus_str
            except (ValueError, TypeError):
                return bus_str
        
        df_plan["bus"] = df_plan["bus"].apply(clean_bus_number)
        
        # determine bus order (numeric sorting for proper 1,2,3...10,11,20 order)
        try:
            # Get unique bus numbers and sort them numerically
            unique_buses = df_plan["bus"].unique()
            bus_order = sorted(unique_buses, key=lambda x: int(x) if str(x).isdigit() else float('inf'))
        except Exception:
            bus_order = list(df_plan["bus"].unique())
        
        # dynamic chart height so every bus label is visible
        n_buses = len(bus_order) or 1
        chart_height = max(30 * n_buses, 300)
        
        return (
            alt.Chart(df_plan)
            .mark_bar(size=12)
            .encode(
                x=alt.X("start:T",
                        title="Time",
                        scale=alt.Scale(domain=[window_start, window_end]),
                        axis=alt.Axis(format="%H:%M", tickCount=25, labelAngle=0)),
                x2=alt.X2("end:T"),
                y=alt.Y("bus:N",
                        title="Bus",
                        sort=bus_order,
                        axis=alt.Axis(labelAngle=0, labelFontSize=11, titleFontSize=12)),
                color=alt.Color("activity:N", 
                              title="Activity", 
                              legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip("bus:N", title="Bus"),
                    alt.Tooltip("activity:N", title="Activity"),
                    alt.Tooltip("start:T", title="Start", format="%H:%M"),
                    alt.Tooltip("end:T", title="End", format="%H:%M"),
                ]
            )
            .properties(title=title, height=chart_height)
            .interactive()
        )

    # Process data for both charts
    g_orig_shifted = _shift_rows_to_window(g_orig) if g_orig is not None else None
    g_opt_shifted = _shift_rows_to_window(g_opt) if g_opt is not None else None

    # Remove zero-length segments
    if g_orig_shifted is not None and not g_orig_shifted.empty:
        g_orig_shifted = g_orig_shifted[g_orig_shifted["end"] > g_orig_shifted["start"]].reset_index(drop=True)
    if g_opt_shifted is not None and not g_opt_shifted.empty:
        g_opt_shifted = g_opt_shifted[g_opt_shifted["end"] > g_opt_shifted["start"]].reset_index(drop=True)

    # Create side-by-side charts without gap
    col1, col2 = st.columns(2, gap="small")
    
    # Data verification check
    data_identical = False
    if g_orig is not None and g_opt is not None:
        try:
            # Check if the dataframes are identical
            data_identical = g_orig.equals(g_opt)
            if data_identical:
                st.warning("**Data Verification Alert**: Original and Optimized Gantt data appear to be identical. This might indicate that the same dataset is being used for both charts.")
        except Exception:
            pass
    
    with col1:
        if g_orig_shifted is not None and not g_orig_shifted.empty:
            chart_orig = _make_gantt_chart(g_orig_shifted, "Original")
            if chart_orig is not None:
                st.altair_chart(chart_orig, use_container_width=True)
        else:
            st.info("No Original bus schedule data available.")
    
    with col2:
        if g_opt_shifted is not None and not g_opt_shifted.empty:
            chart_opt = _make_gantt_chart(g_opt_shifted, "Optimized")
            if chart_opt is not None:
                st.altair_chart(chart_opt, use_container_width=True)
        else:
            st.info("No Optimized bus schedule data available.")
            
    # Additional debug info if data looks suspicious
    if data_identical:
        st.info("""
        **Possible causes for identical data:**
        - The optimization process hasn't been run yet
        - The same file was uploaded for both original and optimized
        - The optimization made no changes to the schedule
        - There's an issue with session state data storage
        
        Try re-running the Optimized Busplan Generator to ensure different data is being compared.
        """)

st.markdown("---")
st.markdown("### Cumulative energy consumed <span style='font-size:0.8em; color:#555;'>(Original vs Optimized busplan)</span>", unsafe_allow_html=True)

if orig_ts is None and opt_ts is None:
    st.info("No usable energy/time data available for cumulative plot.")
else:
    parts = []
    if orig_ts is not None:
        d = orig_ts.copy(); d["Plan"] = "Original"; parts.append(d)
    if opt_ts is not None:
        d = opt_ts.copy(); d["Plan"] = "Optimized"; parts.append(d)
    plot_df = pd.concat(parts, ignore_index=True)

    chart_cum = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("time:T",
                    title="Time",
                    axis=alt.Axis(format="%H:%M", labelAngle=0)), 
            y=alt.Y("cumulative_kwh:Q", title="Cumulative energy consumed (kWh)"),
            color=alt.Color("Plan:N", title="Plan"),
            tooltip=[alt.Tooltip("Plan:N"), alt.Tooltip("time:T", title="Time", format="%H:%M"), alt.Tooltip("cumulative_kwh:Q", title="Cumulative kWh")]
        )
    )

    st.altair_chart(chart_cum, use_container_width=True)

st.markdown("---")
st.markdown("### Service time % per hour <span style='font-size:0.8em; color:#555;'>(Original vs Optimized busplan)</span>", unsafe_allow_html=True)

orig_svc = compute_service_time_percentage_series(orig_df, freq="H")
opt_svc = compute_service_time_percentage_series(opt_df, freq="H")

if orig_svc is None and opt_svc is None:
    st.info("No service-time data available for hourly % plot.")
else:
    parts = []
    if orig_svc is not None:
        d = orig_svc.copy()
        d["pct"] = d.apply(lambda r: (100.0 * r["service_min"] / r["total_min"]) if r["total_min"] > 0 else 0.0, axis=1)
        d["Plan"] = "Original"
        parts.append(d[["time", "pct", "Plan"]])
    if opt_svc is not None:
        d = opt_svc.copy()
        d["pct"] = d.apply(lambda r: (100.0 * r["service_min"] / r["total_min"]) if r["total_min"] > 0 else 0.0, axis=1)
        d["Plan"] = "Optimized"
        parts.append(d[["time", "pct", "Plan"]])

    plot_df = pd.concat(parts, ignore_index=True)

    # shift bucket times before window_start to next day (so x runs 04:00 -> 03:59)
    mask_shift = plot_df["time"] < window_start
    if mask_shift.any():
        plot_df.loc[mask_shift, "time"] = plot_df.loc[mask_shift, "time"] + pd.Timedelta(days=1)

    # produce hour labels and ensure correct ordering
    plot_df["hour_label"] = plot_df["time"].dt.strftime("%H:%M")
    # sort by actual datetime to preserve chronological order
    hour_order = plot_df.sort_values("time")["hour_label"].unique().tolist()

    chart_hourly = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X('hour_label:N', title='Hour', sort=hour_order),
            y=alt.Y('pct:Q', title='Service time (%)', scale=alt.Scale(domain=[0,100])),
            color=alt.Color('Plan:N', title='Plan'),
            tooltip=[alt.Tooltip("Plan:N"), alt.Tooltip("hour_label:N", title="Hour"), alt.Tooltip("pct:Q", title="% Service")],
            xOffset='Plan:N'
        )
        .properties(height=320)
    )

    st.altair_chart(chart_hourly, use_container_width=True)

