import streamlit as st
import base64
import pandas as pd
import io
import os
from datetime import timedelta

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
      - busses (int)
      - energy_consumed_kwh (float|None)
      - service_minutes (int)
      - total_minutes (int)
    """
    if df is None:
        return {'material_trips': None, 'idle_minutes': None, 'busses': None,
                'energy_consumed_kwh': None, 'service_minutes': None, 'total_minutes': None}
    material_trips = int((df['activity'].astype(str).str.lower().str.strip() == 'material trip').sum()) if 'activity' in df.columns else 0
    idle_minutes = compute_total_activity_minutes(df, activity_value='idle', activity_col='activity')
    service_minutes = compute_total_activity_minutes(df, activity_value='service', activity_col='activity', substring_match=True)
    total_minutes = compute_total_time_minutes(df)
    busses = int(df['bus'].nunique()) if 'bus' in df.columns else 0
    energy_consumed = compute_total_energy_consumed_kwh(df)
    return {'material_trips': material_trips,
            'idle_minutes': idle_minutes,
            'busses': busses,
            'energy_consumed_kwh': energy_consumed,
            'service_minutes': service_minutes,
            'total_minutes': total_minutes}

def fmt_val(v, suffix=""):
    return "N/A" if v is None else f"{v}{suffix}"

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
current_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level from pages/
logo_path = os.path.join(current_dir, "transdev_logo_2018.png")
try:
    add_logo(logo_path)
except FileNotFoundError:
    # Fallback: Display text logo
    st.markdown("### 🚌 Transdev KPI Dashboard")

bg_path = os.path.join(current_dir, "bus_streamlit_proef4.png")
set_bg(bg_path)

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
        text-align: center;
        font-size: 1.2em;
        color: #31333f;
        margin-bottom: 1.5em;
    }
    </style>
    <div class='nowrap-title'>KPI's</div>
        .
    </div>
""", unsafe_allow_html=True)

orig_df = st.session_state.get('original_df', None)
opt_df = st.session_state.get('optimized_df', None)

orig_kpis = compute_kpis(orig_df)
opt_kpis = compute_kpis(opt_df)

st.session_state['kpi_material_trips_original'] = orig_kpis['material_trips']
st.session_state['kpi_idle_time_original'] = orig_kpis['idle_minutes']
st.session_state['kpi_busses_used_original'] = orig_kpis['busses']
st.session_state['kpi_energy_consumed_original'] = orig_kpis['energy_consumed_kwh']
st.session_state['kpi_service_minutes_original'] = orig_kpis['service_minutes']

st.session_state['kpi_material_trips_optimized'] = opt_kpis['material_trips']
st.session_state['kpi_idle_time_optimized'] = opt_kpis['idle_minutes']
st.session_state['kpi_busses_used_optimized'] = opt_kpis['busses']
st.session_state['kpi_energy_consumed_optimized'] = opt_kpis['energy_consumed_kwh']
st.session_state['kpi_service_minutes_optimized'] = opt_kpis['service_minutes']

orig_eff = compute_efficiency(orig_kpis['service_minutes'], orig_kpis['total_minutes'])
opt_eff = compute_efficiency(opt_kpis['service_minutes'], opt_kpis['total_minutes'])

col_orig_header, col_opt_header = st.columns([1, 1])
with col_orig_header:
    st.subheader("Original")
with col_opt_header:
    st.subheader("Optimized")

kpi_rows = [
    ("Material Trips", fmt_val(orig_kpis['material_trips']), fmt_val(opt_kpis['material_trips'])),
    ("Idle time", fmt_val(orig_kpis['idle_minutes'], " min"), fmt_val(opt_kpis['idle_minutes'], " min")),
    ("Unique buses", fmt_val(orig_kpis['busses']), fmt_val(opt_kpis['busses'])),
    ("Total energy consumed (kWh)", fmt_val(orig_kpis['energy_consumed_kwh']), fmt_val(opt_kpis['energy_consumed_kwh'])),
    ("Service time (%)", fmt_val(orig_eff, "%"), fmt_val(opt_eff, "%"))
]

for label, left, right in kpi_rows:
    c1, c2 = st.columns([1,1])
    with c1:
        st.metric(label=label, value=left)
    with c2:
        st.metric(label=label, value=right)

st.info("KPIs computed from session data. Upload & optimize a plan first if values show N/A.")
