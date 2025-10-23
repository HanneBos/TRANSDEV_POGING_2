import streamlit as st
import base64
import pandas as pd
import io
import os
from diagnostics_final_checker import run_diagnostics_final

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
    - **SoC Analysis**: Tracks battery State of Charge throughout each bus journey
    - **Violation Detection**: Identifies specific problems like battery depletion or timing conflicts
    - **Data Normalization**: Standardizes and sorts your bus plan data for analysis
    
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
    - **Violations table** = Shows specific issues and their locations
    - **SoC progression** = Battery levels throughout the day per bus
    - **Validated schedule** = Your original plan, cleaned and sorted
    """)

# Reset button
reset_col1, reset_col2, reset_col3 = st.columns([1, 1, 1])
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

            result = run_diagnostics_final(plan_df, dist_df, tt_df)
            
            # Save results to session state
            st.session_state['feasibility_result'] = result
            st.session_state['feasibility_files_uploaded'] = True
            
        st.rerun()
    else:
        st.info("Upload all required files to begin.")

if show_results:
    result = st.session_state['feasibility_result']
    rule_counts = result.get("rule_counts", pd.DataFrame())
    viol_df = result.get("violations", pd.DataFrame())
    soc_df = result.get("soc", pd.DataFrame())
    bp_sorted = result.get("bp_sorted", pd.DataFrame())

    if viol_df.empty:
        st.success("Feasible: YES (no violations).")
    else:
        st.error("Feasible: NO (see violations below).")

    st.subheader("Feasibility summary")
    if not rule_counts.empty:
        st.dataframe(rule_counts)
    else:
        st.success("No violations found! Plan is feasible.")

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
        st.success("No violations found!")

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
        st.info("No SOC data found.")

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
