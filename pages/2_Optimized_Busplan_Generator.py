import streamlit as st
import base64
import pandas as pd
import io
import os
from datetime import datetime
from busplan_optimizer import optimize_busplan, SOC_FLOOR_KWH

# Set page config
st.set_page_config(page_title="Optimized Busplan - Transdev", page_icon=":bus:")

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
    ### Instructions:
    1. **Upload File**: Click "Browse files" and select your Bus Planning Excel file
    2. **Review Data**: Check the preview to ensure your data loaded correctly  
    3. **Generate**: Click "Generate Optimized Busplan" to start the optimization process
    4. **Download**: Once complete, download your optimized busplan
    
    ### What the optimization does:
    - **Energy Management**: Optimizes bus routes to prevent battery depletion
    - **Smart Charging**: Automatically schedules charging sessions at the garage
    - **Service Swapping**: Redistributes services between buses to improve efficiency
    - **Feasibility Check**: Ensures all timing and energy constraints are met
    
    ### File Requirements:
    Your Excel file should contain columns for:
    - Bus number
    - Activity type (service, material, idle, charging)
    - Start/end locations and times
    - Line information
    - Energy consumption (optional - will be calculated if missing)
    
    ### Recommended Workflow:
    - **Direct Upload**: Upload your original busplan here for automatic optimization
    - **Two-Step Process**: First use the Feasibility Checker to analyze issues, then upload the validated busplan here for optimization
    - **Best Practice**: For complex plans, run Feasibility Checker first to understand specific problems
    """)

# Reset button
reset_col1, reset_col2, reset_col3 = st.columns([1, 1, 1])
with reset_col2:
    if st.button("**Reset Page**", type="secondary", help="Clear uploaded file and optimization results"):
        # Clear session state for this page
        for key in ['optimizer_result', 'optimizer_original_df', 'optimizer_file_uploaded']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Check if we have saved results
show_results = st.session_state.get('optimizer_result') is not None
file_uploaded = st.session_state.get('optimizer_file_uploaded', False)
original_df = st.session_state.get('optimizer_original_df')

if not show_results:
    # File upload
    uploaded_file = st.file_uploader("Upload Bus Planning Excel file", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} rows.")

            # Save original dataframe to session_state
            st.session_state['original_df'] = df.copy()
            st.session_state['optimizer_original_df'] = df.copy()

            # Show preview of uploaded data
            with st.expander("**Preview uploaded data**"):
                st.dataframe(df)
            
            # Optimization button
            if st.button("Generate Optimized Busplan", type="primary"):
                try:
                    # Progress placeholder
                    progress_placeholder = st.empty()
                    
                    # Progress callback function
                    def update_progress(message):
                        progress_placeholder.text(message)
                    
                    with st.spinner("Optimizing busplan... This may take a few moments."):
                        # Run optimization with progress callback
                        optimized_df = optimize_busplan(df, progress_callback=update_progress)
                    
                    progress_placeholder.success("Optimization completed successfully!")

                    # Save results to session state
                    st.session_state['optimized_df'] = optimized_df.copy()
                    st.session_state['optimizer_result'] = optimized_df.copy()
                    st.session_state['optimizer_file_uploaded'] = True
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during optimization: {str(e)}")
                    st.error("Please check your file format and try again.")
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.error("Please make sure you uploaded a valid Excel file with the correct format.")

    else:
        st.info("Please upload a Bus Planning Excel file to get started.")

if show_results:
    optimized_df = st.session_state['optimizer_result']
    original_df = st.session_state['optimizer_original_df']
    
    st.success("✅ Optimization completed successfully!")
    
    # Show results summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("**Original Plan Rows**", len(original_df))
    with col2:
        st.metric("**Optimized Plan Rows**", len(optimized_df))
    
    # Show preview of optimized data
    with st.expander("**Preview optimized busplan**", expanded=True):
        st.dataframe(optimized_df)
    
    # Prepare download
    output = io.BytesIO()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"BusPlanning_Optimized_{timestamp}.xlsx"
    
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
    
    # Show SoC summary if available
    if "_soc_before" in optimized_df.columns and "_soc_after" in optimized_df.columns:
        soc_data = pd.concat([optimized_df["_soc_before"], optimized_df["_soc_after"]])
        soc_min = float(soc_data.min(skipna=True))
        soc_breaches = int(((optimized_df["_soc_before"] < SOC_FLOOR_KWH).sum() + 
                          (optimized_df["_soc_after"] < SOC_FLOOR_KWH).sum()))
        
        st.info(f" **SoC Summary:** Minimum SoC: {soc_min:.2f} kWh | SoC breaches: {soc_breaches}")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.9em;">Transdev Optimization Tool - Powered by Advanced Bus Planning Algorithms</div>',
    unsafe_allow_html=True
)
