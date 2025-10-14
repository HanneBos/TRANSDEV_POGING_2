import streamlit as st
import base64
import pandas as pd
import io
import os
from diagnostics_final_checker import run_diagnostics_final

st.set_page_config(page_title="Transdev Optimization Tool", page_icon=":bus:")

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
            header[data-testid="stHeader"] {{
                display: none;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        # Fallback styling if images are not found
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
            header[data-testid="stHeader"] {
                display: none;
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

# Use relative paths that work in cloud deployment
current_dir = os.path.dirname(os.path.abspath(__file__))
bg_path = os.path.join(current_dir, "bus_streamlit_proef4.png")
logo_path = os.path.join(current_dir, "transdev_logo_2018.png")

set_bg(bg_path)
add_logo(logo_path)

st.markdown(
    """
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
    <div class='nowrap-title'>Transdev Optimization Tool</div>
    <div class='subtitle'>
        This tool supports the optimization of bus plan data, and enables adjustments to assess the feasibility and quality of a bus plan, identify areas for improvement, and establish a foundation for creating a more effective bus plan.<br>
        <br>Click the button below to get started.
    </div>
    """,
    unsafe_allow_html=True,
)


col1, col2, col3 = st.columns([2,1,2])
with col2:
    if st.button("START"):
        st.switch_page("pages/1_Feasibility_Checker.py")
