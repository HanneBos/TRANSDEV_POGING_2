import streamlit as st
import base64
import pandas as pd
import io
from diagnostics_final_checker import run_diagnostics_final

st.set_page_config(page_title="Transdev Optimization Tool", page_icon=":bus:")

def set_bg(image_path):
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

def add_logo(logo_path, width=250):
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

bg_path = r"C:\Users\20212160\OneDrive\Documenten\Fontys\Jaar 2\Project 5\Week 6\bus_streamlit_proef4.png"
logo_path = r"C:\Users\20212160\OneDrive\Documenten\Fontys\Jaar 2\Project 5\Week 6\transdev_logo_2018.png"

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