import streamlit as st
import base64
import pandas as pd
import io
import os

# Set page config
st.set_page_config(page_title="KPI - Transdev", page_icon=":bus:")

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
        text-align: center;
        display: block;
        margin: 0 auto;
    }
    </style>
    <div class='nowrap-title'>KPI Calculations</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center; font-size:1.1em; margin-bottom:1em;">Here will be the KPI calculations of the original and optimized busplan.</div>',
    unsafe_allow_html=True
)