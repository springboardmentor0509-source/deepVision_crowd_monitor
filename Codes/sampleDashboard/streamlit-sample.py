import streamlit as st
from pathlib import Path
from PIL import Image
import os

st.set_page_config(page_title="DeepVision Dashboard", layout="wide")

# ---------------- Navigation ----------------
st.sidebar.title("Navigation")
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar buttons
if st.sidebar.button("Home"):
    st.session_state.page = "Home"

if st.sidebar.button("EDA"):
    st.session_state.page = "EDA"

page = st.session_state.page


# ---------------- PATH FIX ----------------
# streamlit-sample.py is in: Codes/sampleDashboard
# charts folder is in: Codes/charts
charts_dir = Path(__file__).resolve().parents[1] / "charts"


# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("DeepVision — Explorer")
    st.header("AI DeepVision — Crowd Monitor")

    st.write("""
    AI DeepVision — Crowd Monitor is an experimental project using deep learning for
    real-time crowd density estimation and pattern analysis.
    """)

    st.subheader("Key advantages")
    st.write("""
    - Lightweight and easy to run locally  
    - Supports rapid prototyping  
    - Provides clear visual insights  
    """)

    st.subheader("Typical use-cases")
    st.write("""
    - Public crowd monitoring  
    - Smart city crowd analytics  
    - Dataset exploration before model training  
    """)


# ---------------- EDA PAGE ----------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    if not charts_dir.exists():
        st.error("Charts directory not found. Run the EDA script first.")
    else:
        st.success(f"Loaded charts from: {charts_dir}")

        image_files = sorted([f for f in os.listdir(charts_dir) if f.endswith(".png")])

        # Exclude sample visualizations
        ignore = ["samples_partA.png", "samples_partB.png"]
        image_files = [f for f in image_files if f not in ignore]

        if not image_files:
            st.warning("No charts found inside charts/ folder.")
        else:
            cols = st.columns(2)
            for idx, img_name in enumerate(image_files):
                img_path = charts_dir / img_name
                img = Image.open(img_path)

                with cols[idx % 2]:
                    st.subheader(img_name.replace(".png", "").replace("_", " ").title())
                    st.image(img,  use_container_width=True)
