import os
from pathlib import Path
import io
import base64
import requests
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd


st.set_page_config(page_title="DeepVision Crowd Counting", layout="wide", page_icon="👁️")

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "../results" 
EDA_DIR = RESULTS_ROOT / "eda_results"

MODEL_RESULT_DIRS = {
    "CSRNet": RESULTS_ROOT / "csrnet_cnn",
    "MobileNetCSRNet": RESULTS_ROOT / "mobile_csrnet",
    "RandomForest": RESULTS_ROOT / "random_forest",
    "SimpleCNN": RESULTS_ROOT / "simple_cnn",
}

# SIDEBAR NAVIGATION

st.sidebar.title("DeepVision Controls")
section = st.sidebar.radio("Select Section", ["About", "Data Visualization", "Model Evaluation Results", "Live Demo"])

# Optional: Backend config in sidebar for Live Demo use
backend_url = st.sidebar.text_input("API URL", "http://localhost:8000")

@st.cache_data
def list_files_safe(path: Path):
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir()])

def read_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

def read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def display_file_preview(path: Path):
    suffix = path.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
        img = read_image(path)
        if img:
            st.image(img, caption=path.name, use_container_width=True)
        else:
            st.write(f"Unable to open image: {path.name}")
    elif suffix in [".csv"]:
        df = read_csv(path)
        if df is not None:
            st.dataframe(df)
        else:
            st.write(f"Unable to read CSV: {path.name}")
    elif suffix in [".txt", ".md", ".json", ".log"]:
        txt = read_text(path)
        if txt is not None:
            st.code(txt[:10000])
        else:
            st.write(f"Unable to read file: {path.name}")
    else:
        st.write(f"No preview available for {path.name} (type: {suffix})")

def download_link_bytes(content_bytes: bytes, filename: str, label: str = "Download"):
    b64 = base64.b64encode(content_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# SECTION: ABOUT
if section == "About":
    st.title(" DeepVision Crowd Counting — About")
    st.markdown(
        """
        **DeepVision Crowd Counting** is a small API + frontend for estimating crowd counts in images.
        
        Features:
        - Multiple models supported: **CSRNet**, **MobileNetCSRNet**, **SimpleCNN**, and a **RandomForest** baseline.
        - CSRNet-style models return a density map (H x W) and we generate a heatmap overlay.
        - Frontend supports live demo inference and result visualization.
        
        **How it works**
        1. The Streamlit front-end uploads an image to the FastAPI backend (`/predict/{model}`).
        2. Backend loads the correct model weights from `models/` and returns predicted count, optional density map and heatmap image bytes.
        3. Streamlit displays the count, density map and heatmap.
        
        **Project Layout (expected)**:
        ```
        project_root/
        ├─ backend/
        ├─ models/
        ├─ results/
        │  ├─ eda_results/
        │  ├─ csrnet_cnn/
        │  ├─ mobile_csrnet/
        │  ├─ random_forest/
        │  └─ simple_cnn/
        └─ app.py  (this file)
        ```
        """
    )

    st.subheader("Edit Short Description")
    desc = st.text_area("Project short description", value="DeepVision Crowd Counting Monitor — upload images and estimate crowd size.")
    if st.button("Save description (session only)"):
        st.success("Saved (session-local).")


# SECTION: DATA VISUALIZATION
elif section == "Data Visualization":
    st.title("Data Visualization / EDA Results")
    st.write("Showing contents of:", str(EDA_DIR))

    if not EDA_DIR.exists():
        st.error(f"EDA folder not found: {EDA_DIR}")
    else:
        files = list_files_safe(EDA_DIR)
        if not files:
            st.info("No EDA result files found in the folder.")
        else:
            image_files = [p for p in files if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            non_image_files = [p for p in files if p not in image_files]

            if image_files:
                st.subheader("Image outputs (plots / charts)")
                cols = st.columns(3)
                for i, p in enumerate(image_files):
                    with cols[i % 3]:
                        img = read_image(p)
                        if img:
                            st.image(img, caption=p.name, use_container_width=True)
                        else:
                            st.write(p.name)

            if non_image_files:
                st.subheader("Other outputs (CSV / Reports)")
                for p in non_image_files:
                    with st.expander(p.name):
                        display_file_preview(p)
                        try:
                            data = p.read_bytes()
                            download_link_bytes(data, p.name, "Download file")
                        except Exception:
                            pass


# SECTION: MODEL EVALUATION RESULTS
elif section == "Model Evaluation Results":
    st.title(" Model Evaluation Results")
    st.write("Choose a model to view its evaluation outputs / reports.")
    model_choice = st.selectbox("Select model results folder", list(MODEL_RESULT_DIRS.keys()))

    model_dir = MODEL_RESULT_DIRS.get(model_choice)
    st.write("Results folder:", str(model_dir))

    if not model_dir.exists():
        st.error(f"Results folder not found for {model_choice}: {model_dir}")
    else:
        files = list_files_safe(model_dir)
        if not files:
            st.info("No files in this model results folder.")
        else:
            
            file_types = sorted({p.suffix.lower() or "noext" for p in files})
            chosen_type = st.selectbox("Filter by file extension", ["All"] + file_types)
            filtered = files if chosen_type == "All" else [p for p in files if (p.suffix.lower() or "noext") == chosen_type]

            for p in filtered:
                with st.expander(p.name, expanded=False):
                    display_file_preview(p)
                    try:
                        data = p.read_bytes()
                        download_link_bytes(data, p.name, "Download file")
                    except Exception:
                        pass


# SECTION: LIVE DEMO
elif section == "Live Demo":
    st.title(" Live Demo — Run Model Inference")
    st.write("Upload an image and choose one of the available models to predict crowd count.")

    @st.cache_data(ttl=60)
    def fetch_models(url):
        try:
            r = requests.get(f"{url}/models", timeout=3)
            if r.status_code == 200:
                return r.json().get("models", [])
        except Exception as e:
            st.error(f"Error fetching models: {e}")
            return []

    models = fetch_models(backend_url)
    if not models:
        st.warning("Unable to fetch models from backend. Check API URL/ backend status.")
    selected_model = st.selectbox("Choose a Model", models)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], help="Upload an image to run crowd counting.")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Unable to open image: {e}")

    if uploaded_file and st.button(" Predict Crowd Count"):
        if not selected_model:
            st.error("Please select a model first!")
        else:
            with st.spinner(f"Running {selected_model} inference..."):
                
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    response = requests.post(f"{backend_url}/predict/{selected_model}", files=files, timeout=30)
                except Exception as e:
                    st.error(f" Connection error: {e}")
                    st.stop()

                if response.status_code != 200:
                    st.error(f" Error from backend: {response.text}")
                    st.stop()

                data = response.json()

                count = data.get("predicted_count", 0)
                try:
                    st.success(f"###  Predicted Crowd Count: **{count:.2f}**")
                except Exception:
                    st.success(f"###  Predicted Crowd Count: **{count}**")

                density_map = data.get("density_map", None)
                if density_map:
                    density_map_np = np.array(density_map)
                    st.subheader("Density Map (Predicted Density Distribution)")
                    if density_map_np.ndim == 2:
                        dm = density_map_np - density_map_np.min()
                        if dm.max() > 0:
                            dm = (dm / dm.max()) * 255
                        st.image(dm.astype(np.uint8), clamp=True, channels="L", use_container_width=True)
                    else:
                        st.write("Density map has unexpected shape; showing array:")
                        st.write(density_map_np)

                else:
                    st.info("ℹ This model does not generate a density map.")

                heatmap_hex = data.get("heatmap_image", None)
                if heatmap_hex:
                    try:
                        heatmap_bytes = bytes.fromhex(heatmap_hex)
                        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                        st.subheader(" Heatmap Overlay (Crowd Density Overlay)")
                        st.image(heatmap_img, caption="Heatmap Overlay", use_container_width=True)
                        download_link_bytes(heatmap_bytes, f"heatmap_{uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error decoding heatmap image: {e}")
                else:
                    st.info("ℹ No heatmap available for this model (SimpleCNN / RandomForest).")


# FOOTER
st.sidebar.markdown("---")
st.sidebar.markdown(" DeepVision Crowd Counting")
