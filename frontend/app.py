import os
from pathlib import Path
from typing import List, Optional, Tuple
import io
import base64
import requests
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIG & STYLE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepVision Crowd Monitor",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b; /* Dark slate blue */
        border-right: 1px solid #334155;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    /* Radio Button Customization in Sidebar */
    .stRadio > label {
        color: #e2e8f0 !important;
        font-weight: bold;
    }
    
    /* Cards/Containers */
    .css-1r6slb0 {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PATHS & CONSTANTS
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "../results" 
EDA_DIR = RESULTS_ROOT / "eda_results"

MODEL_RESULT_DIRS = {
    "CSRNet": RESULTS_ROOT / "csrnet_cnn",
    "MobileNetCSRNet": RESULTS_ROOT / "mobile_csrnet",
    "RandomForest": RESULTS_ROOT / "random_forest",
    "SimpleCNN": RESULTS_ROOT / "simple_cnn",
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def list_files_safe(path: Path) -> List[Path]:
    if path is None or not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir()])

def read_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path)
    except Exception:
        return None

def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def download_link_bytes(content_bytes: bytes, filename: str, label: str = "Download"):
    b64 = base64.b64encode(content_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="text-decoration:none; color:#1f77b4; font-weight:bold;">📥 {label}</a>'
    st.markdown(href, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def fetch_models(url: str) -> List[str]:
    try:
        r = requests.get(f"{url}/models", timeout=3)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        return []
    return []

def run_inference(backend_url: str, model_name: str, file_obj, filename: str) -> Tuple[Optional[dict], Optional[str]]:
    files = {"file": (filename, file_obj, "image/jpeg")} # Default to jpeg for simplicity in bytes
    try:
        response = requests.post(f"{backend_url}/predict/{model_name}", files=files, timeout=30)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("👁️ DeepVision")
    st.markdown("### Crowd Monitoring System")
    st.markdown("---")
    
    section = st.radio(
        "Navigation", 
        ["Home & About", "Live Demo", "Compare Models", "Data Visualization", "Model Evaluation"]
    )
    
    st.markdown("---")
    st.markdown("### Settings")
    backend_url = st.text_input("Backend API URL", "http://localhost:8000")
    
    st.markdown("---")
    st.caption(f"© {pd.Timestamp.now().year} DeepVision")

# -----------------------------------------------------------------------------
# SECTION: HOME & ABOUT
# -----------------------------------------------------------------------------
if section == "Home & About":
    st.title("Welcome to DeepVision Crowd Monitor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### AI-Powered Real-Time Crowd Analysis
        
        **DeepVision** is an advanced system designed to estimate crowd density and detect potential overcrowding situations in real-time. 
        Using state-of-the-art Deep Learning models, it provides accurate counts and visual density maps to assist in crowd management.

        #### Key Features:
        *   **Multi-Model Support**: Choose between high-accuracy CSRNet or lightweight SimpleCNN/RandomForest.
        *   **Visual Insights**: Get instant density maps and heatmap overlays.
        *   **Live Inference**: Upload images or usage camera capture (future) to test on the fly.
        *   **Comparative Analysis**: Compare how different models perform on the same distinct scene.
        """)
        
        with st.expander("System Architecture Details"):
            st.code("""
project_root/
├─ backend/ (FastAPI)
├─ frontend/ (Streamlit)
├─ models/ (PyTorch Weights)
└─ results/ (Evaluation Metrics)
            """, language="text")

    with col2:
        st.info("### Quick Start")
        st.markdown("""
        1. Ensure the **Backend** is running.
        2. Go to **Live Demo**.
        3. Upload an image.
        4. See the magic happen! 
        """)
        st.image("https://images.unsplash.com/photo-1533072684617-38e23f05d5e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=80", caption="Crowd Analysis", use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION: LIVE DEMO
# -----------------------------------------------------------------------------
elif section == "Live Demo":
    st.title("🚀 Live Inference")
    
    models = fetch_models(backend_url)
    if not models:
        st.warning("⚠️ Backend not connected. Please check the URL in the sidebar.")
    
    col_input, col_result = st.columns([1, 1.5])
    
    with col_input:
        st.subheader("1. Input")
        model_choice = st.selectbox("Select Model", models if models else ["No models found"])
        
        input_method = st.radio("Input Source", ["Upload Image", "Camera Capture"])
        
        image_bytes = None
        input_name = "input.jpg"
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
                input_name = uploaded_file.name
                st.image(uploaded_file, caption="Preview", use_container_width=True)
                
        elif input_method == "Camera Capture":
            camera_file = st.camera_input("Take a picture")
            if camera_file:
                image_bytes = camera_file.getvalue()
                st.success("Image captured!")

    with col_result:
        st.subheader("2. Result")
        if image_bytes and st.button("Analyze Image", type="primary"):
            if not models:
                st.error("No models available.")
            else:
                with st.spinner(f"Running inference with {model_choice}..."):
                    # Create a file-like object
                    file_obj = io.BytesIO(image_bytes)
                    data, error = run_inference(backend_url, model_choice, file_obj, input_name)
                    
                    if error:
                        st.error(error)
                    else:
                        # Display Metrics
                        count = data.get("predicted_count", 0)
                        st.metric("Pred. Crowd Count", f"{count:.2f}")
                        
                        # Tabs for visuals
                        tab1, tab2, tab3 = st.tabs(["Heatmap Overlay", "Density Map", "Raw Data"])
                        
                        with tab1:
                            heatmap_hex = data.get("heatmap_image")
                            if heatmap_hex:
                                try:
                                    hm_bytes = bytes.fromhex(heatmap_hex)
                                    st.image(io.BytesIO(hm_bytes), use_container_width=True)
                                    download_link_bytes(hm_bytes, f"heatmap_{input_name}", "Download Heatmap")
                                except Exception:
                                    st.error("Error displaying heatmap.")
                            else:
                                st.info("No heatmap available for this model.")
                                
                        with tab2:
                            dmap = data.get("density_map")
                            if dmap:
                                dmap_np = np.array(dmap)
                                if dmap_np.ndim == 2:
                                    # Normalize for display
                                    dm_norm = (dmap_np - dmap_np.min())
                                    if dm_norm.max() > 0:
                                        dm_norm = (dm_norm / (dm_norm.max() + 1e-6)) * 255
                                    st.image(dm_norm.astype(np.uint8), clamp=True, use_container_width=True)
                                else:
                                    st.write("Invalid density map shape.")
                            else:
                                st.info("No density map returned.")
                                
                        with tab3:
                            st.json(data)

# -----------------------------------------------------------------------------
# SECTION: COMPARE MODELS
# -----------------------------------------------------------------------------
elif section == "Compare Models":
    st.title("⚖️ Model Comparison")
    
    models = fetch_models(backend_url)
    if len(models) < 2:
        st.warning("Need at least 2 models to compare. Check backend.")
    
    c1, c2 = st.columns(2)
    with c1:
        model_a = st.selectbox("Model A", models, index=0 if models else 0)
    with c2:
        # Try to select a different second model
        default_idx = 1 if len(models) > 1 else 0
        model_b = st.selectbox("Model B", models, index=default_idx)
        
    uploaded_compare = st.file_uploader("Upload Image for Comparison", type=["jpg", "png", "jpeg"])
    
    if uploaded_compare and st.button("Compare Results"):
        img_bytes = uploaded_compare.getvalue()
        
        col_a, col_b = st.columns(2)
        
        # Model A Inference
        with col_a:
            st.subheader(f"Model A: {model_a}")
            with st.spinner("Processing A..."):
                res_a, err_a = run_inference(backend_url, model_a, io.BytesIO(img_bytes), uploaded_compare.name)
            
            if err_a:
                st.error(err_a)
            else:
                st.metric("Count", f"{res_a.get('predicted_count', 0):.2f}")
                hm_hex = res_a.get("heatmap_image")
                if hm_hex:
                    st.image(io.BytesIO(bytes.fromhex(hm_hex)), caption="Overlay", use_container_width=True)
                else:
                    st.write("No heatmap")

        # Model B Inference
        with col_b:
            st.subheader(f"Model B: {model_b}")
            with st.spinner("Processing B..."):
                res_b, err_b = run_inference(backend_url, model_b, io.BytesIO(img_bytes), uploaded_compare.name)
            
            if err_b:
                st.error(err_b)
            else:
                st.metric("Count", f"{res_b.get('predicted_count', 0):.2f}")
                hm_hex = res_b.get("heatmap_image")
                if hm_hex:
                    st.image(io.BytesIO(bytes.fromhex(hm_hex)), caption="Overlay", use_container_width=True)
                else:
                    st.write("No heatmap")

# -----------------------------------------------------------------------------
# SECTION: DATA VISUALIZATION
# -----------------------------------------------------------------------------
elif section == "Data Visualization":
    st.title("📊 Data Visualization")
    
    if not EDA_DIR.exists():
        st.error(f"EDA folder not found: {EDA_DIR}")
    else:
        files = list_files_safe(EDA_DIR)
        
        # Separation
        imgs = [p for p in files if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        reports = [p for p in files if p not in imgs]
        
        if imgs:
            st.subheader("Visual Plots")
            # Grid Layout
            cols = st.columns(3)
            for idx, p in enumerate(imgs):
                with cols[idx % 3]:
                    with st.container(): # mimicking a card
                       st.image(str(p), use_container_width=True)
                       st.caption(p.name)
                       
        if reports:
            st.divider()
            st.subheader("Reports & Logs")
            for p in reports:
                with st.expander(f"📄 {p.name}"):
                    # Preview logic
                    if p.suffix == ".csv":
                        st.dataframe(read_csv(p))
                    elif p.suffix in [".txt", ".json", ".md"]:
                        st.code(read_text(p))
                    else:
                        st.write("No preview.")
                        
                    st.download_button(
                        label="Download File",
                        data=p.read_bytes(),
                        file_name=p.name,
                    )

# -----------------------------------------------------------------------------
# SECTION: MODEL EVALUATION
# -----------------------------------------------------------------------------
elif section == "Model Evaluation":
    st.title("📈 Model Evaluation")
    
    selected_model_type = st.selectbox("Select Model Architecture", list(MODEL_RESULT_DIRS.keys()))
    target_dir = MODEL_RESULT_DIRS[selected_model_type]
    
    if not target_dir.exists():
        st.info(f"No results folder found for {selected_model_type}.")
    else:
        files = list_files_safe(target_dir)
        if not files:
            st.info("No result files yet.")
        else:
            for p in files:
                with st.expander(p.name):
                    if p.suffix == ".png":
                        st.image(str(p))
                    elif p.suffix == ".csv":
                        st.dataframe(pd.read_csv(p))
                    else:
                        st.write(p)
