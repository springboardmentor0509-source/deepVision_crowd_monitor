# app.py (updated: filters fixed + descriptive naming)
import os
import io
import zipfile
import base64
from pathlib import Path
from typing import List

import requests
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DeepVision Crowd Counting", layout="wide", page_icon="👁️")

# ---------------- PATHS / CONSTANTS ----------------
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = (ROOT / "../results").resolve()
EDA_DIR = RESULTS_ROOT / "eda_results"

MODEL_RESULT_DIRS = {
    "CSRNet": RESULTS_ROOT / "csrnet_cnn",
    "MobileNetCSRNet": RESULTS_ROOT / "mobile_csrnet",
    "RandomForest": RESULTS_ROOT / "random_forest",
    "SimpleCNN": RESULTS_ROOT / "simple_cnn",
}

# ---------------- SIDEBAR ----------------
st.sidebar.title("DeepVision Controls")
st.sidebar.caption("Crowd Counting Monitor")

section = st.sidebar.radio("Select Section", ["About", "Data Visualization", "Model Evaluation Results", "Live Demo"])

st.sidebar.markdown("---")
st.sidebar.markdown("Backend Settings")
backend_url = st.sidebar.text_input("API URL", "http://localhost:8000", help="Base URL of the FastAPI backend used for predictions.")
st.sidebar.markdown("---")
st.sidebar.markdown("DeepVision Crowd Counting")

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def list_files_safe(path: Path) -> List[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_file()])

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

def pretty_name_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").title()

def download_link_bytes(content_bytes: bytes, filename: str, label: str = "Download"):
    b64 = base64.b64encode(content_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def folder_summary(path: Path):
    files = list_files_safe(path)
    total_size = sum(p.stat().st_size for p in files) if files else 0
    size_mb = total_size / (1024 ** 2)
    return len(files), size_mb

@st.cache_data(ttl=60, show_spinner=False)
def fetch_models(url: str):
    try:
        r = requests.get(f"{url.rstrip('/')}/models", timeout=3)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        return []
    return []

# ---------------- STYLES ----------------
st.markdown(
    """
    <style>
    .dv-card { border-radius: 10px; padding: 8px; background-color: #0f1720; box-shadow: 0 6px 18px rgba(0,0,0,0.35); margin-bottom: 14px; }
    .dv-figure-title { font-weight:600; color: #f3f4f6; margin-top:6px; margin-bottom:2px; }
    .dv-subtle { color: #9ca3af; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- AUTO-TITLE ----------------
def is_plot_name(name: str) -> bool:
    n = name.lower()
    keys = ["plot", "figure", "hist", "loss", "metric", "comparison", "gt_vs", "gt-vs", "residual", "scatter", "correlation"]
    return any(k in n for k in keys)

def auto_title(path: Path, idx: int) -> str:
    stem = path.stem.lower()
    # plot-specific heuristics
    if any(x in stem for x in ["hist", "distribution"]):
        base = "Count Distribution"
        kind = "Plot"
    elif any(x in stem for x in ["gt_vs", "gt-vs", "gtvs", "gt_vs_pred"]):
        base = "Ground Truth vs Prediction"
        kind = "Plot"
    elif any(x in stem for x in ["residual", "error", "resid"]):
        base = "Residual / Error Analysis"
        kind = "Plot"
    elif any(x in stem for x in ["loss", "training", "train_loss"]):
        base = "Training Loss Curve"
        kind = "Plot"
    elif any(x in stem for x in ["scatter", "brightness", "correlation", "entropy"]):
        base = "Scatter / Correlation Plot"
        kind = "Plot"
    elif any(x in stem for x in ["sample", "figure", "image", "img", "example"]):
        base = "Sample Crowd Image"
        kind = "Crowd Image"
    else:
        # fallback: if it's an image but filename not matched, label as crowd image
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            base = pretty_name_from_path(path)
            kind = "Crowd Image" if not is_plot_name(path.name) else "Plot"
        elif path.suffix.lower() == ".csv":
            base = pretty_name_from_path(path)
            kind = "CSV Report"
        else:
            base = pretty_name_from_path(path)
            kind = "File"
    return f"{kind} — {base}"

def display_file_preview(path: Path):
    s = path.suffix.lower()
    if s in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]:
        img = read_image(path)
        if img:
            st.image(img, caption=f"{pretty_name_from_path(path)} — {path.name}", use_container_width=True)
        else:
            st.write(f"Unable to open image: {path.name}")
    elif s == ".csv":
        df = read_csv(path)
        if df is not None:
            st.dataframe(df, use_container_width=True)
        else:
            st.write(f"Unable to read CSV: {path.name}")
    elif s in [".txt", ".md", ".json", ".log"]:
        txt = read_text(path)
        if txt is not None:
            st.code(txt[:10000])
        else:
            st.write(f"Unable to read file: {path.name}")
    else:
        st.write(f"No preview available for {path.name} (type: {s})")

# ---------------- ABOUT ----------------
if section == "About":
    st.markdown(
        """
        <div style="padding:28px;border-radius:14px;background:linear-gradient(135deg,#0b2a4a,#12213d);color:white">
          <h1 style="margin:0 0 6px 0">👁️ DeepVision — Intelligent Crowd Analytics Platform</h1>
          <p style="margin:0;font-size:15px;opacity:0.95">
            Research-grade system for crowd counting, density estimation, visual analytics and real-time safety assessment.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("📘 Project Overview")
    st.markdown(
        """
        **DeepVision** is an end-to-end platform for:
        - Accurate crowd counting using deep learning and classical baselines  
        - Density map generation and heatmap overlays for spatial analysis  
        - Visual analytics and model evaluation tooling for research & deployment
        """
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Supported Models", "4", "+CSRNet +MobileNetCSRNet +SimpleCNN +RF")
    col2.metric("Dataset Scale", "1,198 images", "~330k annotated heads")
    col3.metric("Outputs", "Count + Density + Heatmap", "Real-time inference")

    st.markdown("---")
    st.header("📊 Dataset — ShanghaiTech (used)")
    st.markdown(
        """
        **Part A (dense)** — 482 images (300 train / 182 test). Avg ≈ 500 people/image.  
        **Part B (sparse)** — 716 images (400 train / 316 test). Avg ≈ 120 people/image.  
        Ground truth is stored as (x, y) head coordinates in MATLAB `.mat` files and used to produce geometry-adaptive density maps.
        """
    )

    st.markdown("---")
    st.header("🔄 Complete Pipeline (high level)")
    st.markdown(
        """
        **1. Preprocessing:** parse `.mat`, build density maps (kNN-based sigma), resize and create metadata CSV.  
        **2. Training:** train CNN models (CSRNet / MobileNet variant / SimpleCNN) and a RandomForest baseline.  
        **3. Evaluation:** MAE/RMSE, prediction vs GT plots, error histograms.  
        **4. Deployment:** FastAPI backend with `POST /predict/{model_name}` returning count + density + heatmap.
        """
    )

    st.markdown("---")
    st.header("🧠 Technical Implementation (short)")
    st.markdown(
        """
        - Loss: MSE. Optimizer: Adam. Useful metrics: MAE, RMSE.  
        - Density maps computed by summing adaptive Gaussians per annotated head.  
        - Heatmaps returned as hex-encoded PNG bytes in the API response for frontend display.
        """
    )

    st.markdown("---")
    st.header("🚀 Quick Start")
    st.markdown(
        """
        ```bash
        pip install -r requirements.txt
        python preprocess_data.py
        uvicorn backend:app --reload   # start API
        streamlit run app.py           # start dashboard
        ```
        """
    )

    st.markdown("---")
    st.subheader("💡 Use cases")
    st.write("""
     - Public safety & event monitoring  
     - Smart city planning & transportation analytics  
     - Retail footfall & queue analysis
    """)

# ---------------- DATA VISUALIZATION ----------------
elif section == "Data Visualization":
    st.title("Data Visualization / EDA Results")
    st.write("Explore plots, charts and images produced during preprocessing, training and evaluation.")
    st.write("EDA folder:", f"`{EDA_DIR}`")

    if not EDA_DIR.exists():
        st.error(f"EDA folder not found: {EDA_DIR}")
    else:
        file_count, total_mb = folder_summary(EDA_DIR)
        st.caption(f"Found **{file_count}** files · total size ~ **{total_mb:.2f} MB**")

        files = list_files_safe(EDA_DIR)
        if not files:
            st.info("No EDA files available.")
        else:
            # categorize
            image_files = [p for p in files if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
            csv_files = [p for p in files if p.suffix.lower() == ".csv"]
            other_files = [p for p in files if p not in image_files + csv_files]

            # FILTER UI: persistent selectbox (reliable)
            filter_choice = st.selectbox("Quick filter", ["All", "Plots / Charts", "Sample Images", "CSV Reports"])
            query = st.text_input("Search filenames (contains)...", value="")

            # apply selection
            if filter_choice == "All":
                filtered_images = image_files
                filtered_csvs = csv_files
                filtered_others = other_files
            elif filter_choice == "Plots / Charts":
                filtered_images = [p for p in image_files if is_plot_name(p.name)]
                filtered_csvs = []
                filtered_others = []
            elif filter_choice == "Sample Images":
                filtered_images = [p for p in image_files if not is_plot_name(p.name)]
                filtered_csvs = []
                filtered_others = []
            elif filter_choice == "CSV Reports":
                filtered_images = []
                filtered_csvs = csv_files
                filtered_others = other_files
            else:
                filtered_images = image_files
                filtered_csvs = csv_files
                filtered_others = other_files

            # apply search query across all categories shown
            if query:
                filtered_images = [p for p in filtered_images if query.lower() in p.name.lower()]
                filtered_csvs = [p for p in filtered_csvs if query.lower() in p.name.lower()]
                filtered_others = [p for p in filtered_others if query.lower() in p.name.lower()]

            # GALLERY
            st.markdown("---")
            st.subheader("Image Outputs (Plots / Charts)")
            tab_gallery, tab_detail = st.tabs(["Gallery View", "Detail View"])

            with tab_gallery:
                cols = st.columns(3, gap="large")
                for i, p in enumerate(filtered_images, start=1):
                    c = cols[(i - 1) % 3]
                    with c:
                        st.markdown('<div class="dv-card">', unsafe_allow_html=True)
                        img = read_image(p)
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.write("Unable to load image.")
                        st.markdown(f'<div class="dv-figure-title">{auto_title(p, i)}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="dv-subtle">{p.name}</div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

            with tab_detail:
                for i, p in enumerate(filtered_images, start=1):
                    title = auto_title(p, i)
                    with st.expander(f"{title} — {p.name}", expanded=False):
                        display_file_preview(p)
                        try:
                            size_kb = p.stat().st_size / 1024.0
                            st.write(f"**Size:** {size_kb:.1f} KB")
                        except Exception:
                            st.write("**Size:** N/A")
                        try:
                            img = Image.open(p)
                            st.write(f"**Resolution:** {img.width} × {img.height}")
                        except Exception:
                            st.write("**Resolution:** N/A")
                        try:
                            download_link_bytes(p.read_bytes(), p.name, "📥 Download Image")
                        except Exception:
                            pass
                        # related CSVs
                        related = [c for c in csv_files if c.stem == p.stem]
                        if related:
                            st.write("**Related CSV(s):**")
                            for r in related:
                                with st.expander(r.name):
                                    display_file_preview(r)
                                    try:
                                        download_link_bytes(r.read_bytes(), r.name, "📥 Download CSV")
                                    except Exception:
                                        pass

            # CSVs
            if filtered_csvs:
                st.markdown("---")
                st.subheader("CSV Reports")
                for c in filtered_csvs:
                    with st.expander(f"CSV Report — {c.name}", expanded=False):
                        display_file_preview(c)
                        try:
                            download_link_bytes(c.read_bytes(), c.name, "📥 Download CSV")
                        except Exception:
                            pass

            # Others
            if filtered_others:
                st.markdown("---")
                st.subheader("Other Files")
                for o in filtered_others:
                    with st.expander(f"File — {o.name}", expanded=False):
                        display_file_preview(o)
                        try:
                            download_link_bytes(o.read_bytes(), o.name, "📥 Download")
                        except Exception:
                            pass

            # Export ZIP
            st.markdown("---")
            st.subheader("Export")
            all_shown = filtered_images + filtered_csvs + filtered_others
            if all_shown:
                names = [p.name for p in all_shown]
                selected = st.multiselect("Select files for ZIP (leave empty to include all shown):", names)
                if st.button("📦 Download ZIP of selection"):
                    targets = [p for p in all_shown if (p.name in selected)] if selected else all_shown
                    if not targets:
                        st.warning("No files selected.")
                    else:
                        buf = io.BytesIO()
                        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                            for t in targets:
                                try:
                                    zf.writestr(t.name, t.read_bytes())
                                except Exception:
                                    pass
                        buf.seek(0)
                        download_link_bytes(buf.read(), "eda_selection.zip", "⬇️ Download ZIP")
            else:
                st.info("No files shown with current filter/search combination.")

# ---------------- MODEL EVAL ----------------
elif section == "Model Evaluation Results":
    st.markdown(
        """
        <div style="padding:28px;border-radius:14px;background:linear-gradient(135deg,#0b2a4a,#12213d);color:white">
          <h1 style="margin:0 0 6px 0">Model Evaluation Results</h1>
          <p style="margin:0;font-size:15px;opacity:0.95">
           Choose a model to view its evaluation outputs / reports.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    model_choice = st.selectbox("Select model results folder", list(MODEL_RESULT_DIRS.keys()))
    model_dir = MODEL_RESULT_DIRS.get(model_choice)
    st.write("Results folder:", f"`{model_dir}`")

    if not model_dir.exists():
        st.error(f"Results folder not found for {model_choice}: {model_dir}")
    else:
        files = list_files_safe(model_dir)
        if not files:
            st.info("No files in this model results folder.")
        else:
            count = len(files)
            total_mb = sum(p.stat().st_size for p in files) / (1024 ** 2)
            st.caption(f"Found {count} files · total size ~ {total_mb:.2f} MB")

            metrics_candidates = [p for p in files if "metrics" in p.name.lower() and p.suffix.lower() == ".csv"]
            if metrics_candidates:
                st.subheader("Quick Metrics Overview")
                dfm = read_csv(metrics_candidates[0])
                if dfm is not None:
                    st.dataframe(dfm.head(), use_container_width=True)
                    numerics = dfm.select_dtypes("number").columns
                    if len(numerics) > 0:
                        st.bar_chart(dfm[numerics])

            file_types = sorted({p.suffix.lower() or "noext" for p in files})
            chosen = st.selectbox("Filter by file extension", ["All"] + file_types)
            if chosen == "All":
                filtered = files
            else:
                filtered = [p for p in files if (p.suffix.lower() or "noext") == chosen]

            for p in filtered:
                with st.expander(p.name, expanded=False):
                    display_file_preview(p)
                    try:
                        download_link_bytes(p.read_bytes(), p.name, "📥 Download")
                    except Exception:
                        pass

# ---------------- LIVE DEMO ----------------
elif section == "Live Demo":
    st.markdown(
        """
        <div style="padding:28px;border-radius:14px;background:linear-gradient(135deg,#0b2a4a,#12213d);color:white">
          <h1 style="margin:0 0 6px 0">Live Demo</h1>
          <p style="margin:0;font-size:15px;opacity:0.95">
         Upload images and select a model to run inference.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = fetch_models(backend_url)
    if not models:
        st.warning("Unable to fetch models from backend. Check backend or enter API URL correctly.")
    selected_model = st.selectbox("Choose a Model", models if models else list(MODEL_RESULT_DIRS.keys()))
    decimals = st.slider("Decimal places", min_value=0, max_value=3, value=2)

    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        st.subheader("Uploaded Images")
        cols = st.columns(3)
        for i, uf in enumerate(uploaded_files):
            try:
                img = Image.open(uf)
                with cols[i % 3]:
                    st.image(img, caption=uf.name, use_container_width=True)
            except Exception as e:
                st.error(f"Unable to open image {uf.name}: {e}")

    if "inference_history" not in st.session_state:
        st.session_state.inference_history = []

    if uploaded_files and st.button("🔍 Predict Crowd Count"):
        if not selected_model:
            st.error("Please select a model first!")
        else:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Running {selected_model} on {uploaded_file.name}..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    try:
                        resp = requests.post(f"{backend_url.rstrip('/')}/predict/{selected_model}", files=files, timeout=30)
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception as e:
                        st.error(f"Error calling backend for {uploaded_file.name}: {e}")
                        continue

                    count = data.get("predicted_count", None)
                    st.markdown("---")
                    st.subheader(f"Results — {uploaded_file.name}")
                    if count is not None:
                        try:
                            st.success(f"Predicted Crowd Count: **{float(count):.{decimals}f}**")
                        except Exception:
                            st.success(f"Predicted Crowd Count: **{count}**")
                    else:
                        st.warning("No predicted_count returned by backend.")

                    # show original + heatmap side-by-side
                    heatmap_hex = data.get("heatmap_image", None)
                    col1, col2 = st.columns(2)
                    with col1:
                        try:
                            uploaded_file.seek(0)
                            st.image(Image.open(uploaded_file), caption="Original Image", use_container_width=True)
                        except Exception:
                            st.write("Original preview unavailable.")

                    with col2:
                        if heatmap_hex:
                            try:
                                heat_bytes = bytes.fromhex(heatmap_hex)
                                st.image(Image.open(io.BytesIO(heat_bytes)), caption="Heatmap Overlay", use_container_width=True)
                                download_link_bytes(heat_bytes, f"heatmap_{uploaded_file.name}", "📥 Download heatmap")
                            except Exception as e:
                                st.error(f"Error decoding heatmap: {e}")
                        else:
                            st.info("No heatmap available for this model/image.")

                    # density map if present
                    density_map = data.get("density_map", None)
                    if density_map:
                        dm_np = np.array(density_map)
                        st.subheader("Density Map (visual preview)")
                        if dm_np.ndim == 2:
                            dm = dm_np - dm_np.min()
                            if dm.max() > 0:
                                dm = (dm / dm.max()) * 255
                            st.image(dm.astype("uint8"), clamp=True, channels="L", use_container_width=True)
                        else:
                            st.write("Unexpected density_map shape - showing array")
                            st.write(dm_np)

                    # record history
                    st.session_state.inference_history.append({
                        "image": uploaded_file.name,
                        "model": selected_model,
                        "predicted": count
                    })

    if st.session_state.get("inference_history"):
        st.markdown("---")
        st.subheader("Inference History (this session)")
        df_hist = pd.DataFrame(st.session_state["inference_history"])
        st.dataframe(df_hist, use_container_width=True)
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download history CSV", csv_bytes, "inference_history.csv", mime="text/csv")
