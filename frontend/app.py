import os
from pathlib import Path
import io
import base64
import requests
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd


# --------------------------------------------------------------------
# BASIC PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="DeepVision Crowd Counting",
    layout="wide",
    page_icon="👁️",
)

# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = (ROOT / "../results").resolve()
EDA_DIR = RESULTS_ROOT / "eda_results"

MODEL_RESULT_DIRS = {
    "CSRNet": RESULTS_ROOT / "csrnet_cnn",
    "MobileNetCSRNet": RESULTS_ROOT / "mobile_csrnet",
    "RandomForest": RESULTS_ROOT / "random_forest",
    "SimpleCNN": RESULTS_ROOT / "simple_cnn",
}

# --------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------
st.sidebar.title("DeepVision Controls")
st.sidebar.caption("Crowd Counting Monitor")

section = st.sidebar.radio(
    "Select Section",
    ["About", "Data Visualization", "Model Evaluation Results", "Live Demo"],
)

# Backend config for Live Demo use
st.sidebar.markdown("### Backend Settings")
backend_url = st.sidebar.text_input(
    "API URL",
    "http://localhost:8000",
    help="Base URL of the FastAPI backend used for predictions.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("DeepVision Crowd Counting")


# --------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_files_safe(path: Path):
    """Safely list files in a directory (files only)."""
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
    """Convert filename into a nicer human-readable title."""
    return path.stem.replace("_", " ").title()


def display_file_preview(path: Path):
    """Preview a file depending on its extension (image / csv / text / other)."""
    suffix = path.suffix.lower()

    if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
        img = read_image(path)
        if img:
            st.image(img, caption=pretty_name_from_path(path), use_container_width=True)
        else:
            st.write(f"Unable to open image: {path.name}")

    elif suffix == ".csv":
        df = read_csv(path)
        if df is not None:
            st.dataframe(df, use_container_width=True)
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
    """Render a clickable download link for bytes."""
    b64 = base64.b64encode(content_bytes).decode()
    href = (
        f'<a href="data:application/octet-stream;base64,{b64}" '
        f'download="{filename}">{label}</a>'
    )
    st.markdown(href, unsafe_allow_html=True)


def folder_summary(path: Path):
    """Show a short summary of how many files and total size."""
    files = list_files_safe(path)
    total_size = sum(p.stat().st_size for p in files) if files else 0
    size_mb = total_size / (1024 ** 2)
    st.caption(
        f"Found **{len(files)}** files · total size ~ **{size_mb:.2f} MB**"
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_models(url: str):
    """Fetch available models from backend /models endpoint."""
    try:
        r = requests.get(f"{url}/models", timeout=3)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        # Do not show error here – handled in Live Demo section
        return []
    return []


# --------------------------------------------------------------------
# SECTION: ABOUT
# --------------------------------------------------------------------
if section == "About":
    st.title("DeepVision Crowd Counting — Dashboard Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("What is DeepVision Crowd Counting?")
        st.markdown(
            """
            **DeepVision Crowd Counting** is an end-to-end system that combines:
            
            - A **FastAPI backend** for model inference  
            - A **Streamlit frontend** for visualization and experimentation  

            The goal is to estimate **people count in crowd images** and visualize
            the density distribution using deep learning–based models.
            """
        )

        st.subheader("How the Pipeline Works")
        st.markdown(
            """
            1. User uploads a crowd image through this web app.  
            2. The frontend sends the image to the FastAPI backend endpoint:  
               `POST /predict/{model_name}`.  
            3. The backend loads the selected model from `models/` and runs inference.  
            4. The backend responds with:
               - Predicted **crowd count**  
               - Optional **density map** (H × W)  
               - Optional **heatmap overlay** image bytes  
            5. This app displays the numerical result and visual outputs.
            """
        )

    with col2:
        st.subheader("Available Models")
        model_df = pd.DataFrame(
            {
                "Model": list(MODEL_RESULT_DIRS.keys()),
                "Type": ["CNN (CSRNet)", "Mobile CNN", "Tree-based", "Simple CNN"],
                "Outputs": [
                    "Count + Density + Heatmap",
                    "Count + Density + Heatmap",
                    "Count only",
                    "Count only",
                ],
            }
        )
        st.table(model_df)

        st.subheader("Project Layout")
        st.code(
            """
project_root/
├─ backend/
├─ models/
├─ results/
│  ├─ eda_results/
│  ├─ csrnet_cnn/
│  ├─ mobile_csrnet/
│  ├─ random_forest/
│  └─ simple_cnn/
└─ app.py   # this Streamlit app
            """,
            language="text",
        )

    st.markdown("---")
    with st.expander("Quick Start: How to Use This App", expanded=False):
        st.markdown(
            """
            1. **Data Visualization** – explore example images and EDA plots.  
            2. **Model Evaluation Results** – inspect metrics, training plots, and reports
               for each model.  
            3. **Live Demo** – upload your own image(s), select a model, and view
               the predicted crowd count and visual outputs.  
            """
        )

    st.subheader("Experiment Notes (session only)")
    default_notes = (
        "Use this space to write down observations while testing models, "
        "for example: which model works better on dense crowds, etc."
    )
    notes = st.text_area(
        "Notes",
        value=st.session_state.get("experiment_notes", default_notes),
        height=120,
    )
    st.session_state["experiment_notes"] = notes
    st.info("These notes are stored only for this session and are not saved to disk.")


# --------------------------------------------------------------------
# SECTION: DATA VISUALIZATION
# --------------------------------------------------------------------
elif section == "Data Visualization":
    st.title("Data Visualization / EDA Results")
    st.write("EDA folder:", f"`{EDA_DIR}`")

    if not EDA_DIR.exists():
        st.error(f"EDA folder not found: {EDA_DIR}")
    else:
        files = list_files_safe(EDA_DIR)
        if not files:
            st.info("No EDA result files found in the folder.")
        else:
            folder_summary(EDA_DIR)

            # Separate images and others
            image_files = [p for p in files if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            non_image_files = [p for p in files if p not in image_files]

            # Filter / search controls
            with st.expander("Filter & Search", expanded=False):
                name_filter = st.text_input(
                    "Filter by file name (contains)...",
                    "",
                    help="Type part of a file name to filter images/reports.",
                )
                only_images = st.checkbox("Show only image files", value=False)

            if name_filter:
                image_files = [p for p in image_files if name_filter.lower() in p.name.lower()]
                non_image_files = [p for p in non_image_files if name_filter.lower() in p.name.lower()]

            if only_images:
                non_image_files = []

            # IMAGE GALLERY
            if image_files:
                st.subheader("Image Outputs (Plots / Charts)")

                tab_gallery, tab_details = st.tabs(["Gallery View", "Detail View"])

                with tab_gallery:
                    cols = st.columns(3)
                    for i, p in enumerate(image_files):
                        img = read_image(p)
                        if img:
                            with cols[i % 3]:
                                st.image(
                                    img,
                                    caption=pretty_name_from_path(p),
                                    use_container_width=True,
                                )
                        else:
                            st.write(p.name)

                with tab_details:
                    for p in image_files:
                        with st.expander(pretty_name_from_path(p)):
                            display_file_preview(p)
                            try:
                                data = p.read_bytes()
                                download_link_bytes(data, p.name, "Download image")
                            except Exception:
                                pass

            # NON IMAGE FILES (CSV / reports)
            if non_image_files:
                st.subheader("Other Outputs (CSV / Reports)")
                for p in non_image_files:
                    with st.expander(p.name):
                        display_file_preview(p)
                        try:
                            data = p.read_bytes()
                            download_link_bytes(data, p.name, "Download file")
                        except Exception:
                            pass


# --------------------------------------------------------------------
# SECTION: MODEL EVALUATION RESULTS
# --------------------------------------------------------------------
elif section == "Model Evaluation Results":
    st.title("Model Evaluation Results")
    st.write("Choose a model to view its evaluation outputs and reports.")

    model_choice = st.selectbox(
        "Select model results folder",
        list(MODEL_RESULT_DIRS.keys()),
    )

    model_dir = MODEL_RESULT_DIRS.get(model_choice)
    st.write("Results folder:", f"`{model_dir}`")

    if not model_dir.exists():
        st.error(f"Results folder not found for {model_choice}: {model_dir}")
    else:
        files = list_files_safe(model_dir)
        if not files:
            st.info("No files in this model results folder.")
        else:
            folder_summary(model_dir)

            # Quick metrics overview if a metrics CSV is present
            metrics_candidates = [p for p in files if "metrics" in p.name and p.suffix.lower() == ".csv"]
            if metrics_candidates:
                st.subheader("Quick Metrics Overview")
                metrics_path = metrics_candidates[0]
                df_metrics = read_csv(metrics_path)
                if df_metrics is not None:
                    st.dataframe(df_metrics.head(), use_container_width=True)
                    numeric_cols = df_metrics.select_dtypes("number").columns
                    if len(numeric_cols) > 0:
                        st.bar_chart(df_metrics[numeric_cols])
                st.caption(f"Source: {metrics_path.name}")

            # Filter by file type (extension)
            file_types = sorted({p.suffix.lower() or "noext" for p in files})
            chosen_type = st.selectbox(
                "Filter by file extension",
                ["All"] + file_types,
            )
            if chosen_type == "All":
                filtered = files
            else:
                filtered = [
                    p for p in files if (p.suffix.lower() or "noext") == chosen_type
                ]

            for p in filtered:
                with st.expander(p.name, expanded=False):
                    display_file_preview(p)
                    try:
                        data = p.read_bytes()
                        download_link_bytes(data, p.name, "Download file")
                    except Exception:
                        pass


# --------------------------------------------------------------------
# SECTION: LIVE DEMO
# --------------------------------------------------------------------
elif section == "Live Demo":
    st.title("Live Demo — Run Model Inference")
    st.write(
        "Upload one or more crowd images and choose a model to predict the crowd count. "
        "For CSRNet-style models, density maps and heatmaps are also displayed."
    )

    models = fetch_models(backend_url)
    if not models:
        st.warning(
            "Unable to fetch models from backend. "
            "Check that the FastAPI server is running and exposes a /models endpoint."
        )

    col_model, col_precision = st.columns([2, 1])

    with col_model:
        selected_model = st.selectbox("Choose a Model", models)

    with col_precision:
        decimals = st.slider(
            "Decimal places",
            min_value=0,
            max_value=3,
            value=2,
            help="How many decimal places to show for predicted count.",
        )

    # Multiple image upload
    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can upload one or more images for batch prediction.",
    )

    # Show thumbnails
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

    # Simple history (per session)
    if "inference_history" not in st.session_state:
        st.session_state["inference_history"] = []

    # Predict button
    if uploaded_files and st.button("🔍 Predict Crowd Count"):
        if not selected_model:
            st.error("Please select a model first!")
        else:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Running {selected_model} inference for {uploaded_file.name}..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                    try:
                        response = requests.post(
                            f"{backend_url}/predict/{selected_model}",
                            files=files,
                            timeout=30,
                        )
                    except Exception as e:
                        st.error(f"Connection error for {uploaded_file.name}: {e}")
                        continue

                    if response.status_code != 200:
                        st.error(f"Error from backend for {uploaded_file.name}: {response.text}")
                        continue

                    data = response.json()
                    count = data.get("predicted_count", 0)

                    # Store in history
                    try:
                        numeric_count = float(count)
                    except Exception:
                        numeric_count = count
                    st.session_state["inference_history"].append(
                        {
                            "image_name": uploaded_file.name,
                            "model": selected_model,
                            "predicted_count": numeric_count,
                        }
                    )

                    # Display results for this image
                    st.markdown("---")
                    st.subheader(f"Results for **{uploaded_file.name}**")

                    try:
                        st.success(
                            f"Predicted Crowd Count: **{float(count):.{decimals}f}**"
                        )
                    except Exception:
                        st.success(f"Predicted Crowd Count: **{count}**")

                    # Density map
                    density_map = data.get("density_map", None)
                    heatmap_hex = data.get("heatmap_image", None)

                    # Layout for visual outputs
                    col_orig, col_heat = st.columns(2)

                    # Original image
                    with col_orig:
                        try:
                            uploaded_file.seek(0)
                            orig_img = Image.open(uploaded_file)
                            st.image(orig_img, caption="Original Image", use_container_width=True)
                        except Exception:
                            st.write("Original image preview unavailable.")

                    # Heatmap overlay
                    if heatmap_hex:
                        try:
                            heatmap_bytes = bytes.fromhex(heatmap_hex)
                            heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                            with col_heat:
                                st.image(
                                    heatmap_img,
                                    caption="Heatmap Overlay",
                                    use_container_width=True,
                                )
                            download_link_bytes(
                                heatmap_bytes,
                                f"heatmap_{uploaded_file.name}",
                                "Download heatmap",
                            )
                        except Exception as e:
                            st.error(f"Error decoding heatmap image: {e}")
                    else:
                        with col_heat:
                            st.info("No heatmap available for this model/image.")

                    # Density map visualisation below
                    if density_map:
                        density_map_np = np.array(density_map)
                        st.subheader("Density Map (Predicted Density Distribution)")
                        if density_map_np.ndim == 2:
                            dm = density_map_np - density_map_np.min()
                            if dm.max() > 0:
                                dm = (dm / dm.max()) * 255
                            st.image(
                                dm.astype(np.uint8),
                                clamp=True,
                                channels="L",
                                use_container_width=True,
                            )
                        else:
                            st.write("Density map has unexpected shape; showing array:")
                            st.write(density_map_np)
                    else:
                        st.info("ℹ This model does not generate a density map.")

    # Show inference history
    if st.session_state["inference_history"]:
        st.markdown("---")
        st.subheader("Inference History (this session)")
        hist_df = pd.DataFrame(st.session_state["inference_history"])
        st.dataframe(hist_df, use_container_width=True)
        st.caption(
            "You can download this table from the ⋮ menu on the top-right of the dataframe."
        )
