import os
from pathlib import Path
import io

import streamlit as st
import pandas as pd
from PIL import Image
import requests


# ---------------- PATHS ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"    # experiments live here: results/experiment_name/


# -------------- HELPERS ----------------
def list_experiments():
    """Return list of experiment folders under results/."""
    if not RESULTS_ROOT.exists():
        return []
    return sorted([p.name for p in RESULTS_ROOT.iterdir() if p.is_dir()])


def list_files(exp_name):
    """Return (images, csvs) inside results/exp_name."""
    exp_dir = RESULTS_ROOT / exp_name
    if not exp_dir.exists():
        return [], []

    imgs = sorted(
        [p for p in exp_dir.iterdir()
         if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )
    csvs = sorted(
        [p for p in exp_dir.iterdir()
         if p.suffix.lower() == ".csv"]
    )
    return imgs, csvs


def call_backend(api_url: str, model_name: str, image_bytes: bytes):
    """
    Call backend API for inference.

    Expected FastAPI-style endpoint:

    POST {api_url}/predict/{model_name}
    file: image (multipart/form-data)

    Returns JSON from backend, or {"error": "..."} on failure.
    """
    try:
        url = api_url.rstrip("/") + f"/predict/{model_name}"
        files = {"file": ("upload.jpg", image_bytes, "image/jpeg")}
        resp = requests.post(url, files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# -------------- STREAMLIT CONFIG --------------
st.set_page_config(
    page_title="DeepVision Crowd Counting",
    layout="wide",
)

# Pages
PAGES = ["About", "Data Visualization", "Model Evaluation Results", "Live Demo"]

# Initialize session state
if "active_page" not in st.session_state:
    st.session_state.active_page = "About"
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


def render_sidebar():
    st.sidebar.markdown("### Select Section")

    # Navigation buttons
    for page in PAGES:
        clicked = st.sidebar.button(
            page,
            use_container_width=True,
            type="primary" if st.session_state.active_page == page else "secondary",
            key=f"nav_{page}",
        )
        if clicked:
            st.session_state.active_page = page

    st.session_state.api_url = st.sidebar.text_input(
        "API URL",
        value=st.session_state.api_url,
    )

    st.sidebar.markdown("---")


# Render sidebar and get current section + api_url
render_sidebar()
section = st.session_state.active_page
api_url = st.session_state.api_url


# -------- ABOUT --------
if section == "About":
    st.title("DeepVision — Crowd Monitor")

    st.markdown(
        """
        **DeepVision Crowd Counting** is an experimental project for estimating
        crowd density and counts using classical ML baselines and CNN-based models
        such as CSRNet / SimpleCNN.

        This dashboard provides:

        - 📊 *Data Visualization* – plots & distributions exported from training  
        - 📈 *Model Evaluation Results* – metrics and error analysis CSVs  
        - ⚙️ *Live Demo* – run inference against a backend API on custom images  
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Features")
        st.markdown(
            """
            - Lightweight local dashboard  
            - Pluggable backend (any model behind `/predict`)  
            - Results folder–based, no DB setup required  
            """
        )

    with col2:
        st.subheader("Typical Workflow")
        st.markdown(
            """
            1. Run training / evaluation scripts  
            2. Export metrics & plots into `results/<experiment_name>/`  
            3. Use this UI to inspect results  
            4. Connect backend and test inference on new images  
            """
        )


# -------- DATA VISUALIZATION --------
elif section == "Data Visualization":
    st.title("Data Visualization — Result Plots")

    experiments = list_experiments()
    if not experiments:
        st.warning(
            "No experiment folders found in `results/`. "
            "Create something like `results/mobile_csrnet/` with PNG plots."
        )
    else:
        exp_name = st.selectbox("Choose an experiment folder", experiments, index=0)
        imgs, csvs = list_files(exp_name)

        exp_dir = RESULTS_ROOT / exp_name
        st.caption(f"Results folder: `{exp_dir}`")

        filter_ext = st.selectbox("Filter images by type", ["All", ".png", ".jpg"])
        if filter_ext != "All":
            imgs = [p for p in imgs if p.suffix.lower() == filter_ext]

        if not imgs:
            st.info("No images found in this experiment yet.")
        else:
            for img_path in imgs:
                with st.expander(img_path.name, expanded="gt_vs_pred" in img_path.name):
                    try:
                        img = Image.open(img_path)
                        st.image(img, width=700)
                    except Exception as e:
                        st.error(f"Failed to load {img_path.name}: {e}")


# -------- MODEL EVALUATION RESULTS --------
elif section == "Model Evaluation Results":
    st.title("Model Evaluation Results")

    experiments = list_experiments()
    if not experiments:
        st.warning("No experiment folders found in `results/`.")
    else:
        exp_name = st.selectbox("Choose an experiment folder", experiments, index=0)
        imgs, csvs = list_files(exp_name)
        exp_dir = RESULTS_ROOT / exp_name
        st.caption(f"Results folder: `{exp_dir}`")

        # Highlight best_model_metrics if present
        best_csv = None
        for p in csvs:
            if "best_model_metrics" in p.name:
                best_csv = p
                break

        if best_csv is not None:
            st.subheader("Best Model Summary")
            try:
                df_best = pd.read_csv(best_csv)
                st.dataframe(df_best, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read {best_csv.name}: {e}")
            st.markdown("---")

        if not csvs:
            st.info("No CSV files found in this experiment.")
        else:
            st.subheader("All Evaluation CSVs")
            for csv_path in csvs:
                with st.expander(csv_path.name, expanded=("training_metrics" in csv_path.name)):
                    try:
                        df = pd.read_csv(csv_path)
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to read {csv_path.name}: {e}")

                    with open(csv_path, "rb") as f:
                        st.download_button(
                            label="Download file",
                            data=f,
                            file_name=csv_path.name,
                            mime="text/csv",
                            key=str(csv_path),
                        )

        # Optional: quick access to key plots below
        if imgs:
            st.markdown("### Key Plots")
            for img_path in imgs:
                if any(k in img_path.name for k in ["gt_vs_pred", "error_histogram", "residual"]):
                    with st.expander(img_path.name):
                        try:
                            img = Image.open(img_path)
                            st.image(img, width=700)
                        except Exception as e:
                            st.error(f"Failed to load {img_path.name}: {e}")


# -------- LIVE DEMO --------
elif section == "Live Demo":
    st.title("Live Demo — Run Model Inference")

    st.markdown(
        """
        Upload an image and choose one of the available models to predict crowd count.
        This UI simply sends your image to the backend API; any model can be plugged in
        as long as the API contract is respected.
        """
    )

    model_name = st.selectbox(
        "Choose a Model",
        ["SimpleCNN", "CSRNet", "MobileNetCSRNet", "RandomForest"],
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        help="Limit ~20MB per file",
    )

    if uploaded_file is not None:
        # Show preview
        img_bytes = uploaded_file.read()
        st.image(io.BytesIO(img_bytes), caption="Uploaded Image", width=600)

        if st.button("Predict Crowd Count"):
            with st.spinner("Running inference via backend..."):
                result = call_backend(api_url, model_name, img_bytes)

            if "error" in result:
                st.error(f"Backend error: {result['error']}")
            else:
                count = result.get("predicted_count", None)
                if count is not None:
                    st.success(f"Predicted Crowd Count: **{int(count)}**")

                    # Show heatmap if available
                    if result.get("heatmap_image"):
                        st.subheader("Density Heatmap")
                        try:
                            heatmap_bytes = bytes.fromhex(result["heatmap_image"])
                            st.image(
                                io.BytesIO(heatmap_bytes),
                                caption="Density Map Overlay",
                                width=600,
                            )
                        except Exception as e:
                            st.warning(f"Could not display heatmap: {e}")

                    # Show additional info (without the large hex string)
                    with st.expander("View Detailed Response"):
                        info = {
                            k: v
                            for k, v in result.items()
                            if k not in ["heatmap_image", "density_map"]
                        }
                        st.json(info)
                else:
                    st.warning(
                        "Backend did not return a `predicted_count` field. Full response:"
                    )
                    st.json(result)
    else:
        st.info("Upload an image to enable prediction.")
