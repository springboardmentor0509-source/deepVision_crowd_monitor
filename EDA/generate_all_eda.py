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

# Correct project root (folder where YOUR app.py is located)
PROJECT_ROOT = Path(__file__).resolve().parent

# EDA folder is directly under project_root/eda
EDA_DIR = PROJECT_ROOT / "eda"

# Model results are in project_root/results/...
RESULTS_ROOT = PROJECT_ROOT / "results"

MODEL_RESULT_DIRS = {
    "CSRNet": RESULTS_ROOT / "csrnet_cnn",
    "MobileNetCSRNet": RESULTS_ROOT / "mobile_csrnet",
    "RandomForest": RESULTS_ROOT / "random_forest",
    "SimpleCNN": RESULTS_ROOT / "simple_cnn",
}


# SIDEBAR NAVIGATION

st.sidebar.title("DeepVision Controls")
section = st.sidebar.radio("Select Section", ["About", "Data Visualization", "Model Evaluation Results", "Live Demo"])

# Backend URL
backend_url = st.sidebar.text_input("API URL", "http://localhost:8000")

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
    # Skip directories (e.g., __pycache__)
    if path.is_dir():
        st.info(f"[Directory] {path.name} — skipping")
        return

    suffix = path.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
        img = read_image(path)
        if img:
            st.image(img, caption=path.name, use_container_width=True)
        else:
            st.write(f"Unable to open image: {path.name}")

    elif suffix == ".csv":
        df = read_csv(path)
        if df is not None:
            st.dataframe(df)
        else:
            st.write(f"Unable to read CSV: {path.name}")

    elif suffix in [".txt", ".md", ".json", ".log", ".py"]:
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
# SECTION: ABOUT
if section == "About":
    st.title("✨ DeepVision Crowd Monitor — About")

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

    # Styling
    st.markdown("""
    <style>
    .feature-card {
        padding: 15px 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    .feature-title {
        font-size: 20px;
        font-weight: 600;
        color: #2c3e50;
    }
    .feature-desc {
        font-size: 15px;
        color: #444;
    }
    .section-header {
        font-size: 26px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Intro
    st.markdown("""
    DeepVision Crowd Monitor is a **real-time crowd analysis system** capable of estimating crowd counts,
    generating density heatmaps, and issuing overcrowding alerts using advanced deep learning models.

    Built with:
    - 🚀 **FastAPI** (backend – ML inference)
    - 🎨 **Streamlit** (frontend – live dashboard)
    - 🧠 **CSRNet / MobileNetCSRNet / SimpleCNN / RandomForest**
    ---
    """)

    # Advantages
    st.markdown("<div class='section-header'>🌟 Advantages</div>", unsafe_allow_html=True)

    advantages = [
        ("⚡ Real-time Analytics", "Processes images and videos quickly, enabling fast operational decisions."),
        ("🤖 Multi-Model Support", "Choose between CSRNet, MobileNetCSRNet, SimpleCNN, and RandomForest models."),
        ("🔥 Heatmap Visualization", "Density heatmaps highlight crowd hotspots instantly."),
        ("🧩 Modular & Scalable", "Backend + frontend design ensures easy integration & updates."),
        ("📱 Edge-Friendly", "MobileNetCSRNet runs efficiently on devices like Jetson Nano & Raspberry Pi."),
        ("🔔 Automatic Alerts", "Trigger safety alerts when crowd count crosses threshold."),
    ]

    for title, desc in advantages:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Use Cases
    st.markdown("<div class='section-header'>🎯 Use Cases</div>", unsafe_allow_html=True)

    use_cases = [
        ("🏟️ Public Safety", "Monitor large gatherings, protests, festivals, and concerts."),
        ("🚉 Smart Cities", "Analyze density patterns in metros, airports, and bus stands."),
        ("🎪 Event Management", "Track crowd flow in stadiums, malls, expos, and conferences."),
        ("🛍️ Retail Analytics", "Customer density mapping for malls & supermarkets."),
        ("🚨 Disaster Management", "Monitor evacuation patterns & detect dangerous bottlenecks."),
        ("🎓 Campus Safety", "Detect abnormal crowding in universities & hostels."),
        ("🏛️ Tourism Sites", "Manage visitor flow at temples, museums, and historical monuments."),
    ]

    cols = st.columns(3)
    for i, (title, desc) in enumerate(use_cases):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.info("💡 DeepVision is modular — extendable for CCTV feeds, drone footage, or live video analytics.")

elif section == "Data Visualization":

    st.title("📊 Data Visualization & Insights")
    st.write(
        "This section presents the visual analysis performed during the EDA "
        "(Exploratory Data Analysis) stage. The visuals help understand dataset "
        "patterns, density behavior, feature influence, and overall crowd structure."
    )

    EDA_ROOT = RESULTS_ROOT / "eda_results"

    if not EDA_ROOT.exists():
        st.error("❌ EDA results folder not found.")
        st.stop()

    # ---- Helper: Create a visual card layout ----
    def show_visual_section(title, folder, descriptions):
        folder_path = EDA_ROOT / folder

        if not folder_path.exists():
            return  # Skip missing sections

        st.markdown(f"## 🎨 {title}")

        img_files = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

        if not img_files:
            return

        cols = st.columns(2)

        for i, img_path in enumerate(img_files):
            col = cols[i % 2]

            with col:
                # consistent size for symmetry
                col.image(str(img_path), width=450)

                # description lookup
                desc = descriptions.get(img_path.stem, "Visualization generated during the EDA process.")
                col.caption(f"📝 {desc}")

            st.write("")  # spacing


    # ---- SECTION 1: Dataset Samples ----
    show_visual_section(
        "Dataset Sample Visualizations",
        "samples",
        {
            "random_samples": "A set of randomly selected samples from the dataset with ground-truth head annotations.",
            "sample_annotated": "Annotated dataset sample showcasing complexity and crowd density."
        }
    )

    # ---- SECTION 2: Distribution Visuals ----
    show_visual_section(
        "Crowd Count Distribution",
        "distribution",
        {
            "histogram_counts": "Histogram showing distribution of crowd counts across Part A & Part B.",
            "boxplot_parts": "Boxplot comparing crowd count spread between dataset partitions."
        }
    )

    # ---- SECTION 3: Correlation Visuals ----
    show_visual_section(
        "Correlation & Feature Relationships",
        "correlation",
        {
            "brightness_vs_count": "Scatter plot showing relationship between image brightness and crowd count.",
            "entropy_vs_count": "Entropy vs People Count — measures texture complexity vs number of heads.",
            "correlation_matrix": "Correlation matrix indicating statistical dependencies between features."
        }
    )

    # ---- SECTION 4: Density Maps ----
    show_visual_section(
        "Density Map Visualizations",
        "density",
        {
            "density_map_example": "Generated density map using Gaussian kernels on annotated head positions.",
            "heatmap_overlay_demo": "Density heatmap overlayed on the original image for visual explanation."
        }
    )

    # ---- SECTION 5: General Feature Insights ----
    show_visual_section(
        "Feature & Image Property Insights",
        "features",
        {
            "image_sizes_distribution": "Distribution of image dimensions across dataset.",
            "scatter_density_vs_gt": "Density map sum vs ground-truth count — correctness of density estimation."
        }
    )


elif section == "Model Evaluation Results":
    st.title("Model Evaluation Results")
    model_choice = st.selectbox("Choose a model", list(MODEL_RESULT_DIRS.keys()))
    model_dir = MODEL_RESULT_DIRS[model_choice]

    if not model_dir.exists():
        st.error(f"No directory found for: {model_choice}")
    else:
        files = list_files_safe(model_dir)
        if not files:
            st.info("No evaluation files found.")
        else:
            file_types = sorted({p.suffix.lower() or "noext" for p in files})
            chosen_type = st.selectbox("Filter by extension", ["All"] + file_types)

            selection = files if chosen_type == "All" else [p for p in files if (p.suffix.lower() or "noext") == chosen_type]

            for p in selection:
                with st.expander(p.name):
                    display_file_preview(p)

                    # Skip directories (cannot download)
                    if p.is_dir():
                        st.info(f"[Directory] {p.name} — skipping download")
                        continue

                    download_link_bytes(p.read_bytes(), p.name)



# LIVE DEMO
elif section == "Live Demo":
    st.title("Live Demo — Run Model Inference")

    def fetch_models(url):
        try:
            r = requests.get(f"{url}/models", timeout=3)
            return r.json().get("models", [])
        except:
            return []

    models = fetch_models(backend_url)
    selected_model = st.selectbox("Select Model", models)

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(Image.open(uploaded), caption="Uploaded Image", use_container_width=True)

    if uploaded and st.button("Predict Crowd Count"):
        uploaded.seek(0)
        files = {"file": (uploaded.name, uploaded, uploaded.type)}

        with st.spinner("Running inference..."):
            resp = requests.post(f"{backend_url}/predict/{selected_model}", files=files)

        if resp.status_code != 200:
            st.error(resp.text)
        else:
            data = resp.json()

            count = data.get("predicted_count", 0)
            st.success(f"### Predicted Crowd Count: **{count:.2f}**")

            # decode Base64 heatmap
            heatmap_b64 = data.get("heatmap_image", None)

            if heatmap_b64:
                heatmap_bytes = base64.b64decode(heatmap_b64)
                heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

                st.subheader("Heatmap Overlay")
                st.image(heatmap_img, use_container_width=True)
                download_link_bytes(heatmap_bytes, f"heatmap_{uploaded.name}")
            else:
                st.info("No heatmap produced for this model.")
