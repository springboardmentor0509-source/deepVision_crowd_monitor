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

# SECTION: DATA VISUALIZATION
elif section == "Data Visualization":

    st.title("📊 Data Visualization / EDA Results")

    # -------------------------------------------------------
    # 1. Check folder exists
    # -------------------------------------------------------
    if not EDA_DIR.exists():
        st.error(f"❌ EDA folder missing: {EDA_DIR}")
        st.stop()

    # Fetch all items and keep only actual files
    all_items = list_files_safe(EDA_DIR)
    files = [p for p in all_items if p.is_file()]   # <<🔥 IMPORTANT FIX

    if not files:
        st.info("No EDA files found.")
        st.stop()

    # -------------------------------------------------------
    # 2. Categorize files by type
    # -------------------------------------------------------
    img_ext = {".png", ".jpg", ".jpeg"}
    csv_ext = {".csv"}
    txt_ext = {".txt", ".log"}

    image_files = []
    csv_files = []
    text_files = []
    others = []

    for f in files:
        ext = f.suffix.lower()
        if ext in img_ext:
            image_files.append(f)
        elif ext in csv_ext:
            csv_files.append(f)
        elif ext in txt_ext:
            text_files.append(f)
        else:
            others.append(f)

    # -------------------------------------------------------
    # 3. Create Tabs for File Groups
    # -------------------------------------------------------
    tab_images, tab_csv, tab_text, tab_other = st.tabs(
        ["🖼 Images", "📑 CSV Data", "📝 Text Logs", "📦 Others"]
    )

    # -------------------------------------------------------
    # 4. Image Tab
    # -------------------------------------------------------
    with tab_images:
        st.header("Image Plots / Visual Charts")

        if image_files:
            cols = st.columns(3)
            for i, p in enumerate(image_files):
                img = read_image(p)
                if img is not None:
                    with cols[i % 3]:
                        st.image(img, caption=p.name, use_container_width=True)
                else:
                    st.warning(f"Cannot load: {p.name}")
        else:
            st.info("No image files found.")

    # -------------------------------------------------------
    # 5. CSV Tab
    # -------------------------------------------------------
    with tab_csv:
        st.header("CSV Outputs")

        if csv_files:
            for p in csv_files:
                st.subheader(p.name)
                try:
                    df = pd.read_csv(p)
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        "⬇ Download CSV",
                        data=p.read_bytes(),
                        file_name=p.name,
                    )
                except Exception as e:
                    st.error(f"Error reading {p.name}: {e}")
        else:
            st.info("No CSV files found.")

    # -------------------------------------------------------
    # 6. Text Logs Tab
    # -------------------------------------------------------
    with tab_text:
        st.header("Text / Log Files")

        if text_files:
            for p in text_files:
                st.subheader(p.name)
                try:
                    text = p.read_text(errors="ignore")
                    st.code(text, language="text")

                    st.download_button(
                        "⬇ Download File",
                        data=p.read_bytes(),
                        file_name=p.name,
                    )
                except Exception as e:
                    st.error(f"Error loading {p.name}: {e}")
        else:
            st.info("No text files found.")

    # -------------------------------------------------------
    # 7. Other Files Tab
    # -------------------------------------------------------
    with tab_other:
        st.header("Other File Types")

        if others:
            for p in others:
                st.subheader(p.name)
                st.write("Unknown format — cannot preview.")
                st.download_button(
                    "⬇ Download",
                    data=p.read_bytes(),
                    file_name=p.name,
                )
        else:
            st.info("No other file types found.")



# SECTION: MODEL EVALUATION RESULTS
elif section == "Model Evaluation Results":

    st.title("📊 Model Evaluation Results")
    st.write("Select a model below to view its evaluation metrics, plots, and reports.")

    # -----------------------------
    # 1. Model selection
    # -----------------------------
    model_choice = st.selectbox(
        "Select model results folder",
        list(MODEL_RESULT_DIRS.keys()),
        help="Choose a model to display its evaluation output files."
    )

    model_dir = MODEL_RESULT_DIRS.get(model_choice)
    st.write("📁 Results folder:", str(model_dir))

    # -----------------------------
    # 2. Validate folder existence
    # -----------------------------
    if not model_dir or not model_dir.exists():
        st.error(f"❌ Results folder not found for {model_choice}: {model_dir}")
        st.stop()

    # -----------------------------
    # 3. List files safely
    # -----------------------------
    files = list_files_safe(model_dir)

    if not files:
        st.info("ℹ️ No evaluation files found for this model.")
        st.stop()

    # -----------------------------
    # 4. File type grouping
    # -----------------------------
    file_types = sorted({(p.suffix.lower() or "noext") for p in files})
    chosen_type = st.selectbox(
        "Filter by file type",
        ["All"] + file_types,
        help="Filter results by extension (e.g., .csv, .txt, .png)."
    )

    filtered_files = (
        files if chosen_type == "All"
        else [p for p in files if (p.suffix.lower() or "noext") == chosen_type]
    )

    # -----------------------------
    # 5. Separate images from other files
    # -----------------------------
    image_exts = [".png", ".jpg", ".jpeg"]
    image_files = [p for p in filtered_files if p.suffix.lower() in image_exts]
    other_files = [p for p in filtered_files if p not in image_files]

    # -----------------------------
    # 6. Display image files (plots) in a grid
    # -----------------------------
    if image_files:
        st.subheader("📈 Visual Outputs (Charts / Plots)")
        cols = st.columns(3)
        for i, p in enumerate(image_files):
            with cols[i % 3]:
                img = read_image(p)
                if img is not None:
                    st.image(img, caption=p.name, use_container_width=True)
                else:
                    st.warning(f"Could not read image: {p.name}")

    # -----------------------------
    # 7. Display non-image files
    # -----------------------------
    if other_files:
        st.subheader("📄 Reports / Metrics Files")
        for p in other_files:
            with st.expander(p.name):
                display_file_preview(p)
                try:
                    data = p.read_bytes()
                    download_link_bytes(data, p.name, "⬇ Download file")
                except Exception as err:
                    st.error(f"Error loading download link: {err}")


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
