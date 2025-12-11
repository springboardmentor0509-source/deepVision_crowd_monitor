import os
from pathlib import Path
import io
import base64
import requests
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Page config
st.set_page_config(page_title="DeepVision Crowd Counting", layout="wide", page_icon="👁️")

# --- Paths / constants (make these configurable)
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = (ROOT / "../results").resolve()
EDA_DIR = RESULTS_ROOT / "eda_results"

MODEL_RESULT_DIRS = {
    "CSRNet": RESULTS_ROOT / "csrnet_cnn",
    "MobileNetCSRNet": RESULTS_ROOT / "mobile_csrnet",
    "RandomForest": RESULTS_ROOT / "random_forest",
    "SimpleCNN": RESULTS_ROOT / "simple_cnn",
}

# --- Sidebar navigation / configuration
st.sidebar.title("DeepVision Controls")
section = st.sidebar.radio(
    "Select Section",
    ["About", "Data Visualization", "Model Evaluation Results", "Live Demo"],
    key="main_section",
)
backend_url = st.sidebar.text_input("API URL", "http://localhost:8000", key="backend_url")
# Optional: allow EDA_DIR override from sidebar (useful)
eda_dir_override = st.sidebar.text_input("EDA folder (optional)", value=str(EDA_DIR), key="eda_dir_override")
try:
    if eda_dir_override and eda_dir_override.strip():
        EDA_DIR = Path(eda_dir_override).expanduser().resolve()
except Exception:
    pass

# --- Caching helpers
@st.cache_data
def list_files_safe(path: Path):
    """Return sorted list of Path objects for directory, empty list on error."""
    try:
        if not path.exists() or not path.is_dir():
            return []
        return sorted([p for p in path.iterdir()], key=lambda p: p.name.lower())
    except Exception:
        return []

@st.cache_data
def read_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

@st.cache_data
def read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

@st.cache_data
def read_csv(path: Path):
    try:
        # try common separators
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return None

# --- Small utilities
def download_link_bytes(content_bytes: bytes, filename: str, label: str = "Download"):
    b64 = base64.b64encode(content_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def display_file_preview(path: Path):
    suffix = path.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
        img = read_image(path)
        if img:
            # Use new API: responsive stretch
            st.image(img, caption=path.name, width="stretch")
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

# --- ABOUT
if section == "About":
    st.title(" DeepVision Crowd Counting — About")
    st.markdown(
        """
        **DeepVision — Crowd Counting / Density Estimation**  
        DeepVision is a compact, end-to-end crowd counting system combining a Streamlit front-end with a FastAPI backend for inference and results management. The goal is to provide researchers and students a lightweight but practical tool to run crowd counting experiments, visualize results, and inspect model outputs without the overhead of heavy dashboards.

        **What this project implements:**
        - **Multi-model support** — CSRNet, MobileNetCSRNet, SimpleCNN and a RandomForest baseline served by a FastAPI backend.
        - **Density-map support** — backend can supply per-pixel density maps and overlay heatmaps; frontend decodes and displays them.
        - **Live demo / inference pipeline** — upload image → FastAPI `/predict/{model}` → show predicted count, density map, heatmap and downloads.
        - **Results organization & EDA** — standardized `results/` layout with `eda_results/` for plots, CSVs and evaluation artifacts.
        - **Robust UX** — cached file reads, tolerant CSV parsing (`,` & `;`), thumbnail galleries and interactive plotting for quick validation.

        **Accomplishments:**
        - Single-file Streamlit frontend integrated with a FastAPI backend.
        - Interactive EDA tools (thumbnails, CSV exploration, histograms, scatterplots, correlation matrix).
        - Downloadable artifacts and image previews for reproducibility and reporting.

        **Intended users:** researchers, students, demos and small prototypes.
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
        """
    )

    st.subheader("Edit Short Description")
    desc = st.text_area("Project short description", value="DeepVision Crowd Counting Monitor — upload images and estimate crowd size.", key="short_desc")
    if st.button("Save description (session only)", key="save_desc"):
        st.success("Saved (session-local).")

# --- DATA VISUALIZATION
elif section == "Data Visualization":
    st.title("Data Visualization / EDA Results")
    st.markdown(
        """
        **Data Visualization / EDA Results** — quick reference & user guide

        This section is designed as the first place to inspect everything your model training or EDA pipeline produces. It focuses on fast visual checks and simple, interactive analysis so you can quickly validate pipelines and spot issues.

        **Features & workflow:**
        1. **Directory summary** — Displays counts for all artifacts found in `results/eda_results/` (total files, image outputs, CSV/tables). Useful as a quick sanity check after running training or an EDA script to make sure outputs were generated.
        2. **Image gallery & preview** — Shows up to 9 thumbnails of images located in the EDA folder (plots, sample images, heatmaps, loss curves). Select any thumbnail to open a larger preview and download the original file.
        3. **Interactive CSV exploration** — Pick a CSV (for example, `predictions.csv`, `per_image_counts.csv`, or `feature_stats.csv`) and view its head and shape. If numeric columns are present, plot histograms (adjustable), scatter plots, and compute a correlation matrix heatmap to find strong pairwise correlations or data issues.
        4. **Reports & logs** — Expand text-based artifacts (markdown reports, logs, JSON) to read run summaries and error traces inside the app and download them if needed.

        **Why this helps:**
        - Rapid validation: visually confirm whether density maps/heatmaps align with expected high-density regions.
        - Data quality checks: detect missing values, abnormal ranges, or mislabeled columns via histograms and dtype tables.
        - Reproducibility: download artifacts for reporting and for attaching to GitHub issues / research notebooks.

        **Tips & recommended workflow:**
        - After a training run, place evaluation CSVs and plot images into `results/eda_results/` and refresh the app — use the directory summary to confirm outputs exist.
        - Inspect histograms for per-image counts to ensure predicted ranges match dataset expectations.
        - Use the correlation matrix to identify unexpected relationships, then revisit preprocessing if needed.
        - If a file fails to preview, check the file encoding or delimiter (the CSV reader tries `,` and `;` by default).
        """
    )
    st.write("Showing contents of:", str(EDA_DIR))

    if not EDA_DIR.exists():
        st.error(f"EDA folder not found: {EDA_DIR}")
    else:
        files = list_files_safe(EDA_DIR)
        if not files:
            st.info("No EDA result files found in the folder.")
        else:
            # Basic stats
            total_files = len(files)
            image_files = [p for p in files if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            csv_files = [p for p in files if p.suffix.lower() == ".csv"]
            other_files = [p for p in files if p not in image_files + csv_files]

            st.markdown("**Summary**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total files", total_files)
            c2.metric("Image outputs", len(image_files))
            c3.metric("CSV / tables", len(csv_files))

            # Show image samples as a small gallery (single, correctly-indented block)
            if image_files:
                st.subheader("Image outputs (plots / charts)")

                # Prepare image list (up to 9)
                thumbs = []
                for p in image_files[:9]:
                    img = read_image(p)
                    if img:
                        thumbs.append((p.name, img))

                # Display equal-size images in a clean 3-column grid with spacing
                if thumbs:
                    cols = st.columns(3, gap="large")
                    for idx, (name, img) in enumerate(thumbs):
                        with cols[idx % 3]:
                            st.markdown("<div style='text-align:center; margin-bottom:10px;'>", unsafe_allow_html=True)
                            # fixed width ensures equal-size thumbnails
                            st.image(img, width=250)
                            st.markdown(f"<p style='text-align:center; font-size:14px; margin-top:6px;'>{name}</p>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                # Dropdown for full preview (unique key)
                preview_choice = st.selectbox("Preview which image?", [p.name for p in image_files], key="preview_choice")
                chosen_path = next((p for p in image_files if p.name == preview_choice), None)
                if chosen_path:
                    display_file_preview(chosen_path)
                    try:
                        download_link_bytes(chosen_path.read_bytes(), chosen_path.name, "Download file")
                    except Exception:
                        pass

            # CSV interactive exploration
            if csv_files:
                st.subheader("Interactive CSV exploration")
                csv_choice = st.selectbox("Select CSV file to explore", [p.name for p in csv_files], key="csv_choice")
                chosen_csv = next((p for p in csv_files if p.name == csv_choice), None)
                if chosen_csv:
                    df = read_csv(chosen_csv)
                    if df is None:
                        st.error("Failed to read CSV — file might be malformed or use an uncommon delimiter.")
                    else:
                        st.write("Data shape:", df.shape)
                        st.dataframe(df.head(200))

                        st.markdown("**Column types / quick stats**")
                        st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]).T)

                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            st.markdown("**Numeric column visualizations**")
                            col1, col2 = st.columns(2)
                            hist_col = col1.selectbox("Histogram column", numeric_cols, key="hist_col")
                            bins = col1.slider("Bins", 5, 200, 30, key="bins_slider")

                            # Use sample for large CSVs to keep UI responsive
                            df_for_plot = df
                            MAX_SAMPLE = 50000
                            if df.shape[0] > MAX_SAMPLE:
                                st.warning(f"Large CSV detected ({df.shape[0]} rows). Sampling {MAX_SAMPLE} rows for plotting.")
                                df_for_plot = df.sample(n=MAX_SAMPLE, random_state=42)

                            fig1, ax1 = plt.subplots()
                            ax1.hist(df_for_plot[hist_col].dropna(), bins=bins)
                            ax1.set_title(f"Histogram: {hist_col}")
                            st.pyplot(fig1)

                            # Download histogram as PNG
                            buf = io.BytesIO()
                            fig1.savefig(buf, format="png", bbox_inches="tight")
                            buf.seek(0)
                            st.download_button(label="Download histogram as PNG", data=buf, file_name=f"{hist_col}_hist.png", mime="image/png", key="download_hist")
                            buf.close()

                            # Scatter plot controls
                            if len(numeric_cols) >= 2:
                                x_col = col2.selectbox("X axis (scatter)", numeric_cols, index=0, key="x_col")
                                y_col = col2.selectbox("Y axis (scatter)", numeric_cols, index=1, key="y_col")
                                fig2, ax2 = plt.subplots()
                                ax2.scatter(df_for_plot[x_col].dropna(), df_for_plot[y_col].dropna(), alpha=0.6)
                                ax2.set_xlabel(x_col)
                                ax2.set_ylabel(y_col)
                                ax2.set_title(f"Scatter: {x_col} vs {y_col}")
                                st.pyplot(fig2)

                        # Correlation heatmap (if numeric)
                        if len(numeric_cols) >= 2:
                            st.markdown("**Correlation matrix**")
                            corr = df[numeric_cols].corr()
                            fig3, ax3 = plt.subplots(figsize=(6, 4))
                            im = ax3.imshow(corr, vmin=-1, vmax=1)
                            ax3.set_xticks(range(len(numeric_cols)))
                            ax3.set_yticks(range(len(numeric_cols)))
                            ax3.set_xticklabels(numeric_cols, rotation=45, ha="right")
                            ax3.set_yticklabels(numeric_cols)
                            fig3.colorbar(im, ax=ax3)
                            st.pyplot(fig3)

                        # allow CSV download
                        try:
                            data = chosen_csv.read_bytes()
                            download_link_bytes(data, chosen_csv.name, "Download CSV")
                        except Exception:
                            pass

            # Other files
            if other_files:
                st.subheader("Other outputs (reports / logs)")
                for p in other_files:
                    with st.expander(p.name):
                        display_file_preview(p)
                        try:
                            data = p.read_bytes()
                            download_link_bytes(data, p.name, "Download file")
                        except Exception:
                            pass

# --- MODEL EVALUATION RESULTS
elif section == "Model Evaluation Results":
    st.title(" Model Evaluation Results")
    st.write("Choose a model to view its evaluation outputs / reports.")
    model_choice = st.selectbox("Select model results folder", list(MODEL_RESULT_DIRS.keys()), key="model_choice")

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
            chosen_type = st.selectbox("Filter by file extension", ["All"] + file_types, key="file_type")
            filtered = files if chosen_type == "All" else [p for p in files if (p.suffix.lower() or "noext") == chosen_type]

            for p in filtered:
                with st.expander(p.name, expanded=False):
                    display_file_preview(p)
                    try:
                        data = p.read_bytes()
                        download_link_bytes(data, p.name, "Download file")
                    except Exception:
                        pass

# --- LIVE DEMO
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
    selected_model = st.selectbox("Choose a Model", models, key="selected_model")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], help="Upload an image to run crowd counting.", key="uploader")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width="stretch")
        except Exception as e:
            st.error(f"Unable to open image: {e}")

    if uploaded_file and st.button(" Predict Crowd Count", key="predict_button"):
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
                if density_map is not None:
                    density_map_np = np.array(density_map)
                    st.subheader("Density Map (Predicted Density Distribution)")
                    if density_map_np.ndim == 2:
                        dm = density_map_np - density_map_np.min()
                        if dm.max() > 0:
                            dm = (dm / dm.max()) * 255
                        st.image(dm.astype(np.uint8), clamp=True, channels="L", width="stretch")
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
                        st.image(heatmap_img, caption="Heatmap Overlay", width="stretch")
                        download_link_bytes(heatmap_bytes, f"heatmap_{uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error decoding heatmap image: {e}")
                else:
                    st.info("ℹ No heatmap available for this model (SimpleCNN / RandomForest).")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(" DeepVision Crowd Counting")
