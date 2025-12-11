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
    st.title("🎯 DeepVision — Crowd Monitor")
    
    st.markdown("---")
    
    # Project Overview
    st.header("📖 Project Overview")
    st.markdown(
        """
        **DeepVision Crowd Monitor** is an advanced AI-powered system for real-time crowd density estimation 
        and counting in images and video streams. This project leverages state-of-the-art deep learning 
        architectures and classical machine learning techniques to provide accurate crowd analysis for 
        various applications.
        
        The system combines multiple approaches including:
        - **Deep CNN Models** (CSRNet, SimpleCNN, MobileNet-CSRNet)
        - **Classical ML** (Random Forest baseline)
        - **Geometry-Adaptive Density Maps** for precise crowd distribution visualization
        """
    )
    
    st.markdown("---")
    
    # Dataset Information
    st.header("📊 Dataset: ShanghaiTech")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
            This project uses the **ShanghaiTech Crowd Counting Dataset**, one of the most challenging 
            and widely-used benchmarks in crowd analysis research.
            
            **Dataset Specifications:**
            - **Part A**: 482 images (300 train + 182 test)
              - Dense crowds from the Internet
              - Average count: ~500 people per image
              - Resolution: Variable (avg ~768×1024)
              
            - **Part B**: 716 images (400 train + 316 test)
              - Sparse crowds from metropolitan streets
              - Average count: ~120 people per image
              - Resolution: Variable (avg ~1024×768)
            
            **Ground Truth Annotations:**
            - Manually annotated head positions
            - Stored in `.mat` files (MATLAB format)
            - Contains precise (x, y) coordinates for each person
            """
        )
    
    with col2:
        st.info(
            """
            **Dataset Stats**
            
            📈 Total Images: 1,198
            
            👥 Total Annotated People: ~330,000+
            
            🎯 Challenge: High density variations
            
            🌍 Real-world scenarios
            """
        )
    
    st.markdown("---")
    
    # Use Cases
    st.header("💡 Use Cases & Applications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            ### 🏛️ Public Safety
            - Event management
            - Emergency evacuation
            - Crowd flow monitoring
            - Overcrowding prevention
            """
        )
    
    with col2:
        st.markdown(
            """
            ### 🏙️ Smart Cities
            - Traffic management
            - Public transport optimization
            - Urban planning insights
            - Pedestrian flow analysis
            """
        )
    
    with col3:
        st.markdown(
            """
            ### 🏢 Business Intelligence
            - Retail foot traffic analysis
            - Queue management
            - Customer behavior insights
            - Peak hour detection
            """
        )
    
    st.markdown("---")
    
    # System Architecture
    st.header("⚙️ System Architecture & Workflow")
    
    st.markdown(
        """
        ### 🔄 Complete Pipeline
        
        **1. Data Preprocessing** 📥
        - Load raw images and ground truth annotations
        - Generate geometry-adaptive density maps using k-NN
        - Resize images to consistent dimensions (1024px)
        - Create metadata CSV files for efficient loading
        - Store preprocessed data for faster training
        
        **2. Model Training** 🧠
        
        **Available Models:**
        """
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            **Deep Learning Models:**
            - **SimpleCNN**: Lightweight encoder-decoder
              - 3-layer encoder with max pooling
              - 3-layer decoder with convolutions
              - ~2M parameters
              
            - **CSRNet**: VGG16-based architecture
              - Pre-trained VGG16 frontend
              - Dilated convolution backend
              - ~16M parameters
              
            - **MobileNet-CSRNet**: Efficient variant
              - MobileNetV2 frontend
              - Dilated backend
              - ~3M parameters
            """
        )
    
    with col2:
        st.markdown(
            """
            **Classical ML:**
            - **Random Forest Baseline**
              - Feature extraction: HOG, LBP, ORB
              - Edge density analysis
              - Brightness statistics
              - 200 trees, depth 20
            
            **Training Configuration:**
            - Loss: MSE (Mean Squared Error)
            - Optimizer: Adam
            - Learning Rate: 1e-4 (CNN), 1e-5 (CSRNet)
            - Batch Size: 4
            - Epochs: 20-30
            """
        )
    
    st.markdown(
        """
        **3. Evaluation** 📊
        - Calculate MAE (Mean Absolute Error)
        - Calculate RMSE (Root Mean Squared Error)
        - Generate prediction vs ground truth plots
        - Create error distribution histograms
        - Save metrics to CSV files
        
        **4. Deployment** 🚀
        - FastAPI backend for model serving
        - Streamlit dashboard for visualization
        - REST API for inference
        - Real-time density map generation
        - Heatmap overlay visualization
        """
    )
    
    st.markdown("---")
    
    # Technical Details
    st.header("🔧 Technical Implementation")
    
    tab1, tab2, tab3 = st.tabs(["Preprocessing", "Training", "Inference"])
    
    with tab1:
        st.markdown(
            """
            ### Data Preprocessing Pipeline
            
            **Step 1: Load Raw Data**
            ```
            Dataset/ShanghaiTech/
              ├── part_A/
              │   ├── train_data/
              │   │   ├── images/
              │   │   └── ground-truth/ (*.mat files)
              │   └── test_data/
              └── part_B/
            ```
            
            **Step 2: Generate Density Maps**
            - Parse MATLAB `.mat` files to extract (x, y) coordinates
            - Apply **Geometry-Adaptive Kernel** (k-NN based σ)
            - Generate Gaussian kernels at each head position
            - Adaptive sigma based on k-nearest neighbors
            - Formula: `σ = 0.3 × avg_k_nearest_distance`
            
            **Step 3: Store Preprocessed Data**
            ```
            processed_data/
              ├── part_A/
              │   ├── train_data/
              │   │   ├── density/ (*.npy files)
              │   │   ├── images_resized/
              │   │   └── metadata.csv
              │   └── test_data/
              └── part_B/
            ```
            
            **Benefits:**
            - ✅ 2-3x faster training
            - ✅ Better quality density maps
            - ✅ Consistent preprocessing
            """
        )
    
    with tab2:
        st.markdown(
            """
            ### Model Training Process
            
            **Training Loop:**
            1. Load batch of preprocessed images + density maps
            2. Forward pass through network
            3. Calculate MSE loss between predicted and ground truth density
            4. Backpropagation and parameter update
            5. Validate on test set every epoch
            6. Save best model based on MAE
            
            **Metrics Tracked:**
            - Training Loss (per epoch)
            - Validation MAE (Mean Absolute Error)
            - Validation RMSE (Root Mean Squared Error)
            - Best model checkpoint
            
            **Output Files:**
            ```
            results/[model_name]/
              ├── training_metrics.csv
              ├── predictions.csv
              └── training_plot.png
            
            models and code/
              └── best_[model_name].pth
            ```
            
            **Command to Train:**
            ```bash
            python run_simple_cnn.py
            python run_csrnet.py
            python run_mobile_csrnet.py
            python run_random_forest.py
            ```
            """
        )
    
    with tab3:
        st.markdown(
            """
            ### Inference Pipeline
            
            **Backend API (FastAPI):**
            - Endpoint: `POST /predict/{model_name}`
            - Input: Image file (JPG/PNG)
            - Output: JSON with count + density map
            
            **Processing Steps:**
            1. Receive uploaded image
            2. Preprocess: resize + normalize
            3. Forward pass through model
            4. Generate density map prediction
            5. Calculate total count (sum of density)
            6. Create heatmap overlay
            7. Return JSON response
            
            **Response Format:**
            ```json
            {
              "model": "CSRNet",
              "filename": "crowd.jpg",
              "predicted_count": 342,
              "density_map": [[...], ...],
              "heatmap_image": "hex_encoded_jpg"
            }
            ```
            
            **Frontend Dashboard:**
            - Upload image via Streamlit interface
            - Select model from dropdown
            - Display predicted count
            - Show density heatmap overlay
            - View detailed metrics
            """
        )
    
    st.markdown("---")
    
    # Dashboard Features
    st.header("🎨 Dashboard Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Visualization")
        st.markdown(
            """
            - 📊 View training loss curves
            - 📈 Explore validation metrics over epochs
            - 🖼️ Browse generated plots and charts
            - 📁 Organized by experiment folders
            - 🔍 Filter by image type (.png, .jpg)
            """
        )
        
        st.subheader("Model Evaluation")
        st.markdown(
            """
            - 📋 View metrics CSV files
            - 📉 Analyze error distributions
            - 🎯 Compare model performance
            - 💾 Download results
            - 📊 Interactive data tables
            """
        )
    
    with col2:
        st.subheader("Live Demo")
        st.markdown(
            """
            - 🖼️ Upload custom images
            - 🤖 Select AI model
            - ⚡ Real-time inference
            - 🔥 Density heatmap visualization
            - 📊 Detailed prediction metrics
            """
        )
        
        st.subheader("Backend API")
        st.markdown(
            """
            - 🚀 FastAPI REST endpoints
            - 🔌 Easy integration
            - 📡 HTTP requests
            - 🔄 Model hot-swapping
            - 📈 Scalable architecture
            """
        )
    
    st.markdown("---")
    
    # Quick Start
    st.header("🚀 Quick Start Guide")
    
    st.markdown(
        """
        ### Prerequisites
        ```bash
        pip install -r requirements.txt
        ```
        
        ### Run Preprocessing (One-time)
        ```bash
        cd preprocessing
        python run_preprocess.py
        cd ..
        ```
        
        ### Start Backend Server
        ```bash
        python start_backend.py
        ```
        *Backend will run on http://localhost:8000*
        
        ### Start Dashboard
        ```bash
        streamlit run output.py
        ```
        *Dashboard will open in your browser*
        
        ### Train Models (Optional)
        ```bash
        python run_simple_cnn.py      # Fast training (~30 min)
        python run_csrnet.py          # Medium (~1-2 hours)
        python run_mobile_csrnet.py   # Longer (~2-3 hours)
        ```
        """
    )
    
    st.markdown("---")
    st.caption("DeepVision Crowd Monitor © 2025 | Built with PyTorch, Streamlit & FastAPI")


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
