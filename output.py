import os
from pathlib import Path
import io
import json

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
    st.title("📊 Data Visualization — Training Results")
    st.markdown("Explore comprehensive visualizations from all trained models")
    
    st.markdown("---")

    experiments = list_experiments()
    if not experiments:
        st.warning(
            "No experiment folders found in `results/`. "
            "Create something like `results/mobile_csrnet/` with PNG plots."
        )
    else:
        # Sort experiments: 'eda_results' first, then others alphabetically
        def sort_experiments(exp_list):
            eda_results = [e for e in exp_list if 'eda' in e.lower()]
            other_results = [e for e in exp_list if 'eda' not in e.lower()]
            return eda_results + sorted(other_results)
        
        experiments = sort_experiments(experiments)
        
        # Display each experiment as a section
        for exp_idx, exp_name in enumerate(experiments):
            imgs, csvs = list_files(exp_name)
            exp_dir = RESULTS_ROOT / exp_name
            
            if not imgs:
                continue  # Skip experiments with no images
            
            # Experiment header
            st.header(f"🔬 {exp_name.replace('_', ' ').title()}")
            st.caption(f"📁 Location: `{exp_dir}`")
            
            # Categorize images by type
            training_plots = []
            prediction_plots = []
            other_plots = []
            
            for img_path in imgs:
                name_lower = img_path.name.lower()
                if any(x in name_lower for x in ["training", "loss", "metrics", "epoch"]):
                    training_plots.append(img_path)
                elif any(x in name_lower for x in ["predict", "gt_vs", "comparison", "sample"]):
                    prediction_plots.append(img_path)
                else:
                    other_plots.append(img_path)
            
            # Display Training Plots
            if training_plots:
                st.subheader("📈 Training Progress")
                cols = st.columns(2)
                for idx, img_path in enumerate(training_plots):
                    with cols[idx % 2]:
                        # Create a nice title from filename
                        title = img_path.stem.replace('_', ' ').title()
                        st.markdown(f"**{title}**")
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to load {img_path.name}: {e}")
                        st.caption(f"_{img_path.name}_")
                        st.markdown("")  # Spacing
            
            # Display Prediction Plots
            if prediction_plots:
                st.subheader("🎯 Prediction Results")
                cols = st.columns(2)
                for idx, img_path in enumerate(prediction_plots):
                    with cols[idx % 2]:
                        title = img_path.stem.replace('_', ' ').title()
                        st.markdown(f"**{title}**")
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to load {img_path.name}: {e}")
                        st.caption(f"_{img_path.name}_")
                        st.markdown("")
            
            # Display Other Plots
            if other_plots:
                st.subheader("📊 Additional Visualizations")
                cols = st.columns(2)
                for idx, img_path in enumerate(other_plots):
                    with cols[idx % 2]:
                        title = img_path.stem.replace('_', ' ').title()
                        st.markdown(f"**{title}**")
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to load {img_path.name}: {e}")
                        st.caption(f"_{img_path.name}_")
                        st.markdown("")
            
            # Add summary statistics if CSV files exist
            if csvs:
                with st.expander(f"📄 View Data Files ({len(csvs)} available)"):
                    csv_name = st.selectbox(
                        "Select CSV file",
                        [c.name for c in csvs],
                        key=f"csv_{exp_name}"
                    )
                    selected_csv = next(c for c in csvs if c.name == csv_name)
                    try:
                        import pandas as pd
                        df = pd.read_csv(selected_csv)
                        st.dataframe(df, use_container_width=True)
                        
                        # Show quick stats for numeric columns
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        if len(numeric_cols) > 0:
                            st.markdown("**Quick Statistics:**")
                            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading CSV: {e}")
            
            # Separator between experiments
            if exp_idx < len(experiments) - 1:
                st.markdown("---")


# -------- MODEL EVALUATION RESULTS --------
elif section == "Model Evaluation Results":
    st.title("📈 Model Evaluation Results")
    st.markdown("Comprehensive performance metrics and evaluation data for all trained models")
    
    st.markdown("---")

    experiments = list_experiments()
    if not experiments:
        st.warning("No experiment folders found in `results/`.")
    else:
        # Sort experiments: EDA first, then others alphabetically
        def sort_experiments(exp_list):
            eda_results = [e for e in exp_list if 'eda' in e.lower()]
            other_results = [e for e in exp_list if 'eda' not in e.lower()]
            return eda_results + sorted(other_results)
        
        experiments = sort_experiments(experiments)
        
        # Display each experiment's evaluation results
        for exp_idx, exp_name in enumerate(experiments):
            imgs, csvs = list_files(exp_name)
            exp_dir = RESULTS_ROOT / exp_name
            
            if not csvs:
                continue  # Skip experiments with no CSV files
            
            # Experiment header
            st.header(f"🎯 {exp_name.replace('_', ' ').title()}")
            st.caption(f"📁 Location: `{exp_dir}`")
            
            # Categorize CSV files
            best_metrics = []
            training_metrics = []
            predictions = []
            other_csvs = []
            
            for csv_path in csvs:
                name_lower = csv_path.name.lower()
                if "best" in name_lower and "metric" in name_lower:
                    best_metrics.append(csv_path)
                elif any(x in name_lower for x in ["training", "csr_training", "cnn_training"]):
                    training_metrics.append(csv_path)
                elif any(x in name_lower for x in ["predict", "error", "sample"]):
                    predictions.append(csv_path)
                else:
                    other_csvs.append(csv_path)
            
            # Display Best Model Metrics (Highlighted)
            if best_metrics:
                st.subheader("⭐ Best Model Performance")
                for csv_path in best_metrics:
                    try:
                        df = pd.read_csv(csv_path)
                        
                        # Display as metrics cards if it's a summary
                        if len(df) <= 3 and len(df.columns) <= 5:
                            cols = st.columns(min(len(df.columns), 4))
                            for idx, col_name in enumerate(df.columns):
                                with cols[idx % 4]:
                                    value = df[col_name].iloc[0] if len(df) > 0 else "N/A"
                                    if isinstance(value, (int, float)):
                                        st.metric(label=col_name, value=f"{value:.4f}")
                                    else:
                                        st.metric(label=col_name, value=value)
                        
                        # Display full dataframe
                        with st.expander(f"📊 View Full Data - {csv_path.name}"):
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv_data,
                                file_name=csv_path.name,
                                mime="text/csv",
                                key=f"download_{exp_name}_{csv_path.name}"
                            )
                    except Exception as e:
                        st.error(f"Failed to read {csv_path.name}: {e}")
            
            # Display Training Metrics
            if training_metrics:
                st.subheader("📚 Training History")
                tabs = st.tabs([csv.stem.replace('_', ' ').title() for csv in training_metrics])
                for idx, csv_path in enumerate(training_metrics):
                    with tabs[idx]:
                        try:
                            df = pd.read_csv(csv_path)
                            
                            # Show summary statistics
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.dataframe(df, use_container_width=True)
                            with col2:
                                st.markdown("**Summary Statistics**")
                                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                                if len(numeric_cols) > 0:
                                    summary = df[numeric_cols].describe().loc[['mean', 'min', 'max']]
                                    st.dataframe(summary, use_container_width=True)
                            
                            # Download button
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv_data,
                                file_name=csv_path.name,
                                mime="text/csv",
                                key=f"download_{exp_name}_{csv_path.name}"
                            )
                        except Exception as e:
                            st.error(f"Failed to read {csv_path.name}: {e}")
            
            # Display Predictions and Error Analysis
            if predictions:
                st.subheader("🔍 Predictions & Error Analysis")
                for csv_path in predictions:
                    with st.expander(f"📄 {csv_path.stem.replace('_', ' ').title()}", expanded=False):
                        try:
                            df = pd.read_csv(csv_path)
                            
                            # Show first few rows and stats
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Preview** (showing {min(10, len(df))} of {len(df)} rows)")
                                st.dataframe(df.head(10), use_container_width=True)
                            with col2:
                                st.markdown("**Dataset Info**")
                                st.info(f"Total Rows: {len(df)}\nColumns: {len(df.columns)}")
                                
                                # Show error distribution if applicable
                                if 'error' in df.columns or 'absolute_error' in df.columns:
                                    error_col = 'error' if 'error' in df.columns else 'absolute_error'
                                    st.metric("Mean Error", f"{df[error_col].mean():.4f}")
                                    st.metric("Std Dev", f"{df[error_col].std():.4f}")
                            
                            # Download button
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv_data,
                                file_name=csv_path.name,
                                mime="text/csv",
                                key=f"download_{exp_name}_{csv_path.name}"
                            )
                        except Exception as e:
                            st.error(f"Failed to read {csv_path.name}: {e}")
            
            # Display Other CSV Files
            if other_csvs:
                st.subheader("📑 Additional Data Files")
                for csv_path in other_csvs:
                    with st.expander(f"📄 {csv_path.name}"):
                        try:
                            df = pd.read_csv(csv_path)
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv_data,
                                file_name=csv_path.name,
                                mime="text/csv",
                                key=f"download_{exp_name}_{csv_path.name}"
                            )
                        except Exception as e:
                            st.error(f"Failed to read {csv_path.name}: {e}")
            
            # Display Key Evaluation Plots
            if imgs:
                key_imgs = [img for img in imgs if any(k in img.name.lower() for k in 
                           ["gt_vs_pred", "error", "residual", "histogram", "confusion", "comparison"])]
                
                if key_imgs:
                    st.subheader("📊 Key Evaluation Plots")
                    cols = st.columns(2)
                    for idx, img_path in enumerate(key_imgs):
                        with cols[idx % 2]:
                            title = img_path.stem.replace('_', ' ').title()
                            st.markdown(f"**{title}**")
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            except Exception as e:
                                st.error(f"Failed to load {img_path.name}: {e}")
                            st.caption(f"_{img_path.name}_")
                            st.markdown("")
            
            # Separator between experiments
            if exp_idx < len(experiments) - 1:
                st.markdown("---")


# -------- LIVE DEMO --------
elif section == "Live Demo":
    st.title("🎥 Live Crowd Monitoring & Safety Alert System")
    st.markdown("Real-time crowd analysis with intelligent safety assessment and alert notifications")
    
    st.markdown("---")
    
    # Configuration section in sidebar
    with st.sidebar:
        st.header("⚙️ Alert Configuration")
        
        # Safety thresholds
        st.subheader("Crowd Density Thresholds")
        threshold_safe = st.slider("Safe (Green)", 0, 500, 50, 10, 
                                   help="Below this count is considered safe")
        threshold_moderate = st.slider("Moderate (Yellow)", 50, 1000, 150, 10,
                                      help="Moderate crowd density")
        threshold_high = st.slider("High Risk (Orange)", 150, 2000, 300, 10,
                                  help="High risk - monitoring required")
        threshold_critical = st.slider("Critical (Red)", 300, 5000, 500, 10,
                                      help="Critical - immediate action needed")
        
        st.markdown("---")
        st.subheader("Alert Settings")
        enable_alerts = st.checkbox("Enable Safety Alerts", value=True)
        enable_sound = st.checkbox("Enable Alert Sound", value=False, 
                                   help="Play sound on critical alerts")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Image for Analysis")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload crowd images (max ~20MB)"
        )
    
    with col2:
        st.subheader("🤖 Select Model")
        model_name = st.selectbox(
            "AI Model",
            ["CSRNet", "MobileNetCSRNet", "SimpleCNN", "RandomForest"],
            help="Choose detection algorithm"
        )
        
        model_info = {
            "CSRNet": "High accuracy, slower",
            "MobileNetCSRNet": "Balanced speed & accuracy",
            "SimpleCNN": "Fast, basic detection",
            "RandomForest": "Classical ML baseline"
        }
        st.caption(f"ℹ️ {model_info[model_name]}")
    
    if uploaded_file is not None:
        # Display uploaded image
        img_bytes = uploaded_file.read()
        
        st.markdown("---")
        st.subheader("📷 Uploaded Image")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(io.BytesIO(img_bytes), use_container_width=True)
        
        # Predict button
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_btn = st.button("🔍 Analyze Crowd Density", type="primary", use_container_width=True)
        
        if predict_btn:
            with st.spinner("🔄 Processing image... Running AI analysis..."):
                result = call_backend(api_url, model_name, img_bytes)
            
            if "error" in result:
                st.error(f"❌ Backend error: {result['error']}")
            else:
                count = result.get("predicted_count", None)
                if count is not None:
                    count = int(count)
                    
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Determine safety level
                    if count <= threshold_safe:
                        safety_level = "🟢 SAFE"
                        safety_color = "green"
                        risk_score = "Low"
                        action = "✅ No action required. Normal operations."
                    elif count <= threshold_moderate:
                        safety_level = "🟡 MODERATE"
                        safety_color = "orange"
                        risk_score = "Medium"
                        action = "⚠️ Monitor situation. Increase surveillance."
                    elif count <= threshold_high:
                        safety_level = "🟠 HIGH RISK"
                        safety_color = "orange"
                        risk_score = "High"
                        action = "⚠️ Alert security personnel. Prepare crowd control measures."
                    else:
                        safety_level = "🔴 CRITICAL"
                        safety_color = "red"
                        risk_score = "Critical"
                        action = "🚨 IMMEDIATE ACTION REQUIRED! Deploy crowd control. Consider area evacuation."
                    
                    # Display metrics in cards
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            label="Detected Crowd Count",
                            value=f"{count:,}",
                            delta=None
                        )
                    
                    with metric_col2:
                        st.metric(
                            label="Safety Status",
                            value=safety_level,
                            delta=risk_score
                        )
                    
                    with metric_col3:
                        density = count / 1000  # Normalize per 1000 sq units
                        st.metric(
                            label="Crowd Density",
                            value=f"{density:.2f}",
                            delta="per 1k units"
                        )
                    
                    # Safety Alert Box
                    if enable_alerts:
                        st.markdown("---")
                        if count > threshold_critical:
                            st.error(f"### 🚨 CRITICAL ALERT")
                            st.error(action)
                        elif count > threshold_high:
                            st.warning(f"### ⚠️ HIGH RISK ALERT")
                            st.warning(action)
                        elif count > threshold_moderate:
                            st.info(f"### 🟡 MODERATE ALERT")
                            st.info(action)
                        else:
                            st.success(f"### ✅ SAFE STATUS")
                            st.success(action)
                    
                    # Recommended Actions
                    st.markdown("---")
                    st.subheader("📋 Recommended Actions")
                    
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        st.markdown("**Immediate Steps:**")
                        if count > threshold_critical:
                            st.markdown("""
                            - 🚓 Deploy emergency response teams
                            - 📢 Activate public announcement system
                            - 🚪 Open additional exit routes
                            - 📞 Contact local law enforcement
                            """)
                        elif count > threshold_high:
                            st.markdown("""
                            - 👮 Alert on-site security personnel
                            - 📹 Increase camera monitoring
                            - 🚧 Prepare crowd barriers
                            - 📱 Ready communication channels
                            """)
                        elif count > threshold_moderate:
                            st.markdown("""
                            - 👁️ Continue active monitoring
                            - 📊 Log event for analysis
                            - 🔔 Brief security team
                            - 📈 Track crowd trends
                            """)
                        else:
                            st.markdown("""
                            - ✅ Maintain normal surveillance
                            - 📝 Standard reporting
                            - 🔄 Routine monitoring
                            - 📊 Collect baseline data
                            """)
                    
                    with action_col2:
                        st.markdown("**Prevention Measures:**")
                        if count > threshold_moderate:
                            st.markdown("""
                            - 🚦 Implement traffic control
                            - 📣 Issue crowd dispersal advisories
                            - 🎫 Consider entry restrictions
                            - 🗺️ Direct crowds to alternate areas
                            """)
                        else:
                            st.markdown("""
                            - 📍 Maintain clear signage
                            - 🚶 Ensure smooth flow
                            - 🔄 Regular checkpoints
                            - 📱 Public information updates
                            """)
                    
                    # Density Heatmap
                    st.markdown("---")
                    st.subheader("🔥 Crowd Density Heatmap")
                    
                    if result.get("heatmap_image"):
                        try:
                            heatmap_bytes = bytes.fromhex(result["heatmap_image"])
                            
                            col1, col2, col3 = st.columns([1, 3, 1])
                            with col2:
                                st.image(
                                    io.BytesIO(heatmap_bytes),
                                    caption="AI-Generated Density Map - Red indicates high concentration",
                                    use_container_width=True
                                )
                            
                            st.info("🎯 The heatmap shows crowd distribution. Red/yellow areas indicate higher density.")
                        except Exception as e:
                            st.warning(f"Could not display heatmap: {e}")
                    
                    # Technical Details
                    with st.expander("🔧 Technical Details & Raw Data"):
                        info = {
                            k: v
                            for k, v in result.items()
                            if k not in ["heatmap_image", "density_map"]
                        }
                        
                        st.markdown("**Model Information:**")
                        st.json({
                            "Model Used": model_name,
                            "Prediction Count": count,
                            "Safety Level": safety_level,
                            "Risk Assessment": risk_score
                        })
                        
                        st.markdown("**API Response:**")
                        st.json(info)
                        
                        st.markdown("**Threshold Configuration:**")
                        st.json({
                            "Safe": f"0 - {threshold_safe}",
                            "Moderate": f"{threshold_safe + 1} - {threshold_moderate}",
                            "High Risk": f"{threshold_moderate + 1} - {threshold_high}",
                            "Critical": f"{threshold_high + 1}+"
                        })
                    
                    # Export Report
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        report_data = {
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model": model_name,
                            "crowd_count": count,
                            "safety_status": safety_level,
                            "risk_score": risk_score,
                            "action_required": action
                        }
                        
                        report_json = json.dumps(report_data, indent=2)
                        st.download_button(
                            label="📄 Download Analysis Report",
                            data=report_json,
                            file_name=f"crowd_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                
                else:
                    st.warning("Backend did not return a `predicted_count` field.")
                    with st.expander("View Full Response"):
                        st.json(result)
    
    else:
        # Instructions when no image uploaded
        st.info("👆 Upload an image above to begin crowd analysis")
        
        st.markdown("---")
        st.subheader("📖 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Upload
            Upload a crowd image from your camera or device
            """)
        
        with col2:
            st.markdown("""
            ### 2️⃣ Analyze
            AI model processes the image and detects crowd density
            """)
        
        with col3:
            st.markdown("""
            ### 3️⃣ Alert
            Receive safety assessment and recommended actions
            """)
        
        st.markdown("---")
        st.info("""
        💡 **Tip:** Configure alert thresholds in the sidebar to match your venue capacity and safety requirements.
        The system will automatically categorize crowd levels and suggest appropriate actions.
        """)
