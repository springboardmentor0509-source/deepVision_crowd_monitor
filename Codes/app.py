import streamlit as st
import cv2
import os
import tempfile
import time

# Page Setup
st.set_page_config(page_title="DeepVision Monitor", layout="wide")

st.title("👁️ DeepVision Crowd Monitor System")
st.markdown("Real-time crowd analysis dashboard")

# Sidebar
st.sidebar.header("Control Panel")
option = st.sidebar.selectbox("Choose Video Source", ["Sample Video", "Upload New Video"])

video_path = None

# Logic to handle file paths
if option == "Sample Video":
    # Since app.py and video.mp4 are in the SAME folder
    if os.path.exists("video.mp4"):
        video_path = "video.mp4"
        st.sidebar.success("Loaded local sample video.")
    else:
        st.sidebar.error("video.mp4 not found in sampleDashboard folder!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["mp4", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

# Main Display
col1, col2 = st.columns([2, 1])

if video_path:
    with col1:
        st.subheader("Live Video Feed")
        st_image = st.empty()

    with col2:
        st.subheader("Analytics")
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.markdown("**Status**")
            status_kpi = st.empty()
        with kpi2:
            st.markdown("**People Count**")
            count_kpi = st.empty()

        st.markdown("---")
        st.write("Detection Log:")
        log_text = st.empty()

    # Video Processing Loop
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Restart video or stop

        # Simulate Model Prediction (Replace this with real model later)
        # For now, we just generate a fake number to show the UI works
        fake_count = int(abs(frame.mean() / 3)) 

        # Update UI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_image.image(frame, channels="RGB", use_column_width=True)

        status_kpi.success("Active")
        count_kpi.metric(label="Count", value=fake_count)

        # Small sleep to match video speed roughly
        time.sleep(0.03)

    cap.release()
else:
    st.info("👈 Please select a video source from the sidebar.")