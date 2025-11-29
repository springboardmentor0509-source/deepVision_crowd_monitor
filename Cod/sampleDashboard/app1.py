import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import time

# ------------------------------
# Header / Intro
# ------------------------------
st.set_page_config(page_title="Start-up Dashboard", layout="wide")
st.title("Start-up Dashboard")
st.header("Welcome to the Start-up Dashboard")
st.subheader("Your one-stop solution for managing start-up data")
st.write("This dashboard provides insights and analytics for start-up companies.")

st.markdown("""
### Features:
- Real-time data visualization  
- Customizable dashboards  
- Collaboration tools for teams  
""")

st.code("""
import streamlit as st
            
st.title("Start-up Dashboard")
st.header("Welcome to the Start-up Dashboard")
st.subheader("Your one-stop solution for managing start-up data")
st.write("This dashboard provides insights and analytics for start-up companies.")
""")

st.write("E = mc^2")
st.latex("E = mc^2")

# ------------------------------
# Simple DataFrame + Metrics
# ------------------------------
df_static = pd.DataFrame({
    "Metric": ["Revenue", "Expenses", "Profit"],
    "Value": [100000, 80000, 20000]
})
st.dataframe(df_static)

st.metric(label="Revenue", value="$100,000", delta="-$5,000 greater than last month")

st.json({
    "name": "Start-up Dashboard",
    "version": "1.0",
    "description": "A dashboard for start-up companies to visualize their data.",
    "author": {"name": "Your Name", "email": "your.email@example.com", "license": "MIT"}
})

# ------------------------------
# Asset paths (robust)
# ------------------------------
HERE = Path(__file__).parent
IMG = HERE / "photo.jpg"
VID = HERE / "video.mp4"

# Show image (if present) or friendly error
if IMG.exists():
    try:
        img = Image.open(IMG)
        st.image(img, caption="Start-up Team")
    except Exception as e:
        st.error(f"Failed to open image {IMG}: {e}")
else:
    st.info(f"(Tip) Put 'photo.jpg' in the same folder as this script to display the team image. Expected path: {IMG}")

# Show video (if present)
if VID.exists():
    try:
        st.video(str(VID))
    except Exception as e:
        st.warning(f"Failed to play video {VID}: {e}")
else:
    st.info(f"(Tip) Put 'video.mp4' in the same folder as this script to show a demo video. Expected path: {VID}")

# ------------------------------
# Layout: image + video side-by-side
# ------------------------------
c1, c2 = st.columns(2)
with c1:
    if IMG.exists():
        st.image(str(IMG), caption="Team (left column)")
with c2:
    if VID.exists():
        st.video(str(VID))

# ------------------------------
# Progress bar controls
# ------------------------------
st.write("### Progress Control")
my_bar = st.progress(0)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Start"):
        for i in range(101):
            time.sleep(0.02)
            my_bar.progress(i)
        st.success("Process Completed Successfully")

with col2:
    if st.button("Stop"):
        st.error("Process Stopped — Something went wrong!")

with col3:
    if st.button("Warn"):
        st.warning("Process might take longer than expected")
        for i in range(101):
            time.sleep(0.03)
            my_bar.progress(i)
        st.success("Long Process Completed")

with col4:
    if st.button("Info ℹ"):
        info = st.empty()
        info.info("Info: Progress will start when you click Start")
        time.sleep(1)
        info.empty()

# ------------------------------
# Input form
# ------------------------------
st.write("### User Information Form")
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=120)
dob = st.date_input("Enter your date of birth")
password = st.text_input("Enter your password", type="password")

if st.button("Submit Info"):
    if name and age and dob and password:
        placeholder = st.empty()
        placeholder.success(f"Hello {name}, your age is {age} and you were born on {dob}.")
        time.sleep(3)
        placeholder.empty()
    else:
        st.error("Please fill all the fields.")

# ------------------------------
# Selectors
# ------------------------------
language = st.selectbox("Favorite programming language:", ["Python", "JavaScript", "Java", "C++", "Ruby"])
st.write(f"You selected: {language}")

frameworks = st.multiselect("Frameworks you use:", ["Django", "Flask", "React", "Angular", "Vue"])
st.write(f"You selected: {frameworks}")

rating = st.slider("Rate your experience with Streamlit:", 0, 10)
st.write(f"You rated: {rating}/10")

agree = st.checkbox("I agree to the terms and conditions")
st.write("Status:", "Agreed" if agree else "Not agreed")

gender = st.radio("Select gender:", ["Male", "Female", "Other"])
st.write("You selected:", gender)

# ------------------------------
# SINGLE File uploader (unique key)
# ------------------------------
st.write("### Upload a File (CSV or XLSX)")

uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"], key="main_uploader")
if uploaded_file is not None:
    # show file metadata
    file_details = {
        "Filename": uploaded_file.name,
        "FileType": uploaded_file.type,
        "FileSize": uploaded_file.size
    }
    st.write(file_details)

    # make sure to reset pointer if reused
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    # read file robustly
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)
        st.dataframe(df_uploaded)
        st.write(df_uploaded.describe(include="all"))
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
