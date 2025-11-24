import streamlit as st
import pandas as pd
# import numpy as np
# text utilities
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
st.latex(""" 
    E = mc^2
    """)
# display utilities
df = pd.DataFrame({
    "Metric": ["Revenue", "Expenses", "Profit"],
    "Value": [100000, 80000, 20000]
})

st.dataframe(df)

st.metric(label="Revenue", value="$100,000", delta="-$5,000 grater than last month")

st.json({
    "name": "Start-up Dashboard",
    "version": "1.0",
    "description": "A dashboard for start-up companies to visualize their data.",
    "author": {
        "name": "Your Name",
        "email": "your.email@example.com",
        "license": "MIT"
    }
})

# media utilities
st.image('deepVision_crowd_monitor-main\Cod\sampleDashboard\photo.jpg', caption='Start-up Team')
st.video('deepVision_crowd_monitor-main\Cod\sampleDashboard\video.mp4')

# creating layouts
st.sidebar.title("Navigation")
st.sidebar.header("Go to")

c1, c2 = st.columns(2)
with c1:
    st.image('deepVision_crowd_monitor-main\Cod\sampleDashboard\photo.jpg', caption='Start-up Team')
with c2:
    st.video('deepVision_crowd_monitor-main\Cod\sampleDashboard\video.mp4')


#showing progress
import time

my_bar = st.progress(0)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Start "):
        for i in range(100):
            time.sleep(0.02)
            my_bar.progress(i)
        st.success("Process Completed Successfully ")


with col2:
    if st.button("Stop "):
        st.error("Process Stopped — Something went wrong!")


with col3:
    if st.button("Warn"):
        st.warning("Process might take longer than expected")
        for i in range(100):
            time.sleep(0.06)
            my_bar.progress(i)
        with col1:
            st.success("Long Process Completed")


with col4:
    if st.button("Info ℹ"):
        info_placeholder = st.empty()
        info_placeholder.info("Info: Progress will start on start button click")
        time.sleep(1)
        info_placeholder.empty()


# taking inputs
name = st.text_input("Enter your name", key="name")
age = st.number_input("Enter your age", min_value=0, max_value=120, key="age")
dob = st.date_input("Enter your date of birth", key="dob")
password = st.text_input("Enter your password", type="password", key="password")

btn = st.button("Submit")
if btn:
    success_placeholder = st.empty()
    if name and age and dob and password:
        # st.balloons()
        success_placeholder.success(f"Hello {name}, your age is {age} and you were born on {dob}.")
        time.sleep(3)
        success_placeholder.empty()
        
    else:
        success_placeholder.error("Please fill all the fields.")
        time.sleep(1)
        success_placeholder.empty()



# dropdowns
language = st.selectbox("Select your favorite programming language",
             ["Python", "JavaScript", "Java", "C++", "Ruby"], key="language")
st.write(f"You selected: {language}")


# multi-select
frameworks = st.multiselect("Select the frameworks you have experience with",
             ["Django", "Flask", "React", "Angular", "Vue"], key="frameworks")
st.write(f"You selected: {frameworks}")

# sliders
rating = st.slider("Rate your experience with Streamlit", 0, 10, key="rating")
st.write(f"You rated: {rating}")

# checkboxes
agree = st.checkbox("I agree to the terms and conditions", key="agree")
st.write(f"You {'agreed' if agree else 'did not agree'} to the terms and conditions.")

# radio buttons
gender = st.radio("Select your gender", ["Male", "Female", "Other"], key="gender")
st.write(f"You selected: {gender}")

#file uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="file_uploader")
if uploaded_file is not None:
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.describe())
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.describe())


uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="file_uploader")
if uploaded_file is not None:
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.describe())
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.describe())