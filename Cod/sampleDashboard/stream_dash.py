# app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import os
import io
import traceback

# ---------------- Page config ----------------
st.set_page_config(page_title="DeepVision Explorer", layout="wide")

# ---------------- Session state & Navigation ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "selected_chart" not in st.session_state:
    st.session_state.selected_chart = None

with st.sidebar:
    st.title("Navigation")
    if st.button("Home", key="nav_home"):
        st.session_state.page = "Home"
    if st.button("EDA", key="nav_eda"):
        st.session_state.page = "EDA"
    if st.button("Upload", key="nav_upload"):
        st.session_state.page = "Upload"
    if st.button("Settings", key="nav_settings"):
        st.session_state.page = "Settings"

page = st.session_state.page

# ---------------- Path resolution ----------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
charts_dir = PROJECT_ROOT / "charts"
data_dir = PROJECT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# ---------------- Helpers ----------------
def list_chart_files(folder: Path, ext=".png"):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ext and p.is_file()])

def image_to_bytes(img_path: Path):
    buf = io.BytesIO()
    Image.open(img_path).save(buf, format="PNG")
    buf.seek(0)
    return buf

def dataframe_to_csv_bytes(df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("DeepVision — Explorer")
    st.header("AI DeepVision — Crowd Monitor")
    st.write(
        "AI DeepVision — Crowd Monitor is an experimental project for crowd density estimation "
        "and dataset exploration before model training."
    )

    st.subheader("Key advantages")
    st.write("- Lightweight and easy to run locally\n- Supports rapid prototyping\n- Clear visual insights")

    st.subheader("Quick links")
    st.markdown(
        "- Go to **EDA** to preview generated charts\n"
        "- Go to **Upload** to upload CSV/XLSX and inspect data\n"
        "- Go to **Settings** to inspect paths and folders"
    )

# ---------------- EDA PAGE ----------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis (Charts)")

    if not charts_dir.exists():
        st.error(f"Charts directory not found at: `{charts_dir}`. Generate charts there or update path in the code.")
    else:
        chart_files = list_chart_files(charts_dir, ".png")
        if not chart_files:
            st.warning(f"No PNG charts found in: {charts_dir}")
        else:
            st.success(f"Found {len(chart_files)} chart(s) in: {charts_dir}")

            with st.sidebar.expander("Chart Controls", expanded=True):
                pattern = st.text_input("Filter charts by name (substring)", key="chart_filter")
                size_choice = st.selectbox("Display size", ["Small", "Medium", "Large"], index=1, key="chart_size")

            filtered = [p for p in chart_files if pattern.lower() in p.name.lower()]

            if not filtered:
                st.warning("No charts match your filter.")
            else:
                cols = st.columns(2)
                for idx, p in enumerate(filtered):
                    col = cols[idx % 2]
                    col.subheader(p.stem.replace("_", " ").title())
                    try:
                        # show thumbnail using container width (no deprecated param)
                        col.image(str(p), use_container_width=True)
                    except Exception as e:
                        col.error(f"Failed to load {p.name}: {e}")

                    # unique view button per file
                    if col.button(f"View → {p.name}", key=f"view_{p.name}"):
                        st.session_state.selected_chart = str(p)

                # show large view if selected
                sel = st.session_state.get("selected_chart")
                if sel:
                    sel_path = Path(sel)
                    if sel_path.exists():
                        st.markdown("---")
                        st.subheader(f"Selected chart: {sel_path.name}")
                        try:
                            img = Image.open(sel_path)
                            if size_choice == "Small":
                                st.image(img, width=400)
                            elif size_choice == "Medium":
                                st.image(img, width=700)
                            else:
                                st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to display selected chart: {e}")
                        else:
                            # download button
                            img_bytes = image_to_bytes(sel_path)
                            st.download_button(
                                label="Download chart (PNG)",
                                data=img_bytes,
                                file_name=sel_path.name,
                                mime="image/png",
                                key=f"download_{sel_path.name}"
                            )
                    else:
                        st.error("Selected chart file not found. It may have been moved or deleted.")
                        st.session_state.selected_chart = None

# ---------------- UPLOAD PAGE ----------------
elif page == "Upload":
    st.title("📥 Upload & Inspect Data")
    st.write("Upload a CSV or XLSX file. Uploaded files are saved to the `data/` folder.")

    uploaded = st.file_uploader("Choose CSV or XLSX", type=["csv", "xlsx"], key="uploader_main")
    if uploaded is not None:
        try:
            save_path = data_dir / uploaded.name
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved uploaded file to: {save_path}")
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}\n{traceback.format_exc()}")

        # Read and display preview + stats
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(save_path)
            else:
                df = pd.read_excel(save_path)
            st.subheader("Preview (first 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)
            st.subheader("Basic statistics")
            try:
                st.write(df.describe(include="all"))
            except Exception:
                st.write("Could not compute describe for mixed types.")
            # download processed csv
            csv_bytes = dataframe_to_csv_bytes(df)
            st.download_button("Download as CSV", data=csv_bytes, file_name=f"{uploaded.name}.csv", mime="text/csv", key="download_uploaded")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}\n{traceback.format_exc()}")

    # show existing uploaded files and allow loading them
    existing = sorted(list(data_dir.iterdir())) if data_dir.exists() else []
    if existing:
        st.write("### Previously uploaded files")
        for f in existing:
            st.write(f"- {f.name} ({f.stat().st_size} bytes)")
            if st.button(f"Load {f.name}", key=f"load_{f.name}"):
                try:
                    if f.suffix.lower() == ".csv":
                        df_load = pd.read_csv(f)
                    else:
                        df_load = pd.read_excel(f)
                    st.dataframe(df_load.head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load {f.name}: {e}")

# ---------------- SETTINGS PAGE ----------------
elif page == "Settings":
    st.title("⚙️ Settings & Paths")
    st.write("Paths used by this app (edit code if needed):")
    st.write(f"- `HERE` (app folder): `{HERE}`")
    st.write(f"- `PROJECT_ROOT`: `{PROJECT_ROOT}`")
    st.write(f"- `charts_dir`: `{charts_dir}`")
    st.write(f"- `data_dir`: `{data_dir}`")

    if st.button("Show charts folder contents", key="show_charts"):
        if charts_dir.exists():
            items = sorted(charts_dir.iterdir())
            if not items:
                st.info("Charts folder is empty.")
            else:
                for it in items:
                    st.write(it.name)
        else:
            st.warning("Charts folder does not exist.")

# ---------------- fallback ----------------
else:
    st.error("Unknown page selected. Resetting to Home.")
    st.session_state.page = "Home"
