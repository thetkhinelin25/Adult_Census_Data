import streamlit as st
import pandas as pd

st.set_page_config(page_title="Home", layout="wide")
st.title("🤖 Welcome to the Smart Data Analysis App!")

st.markdown("""
This application supports **any tabular dataset**, not just the Adult Census dataset.
Whether you're preparing data, exploring insights, clustering patterns, or training models, 
this tool offers an end-to-end solution — **no coding required**.
""")

st.header("🔍 Core Features")

# Missing Data
st.subheader("📉 Missing Data Exploration & Imputation")
st.markdown("""
- Detect and visualize missing values
- Apply various imputation techniques directly within the app
""")

# Data Exploration
st.subheader("📊 Interactive Data Exploration")
st.markdown("""
- Filter datasets dynamically and export the filtered subset  
- Visualize relationships and correlations between up to two features  
- Generate AI-powered insights to guide your analysis  
- Save plots and documentation for reporting purposes
""")

# Dimensionality & Clustering
st.subheader("🧬 Dimensionality Reduction & Clustering")
st.markdown("""
- Apply dimensionality reduction using **PCA**, **MCA**, or **FAMD**  
- Perform clustering using **KMeans** or **DBSCAN**  
- View summaries of numerical and categorical features by cluster  
- Annotate data points using row indices or custom IDs  
- Predict cluster memberships for new samples
""")

# Model Development
st.subheader("🤖 Model Development & Prediction")
st.markdown("""
- Train models directly in the app:
  - **Gradient Boosting Machines (GBM)**  
  - **AdaBoost**  
  - **Random Forest**  
  - **Artificial Neural Networks (ANN)**  
- Upload pretrained models (`.pkl` or `.h5`) to make predictions  
- Use built-in pretrained models for instant predictions  
- Input new data points for live model inference
""")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0  # Initial key version

if st.button("🔄 Reset All"):
    # Keys used in the missing data handling app
    keys_to_delete = ["uploaded_df"]

    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

    # Increment uploader_key to reset the file_uploader widgets
    st.session_state["uploader_key"] += 1

    # Rerun app to reflect cleared uploaders
    st.rerun()

# --- Upload Section ---
st.subheader("📥 Upload Your Dataset")
st.info("Please upload a CSV file to preview your dataset and check summary statistic.")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key=f"train_file_{st.session_state['uploader_key']}")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["uploaded_df"] = df
elif "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
else:
    st.stop()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📊 Summary Statistics")
st.write(df.describe(include='all'))

if st.checkbox("📐 Show dataset shape and column types"):
    st.write(df.shape)
    st.write(df.dtypes)
    
