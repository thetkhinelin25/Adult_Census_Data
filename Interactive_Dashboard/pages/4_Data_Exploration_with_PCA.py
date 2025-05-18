import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import prince
import io

st.set_page_config(page_title="Clustering Dashboard", layout="wide")
st.title("\U0001F5A5\ufe0f Dimensionality Reduction and Clustering")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0  # Initial key version

if st.button("🔄 Reset All"):
    # List of session state keys used in this app
    keys_to_delete = [
        "original_df",
        "feature_df",
        "reduction_model",
        "reduced_df",
        "method",
        "kmeans_model",
        "dbscan_model",
        "clustering_scores",
        "cluster_labels",
        "selected_methods",
        "annotated_df",
        "classified_result_df",
        "new_samples_csv",  # uploader key for new samples
        "batch_methods",    # clustering method selection for batch
    ]

    # Clear everything except the uploader_key
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

    # Increment uploader_key to reset the file_uploader widgets
    st.session_state["uploader_key"] += 1

    # Rerun app to reflect cleared uploaders
    st.rerun()

# --- Step 0: Load Dataset ---
st.header("\U0001F4C2 Step 0: Load Your Dataset")
uploaded_file = st.file_uploader("Upload your imputed train dataset (.csv)", type="csv", key=f"train_file_{st.session_state['uploader_key']}")


if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    if "original_df" not in st.session_state or not st.session_state["original_df"].equals(df_new):
        st.session_state["original_df"] = df_new
        st.session_state["feature_df"] = df_new.drop(columns=["income"], errors="ignore")
        st.session_state.pop("kmeans_model", None)
        st.session_state.pop("dbscan_model", None)

if "original_df" not in st.session_state:
    st.stop()

df = st.session_state["original_df"]
df_features = st.session_state["feature_df"]

# --- Step 2: Dimensionality Reduction ---
st.header("\U0001F4CA Step 2: Choose Dimensionality Reduction Method")
method = st.selectbox("Select a dimensionality reduction method:", ["FAMD", "PCA", "MCA"])

for col in df_features.select_dtypes(include='int64').columns:
    df_features[col] = df_features[col].astype(float)

@st.cache_data(show_spinner=False)
def apply_dimensionality_reduction(method, df_features):
    if method == "FAMD":
        model = prince.FAMD(n_components=df_features.shape[1], random_state=42).fit(df_features)
        reduced = model.transform(df_features)
    elif method == "PCA":
        df_encoded = pd.get_dummies(df_features)
        df_scaled = StandardScaler().fit_transform(df_encoded)
        model = PCA(n_components=df_scaled.shape[1], random_state=42).fit(df_scaled)
        reduced = pd.DataFrame(model.transform(df_scaled))
    elif method == "MCA":
        df_cat = df_features.select_dtypes(include=["object", "category"]).dropna()
        model = prince.MCA(n_components=df_cat.shape[1], random_state=42).fit(df_cat)
        reduced = model.transform(df_cat)
    return model, reduced, method

model, reduced_df, method = apply_dimensionality_reduction(method, df_features)
st.session_state["reduction_model"] = model
st.session_state["reduced_df"] = reduced_df
st.session_state["method"] = method
st.dataframe(reduced_df.head())

# --- Step 3: Download Outputs ---
st.subheader("📥 Download Outputs")
download_options = st.multiselect(
    "Select files you want to download:",
    ["Transformed Dataset", "Feature Contributions to PCs", "Eigenvalues Summary"]
)

if "Transformed Dataset" in download_options:
    st.download_button(
        "Download Transformed Dataset",
        data=reduced_df.to_csv(index=False).encode("utf-8"),
        file_name="reduced_data.csv",
        mime="text/csv"
    )

if "Feature Contributions to PCs" in download_options:
    try:
        if method == "FAMD":
            contributions = model.column_coordinates_

        elif method == "MCA":
            contributions = model.column_coordinates(df)

        elif method == "PCA":
            contributions = pd.DataFrame(model.components_.T, index=pd.get_dummies(df_features).columns)

        st.download_button(
            label="Download Feature Contributions to PCs",
            data=contributions.to_csv().encode("utf-8"),
            file_name="feature_contributions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Unable to extract feature contributions: {e}")



if "Eigenvalues Summary" in download_options:
    if method in ["FAMD", "MCA"]:
        eigenvalues = model.eigenvalues_summary
    elif method == "PCA":
        eigenvalues = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(model.explained_variance_ratio_))],
            "Explained Variance Ratio": model.explained_variance_ratio_
        })
    st.download_button(
        "Download Eigenvalues Summary",
        data=eigenvalues.to_csv().encode("utf-8"),
        file_name="eigenvalues_summary.csv",
        mime="text/csv"
    )


# --- Step 4: Clustering Method Selection ---
st.header("\U0001F52C Step 3: Choose Clustering Method(s)")
selected_methods = st.multiselect("Select clustering method(s) to evaluate:", ["KMeans", "DBSCAN"])
params = {}

if selected_methods:
    if len(selected_methods) == 1:
        method = selected_methods[0]
        if method == "KMeans":
            st.subheader("🎯 KMeans Clustering Settings")
            k_selection_method = st.radio("Select KMeans configuration method:", ["Manual", "Automatic (Silhouette Score)"])
            if k_selection_method == "Manual":
                k_value = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=3)
                params["KMeans"] = {"k_mode": "manual", "k": k_value}
            else:
                params["KMeans"] = {"k_mode": "auto"}

        elif method == "DBSCAN":
            st.subheader("🎯 DBSCAN Clustering Settings")
            dbscan_setting = st.radio("Choose DBSCAN configuration:", ["Default", "Custom"])
            if dbscan_setting == "Custom":
                eps_val = st.slider("Select eps (Neighborhood radius):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples_val = st.slider("Select min_samples (Minimum points):", min_value=1, max_value=20, value=5)
                params["DBSCAN"] = {"eps": eps_val, "min_samples": min_samples_val}
            else:
                params["DBSCAN"] = {"eps": 0.5, "min_samples": 5}

    elif len(selected_methods) == 2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 KMeans Clustering Settings")
            k_selection_method = st.radio("Select KMeans configuration method:", ["Manual", "Automatic (Silhouette Score)"], key="kmeans_mode")
            if k_selection_method == "Manual":
                k_value = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=3, key="kmeans_k")
                params["KMeans"] = {"k_mode": "manual", "k": k_value}
            else:
                params["KMeans"] = {"k_mode": "auto"}

        with col2:
            st.subheader("🎯 DBSCAN Clustering Settings")
            dbscan_setting = st.radio("Choose DBSCAN configuration:", ["Default", "Custom"], key="dbscan_mode")
            if dbscan_setting == "Custom":
                eps_val = st.slider("Select eps (Neighborhood radius):", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="eps")
                min_samples_val = st.slider("Select min_samples (Minimum points):", min_value=1, max_value=20, value=5, key="min_samples")
                params["DBSCAN"] = {"eps": eps_val, "min_samples": min_samples_val}
            else:
                params["DBSCAN"] = {"eps": 0.5, "min_samples": 5}

    if st.button("Run Clustering"):
        scores = {}
        labels_dict = {}

        if "KMeans" in selected_methods:
            if params["KMeans"]["k_mode"] == "manual":
                st.session_state["kmeans_model"] = KMeans(n_clusters=params["KMeans"]["k"], random_state=42)
                labels_kmeans = st.session_state["kmeans_model"].fit_predict(reduced_df)
                scores["KMeans"] = silhouette_score(reduced_df, labels_kmeans)
                labels_dict["KMeans"] = labels_kmeans
            else:
                silhouette_scores = {}
                for k in range(2, 11):
                    temp_model = KMeans(n_clusters=k, random_state=42)
                    labels = temp_model.fit_predict(reduced_df)
                    silhouette_scores[k] = silhouette_score(reduced_df, labels)
                best_k = max(silhouette_scores, key=silhouette_scores.get)
                st.success(f"Best k determined by silhouette score: {best_k}")
                st.session_state["kmeans_model"] = KMeans(n_clusters=best_k, random_state=42)
                labels_kmeans = st.session_state["kmeans_model"].fit_predict(reduced_df)
                scores["KMeans"] = silhouette_scores[best_k]
                labels_dict["KMeans"] = labels_kmeans

        if "DBSCAN" in selected_methods:
            st.session_state["dbscan_model"] = DBSCAN(**params["DBSCAN"])
            labels_dbscan = st.session_state["dbscan_model"].fit_predict(reduced_df)
            scores["DBSCAN"] = silhouette_score(reduced_df, labels_dbscan) if len(set(labels_dbscan)) > 1 else -1
            labels_dict["DBSCAN"] = labels_dbscan

        st.session_state["clustering_scores"] = scores
        st.session_state["cluster_labels"] = labels_dict
        st.session_state["selected_methods"] = selected_methods

# --- Show Clustering Results if Available ---
if all(key in st.session_state for key in ["clustering_scores", "cluster_labels", "selected_methods"]):
    scores = st.session_state["clustering_scores"]
    labels_dict = st.session_state["cluster_labels"]
    selected_methods = st.session_state["selected_methods"]

    st.subheader("\U0001F4C8 Silhouette Scores")
    for m, score in scores.items():
        st.write(f"{m}'s Silhouette Score: {score:.3f}")

    st.markdown("""
        #### 🧠 **Average Silhouette Score (for the whole dataset)**:
        - **> 0.7** → Strong structure: clusters are well separated
        - **0.5–0.7** → Reasonable structure: some overlapping
        - **0.25–0.5** → Weak structure: clusters may be artificial
        - **< 0.25** → Very weak clustering; consider rethinking number of clusters or algorithm
        """)

    st.subheader("\U0001F3A8 Cluster Plot on Principal Components")

    for m in selected_methods:
        # Prepare plotting DataFrame
        df_plot = reduced_df.copy()
        df_plot["Cluster"] = labels_dict[m].astype(str)  # Ensure string labels for color mapping
        df_plot["Index"] = df_plot.index  # Add index for hover info

        # Create Plotly scatter plot
        fig = px.scatter(
            df_plot,
            x=0,
            y=1,
            color="Cluster",  # Use column name (string)
            hover_data=["Index"],
            labels={0: "PC1", 1: "PC2"},
            color_discrete_sequence=px.colors.qualitative.Set1,  # Optional: consistent color palette
            title=f"{m}: PC1 vs PC2"
        )

        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Convert figure to PNG using Kaleido
        img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
        img_buffer = io.BytesIO(img_bytes)

        # Download button
        st.download_button(
            label=f"📥 Download {m} Plot as PNG",
            data=img_buffer,
            file_name=f"{m}_cluster_plot.png",
            mime="image/png"
        )

    # --- Cluster-wise Summary Statistics ---
    st.header("\U0001F4CB Cluster-wise Summary Statistics")

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for m in selected_methods:
        df_temp = df.copy()
        df_temp["Cluster"] = labels_dict[m]

        with st.expander(f"📊 Summary for {m}", expanded=False):
            tab1, tab2 = st.tabs(["🔢 Numeric Summary", "🔣 Categorical Proportions"])

            with tab1:
                st.dataframe(df_temp.groupby("Cluster")[numeric_cols].mean().round(3))

            with tab2:
                for col in cat_cols:
                    st.write(f"### {col}")
                    cat_summary = df_temp.groupby("Cluster")[col].value_counts(normalize=True).unstack().round(2)
                    st.dataframe(cat_summary)


    # --- Annotate Comment ---
    st.header("📝 Annotate Multiple Data Points Individually")

    # Step 1: Select indexes to annotate
    selected_indices = st.multiselect("Select index numbers to annotate", options=df.index.tolist())

    # Step 2: Show input boxes for each selected index
    annotated_comments = {}
    if selected_indices:
        st.subheader("✏️ Enter Comments for Each Selected Index")
        for idx in selected_indices:
            comment = st.text_input(f"Comment for Index {idx}", key=f"comment_{idx}")
            annotated_comments[idx] = comment

    # Step 3: Apply comments
    if st.button("Save Comments"):
        df_annotated = df.copy()
        df_annotated["Comment"] = ""
        for idx, comment in annotated_comments.items():
            if idx in df_annotated.index:
                df_annotated.at[idx, "Comment"] = comment
        st.session_state["annotated_df"] = df_annotated
        st.success("✅ Comments saved successfully.")

    # Step 4: Display and download
    if "annotated_df" in st.session_state:
        st.subheader("📋 Annotated Dataset Preview")
        st.dataframe(st.session_state["annotated_df"])

        csv_annotated = st.session_state["annotated_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Annotated Dataset",
            data=csv_annotated,
            file_name="annotated_dataset.csv",
            mime="text/csv"
        )


    # --- New Sample Prediction ---
    st.header("\U0001F195 Add and Analyze New Sample(s)")

    tab1, tab2 = st.tabs(["➕ Enter Manually", "📁 Upload CSV"])

    # Manual Entry (Single Sample)
    with tab1:
        if st.checkbox("Add a single new sample"):
            new_sample = {}
            st.subheader("\U0001F4CB Enter Feature Values")
            input_cols = st.columns(2)
            for i, col in enumerate(df_features.columns):
                with input_cols[i % 2]:
                    if df_features[col].dtype == 'object':
                        new_sample[col] = st.selectbox(f"{col}", df_features[col].unique(), key=f"input_{col}")
                    else:
                        new_sample[col] = st.number_input(f"{col}", value=float(df_features[col].mean()), key=f"input_{col}")

            chosen_methods = st.multiselect("Select clustering method(s) to classify the new sample:", [m for m in ["KMeans", "DBSCAN"] if m in selected_methods])

            if st.button("Classify New Sample"):
                new_df = pd.DataFrame([new_sample])

                # Transform sample
                if st.session_state["method"] == "FAMD":
                    transformed_sample = st.session_state["reduction_model"].transform(new_df)
                elif st.session_state["method"] == "PCA":
                    encoded_sample = pd.get_dummies(new_df)
                    aligned = pd.DataFrame(columns=st.session_state["reduction_model"].feature_names_in_)
                    for col in aligned.columns:
                        aligned[col] = encoded_sample[col] if col in encoded_sample.columns else 0
                    scaled = StandardScaler().fit_transform(aligned)
                    transformed_sample = st.session_state["reduction_model"].transform(scaled)
                elif st.session_state["method"] == "MCA":
                    transformed_sample = st.session_state["reduction_model"].transform(new_df)

                for m in chosen_methods:
                    if m == "KMeans" and "kmeans_model" in st.session_state:
                        pred = st.session_state["kmeans_model"].predict(transformed_sample)
                        st.success(f"\u2B50 New sample (KMeans) belongs to cluster: {pred[0]}")
                    elif m == "DBSCAN" and "dbscan_model" in st.session_state:
                        all_data = pd.concat([reduced_df, transformed_sample])
                        labels = st.session_state["dbscan_model"].fit_predict(all_data)
                        st.success(f"\u2B50 New sample (DBSCAN) belongs to cluster: {labels[-1]}")
                    else:
                        st.warning(f"Please run {m} clustering first.")

    # Batch Upload (Multiple Samples)
    with tab2:
        uploaded_new_samples = st.file_uploader("Upload new samples (.csv)", type="csv", key="new_samples_csv")
        chosen_methods_csv = st.multiselect(
            "Select clustering method(s) to classify uploaded samples:",
            [m for m in ["KMeans", "DBSCAN"] if m in selected_methods],
            key="batch_methods"
        )

        if uploaded_new_samples and st.button("Classify Uploaded Samples"):
            new_batch_df = pd.read_csv(uploaded_new_samples)

            # Transform
            if st.session_state["method"] == "FAMD":
                transformed_batch = st.session_state["reduction_model"].transform(new_batch_df)
            elif st.session_state["method"] == "PCA":
                encoded_batch = pd.get_dummies(new_batch_df)
                aligned = pd.DataFrame(columns=st.session_state["reduction_model"].feature_names_in_)
                for col in aligned.columns:
                    aligned[col] = encoded_batch[col] if col in encoded_batch.columns else 0
                scaled = StandardScaler().fit_transform(aligned)
                transformed_batch = st.session_state["reduction_model"].transform(scaled)
            elif st.session_state["method"] == "MCA":
                transformed_batch = st.session_state["reduction_model"].transform(new_batch_df)

            result_df = new_batch_df.copy()

            # Apply clustering
            for m in chosen_methods_csv:
                if m == "KMeans" and "kmeans_model" in st.session_state:
                    preds = st.session_state["kmeans_model"].predict(transformed_batch)
                    result_df[f"Cluster_{m}"] = preds
                elif m == "DBSCAN" and "dbscan_model" in st.session_state:
                    all_data = pd.concat([reduced_df, transformed_batch])
                    labels = st.session_state["dbscan_model"].fit_predict(all_data)
                    result_df[f"Cluster_{m}"] = labels[-len(new_batch_df):]
                else:
                    st.warning(f"Please run {m} clustering first.")

            # Save to session state
            st.session_state["classified_result_df"] = result_df
            st.success("✅ Classification complete.")

        # Always display the dataframe if available
        if "classified_result_df" in st.session_state:
            st.dataframe(st.session_state["classified_result_df"])

            csv = st.session_state["classified_result_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Classified Results",
                data=csv,
                file_name="classified_new_samples.csv",
                mime="text/csv"
            )
