import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from io import StringIO


st.set_page_config(page_title="Missing Data Handling", layout="wide")
st.title("🧹 Detect & Handle Missing Values")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0  # Initial key version

if st.button("🔄 Reset All"):
    # Keys used in the missing data handling app
    keys_to_delete = [
        "train_df", "test_df",
        "original_train_df",
        "train_df_drop_1", "test_df_drop_1",
        "train_df_drop_2", "test_df_drop_2",
        "train_file", "test_file",
    ]

    # Also clear dynamic widget keys (e.g. from file_uploader or imputation options)
    keys_to_delete += [key for key in st.session_state.keys() if key.startswith((
        "cramers_", "eta_", "evaluate_button", "download_", "plot_", "range_", "cat_", "file_uploader", "impute_"
    ))]

    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

    # Increment uploader_key to reset the file_uploader widgets
    st.session_state["uploader_key"] += 1

    # Rerun app to reflect cleared uploaders
    st.rerun()


# --- 1. Upload datasets ---
st.header("📥 Step 1: Load Train and Test Datasets")

train_file = st.file_uploader("Upload Training Set CSV", type="csv", key=f"train_file_{st.session_state['uploader_key']}")
test_file = st.file_uploader("Upload Test Set CSV", type="csv", key=f"test_file_{st.session_state['uploader_key']}")

if train_file:
    train_df = pd.read_csv(train_file)
    st.session_state["train_df"] = train_df.copy()
    if "original_train_df" not in st.session_state:
        st.session_state["original_train_df"] = train_df.dropna().copy()
elif "train_df" in st.session_state:
    train_df = st.session_state["train_df"]
else:
    st.warning("Please upload the training dataset.")
    st.stop()

if test_file:
    test_df = pd.read_csv(test_file)
    st.session_state["test_df"] = test_df.copy()
elif "test_df" in st.session_state:
    test_df = st.session_state["test_df"]
else:
    st.warning("Please upload the test dataset.")
    st.stop()

# --- 2. Show basic missing info ---
st.header("🔍 Step 2: Check Missing Values")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Train Set Missing Values")
    st.dataframe(train_df.isnull().sum().to_frame("Missing Values").query("`Missing Values` > 0"))
with col2:
    st.subheader("Test Set Missing Values")
    st.dataframe(test_df.isnull().sum().to_frame("Missing Values").query("`Missing Values` > 0"))

# --- 3a. Nullity matrix and missing correlation heatmap ---
st.header("📊 Step 3: Explore Missing Data Patterns")

st.subheader("Nullity Matrix (Train Set)")
fig, ax = plt.subplots(figsize=(12, 3))
msno.matrix(train_df, ax=ax)
st.pyplot(fig)

st.subheader("Missingness Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
msno.heatmap(train_df, ax=ax)
st.pyplot(fig)

# --- 3b. Sorted Nullity Matrix by Missing Columns ---
st.subheader("🔍 Sorted Nullity Matrix (Train Set)")

# Identify columns with missing values
missing_cols = train_df.columns[train_df.isnull().any()].tolist()

if missing_cols:
    sort_by_col = st.selectbox("Sort matrix by column (with missing values):", missing_cols)
    fig, ax = plt.subplots(figsize=(12, 3))
    msno.matrix(train_df.sort_values(by=sort_by_col), ax=ax)
    st.pyplot(fig)

    st.markdown(f"""
    Matrix sorted by `{sort_by_col}` to help identify patterns of missingness.
    Now you may choose to **drop rows where column has missing values at random**.
    """)

    drop_target_col = st.selectbox("Select column to drop rows where it's missing:", missing_cols)

    if "train_df_drop_1" not in st.session_state:
        st.session_state["train_df_drop_1"] = st.session_state["train_df"]
        
    if "test_df_drop_1" not in st.session_state:
        st.session_state["test_df_drop_1"] = st.session_state["test_df"]

    if st.button(f"❌ Drop rows with missing values in `{drop_target_col}`"):
        rows_before_train = st.session_state["train_df"].shape[0]
        rows_before_test = st.session_state["test_df"].shape[0]

        st.session_state["train_df_drop_1"] = st.session_state["train_df"].dropna(subset=[drop_target_col])
        st.session_state["test_df_drop_1"] = st.session_state["test_df"].dropna(subset=[drop_target_col])

        rows_after_train = st.session_state["train_df_drop_1"].shape[0]
        rows_after_test = st.session_state["test_df_drop_1"].shape[0]

        st.success(
            f"Dropped {rows_before_train - rows_after_train} rows from train set "
            f"and {rows_before_test - rows_after_test} rows from test set."
        )

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Train Set Missing Values")
            st.dataframe(st.session_state["train_df_drop_1"].isnull().sum().to_frame("Missing Values").query("`Missing Values` > 0"))
        with col4:
            st.subheader("Test Set Missing Values")
            st.dataframe(st.session_state["test_df_drop_1"].isnull().sum().to_frame("Missing Values").query("`Missing Values` > 0"))
        
else:
    st.info("✅ No columns with missing values in train set.")

# --- 4. Handle Missing Data ---
st.header("🧠 Step 4: Handle Missing Values")

impute_method = st.radio("Choose an imputation method:",
                        ["Drop rows with NaNs",
                        "Simple Imputation (Mean/Median/Mode)",
                        "Advanced Imputation (KNN)"])

if impute_method == "Drop rows with NaNs":
    st.warning("⚠️ This will drop all rows with at least one missing value.")
    if st.button("Drop Rows (Train & Test)"):
        st.session_state["train_df_drop_2"] = st.session_state["train_df_drop_1"].dropna()
        st.session_state["test_df_drop_2"] = st.session_state["test_df_drop_1"].dropna()
        st.success(f"Dropped NaNs. Train: {st.session_state['train_df'].shape}, Test: {st.session_state['test_df'].shape}")
    else:
        st.session_state["train_df_drop_2"] = st.session_state["train_df_drop_1"]
        st.session_state["test_df_drop_2"] = st.session_state["test_df_drop_1"]

elif impute_method == "Simple Imputation (Mean/Median/Mode)":
    user_choice = st.selectbox("Choose strategy for numerical columns", ["Mean", "Median"])
    
    if st.button("Apply Simple Imputation"):
        df_train_imp = st.session_state["train_df_drop_1"]
        df_test_imp = st.session_state["test_df_drop_1"]
        for col in df_train_imp.columns:
            if df_train_imp[col].isnull().sum() == 0 and df_test_imp[col].isnull().sum() == 0:
                continue
            if df_train_imp[col].dtype in ["int64", "float64"]:
                if user_choice == "Mean":
                    fill_val_train = df_train_imp[col].mean()
                    fill_val_test = df_test_imp[col].mean()
                else:
                    fill_val_train = df_train_imp[col].median()
                    fill_val_test = df_test_imp[col].median()
            else:
                fill_val_train = df_train_imp[col].mode()[0]
                fill_val_test = df_test_imp[col].mode()[0]

            df_train_imp[col] = df_train_imp[col].fillna(fill_val_train)
            df_test_imp[col] = df_test_imp[col].fillna(fill_val_test)

        st.session_state["train_df_drop_2"] = df_train_imp
        st.session_state["test_df_drop_2"] = df_test_imp
        st.success("Simple imputation applied to both datasets.")

    else:
        st.session_state["train_df_drop_2"] = st.session_state["train_df_drop_1"]
        st.session_state["test_df_drop_2"] = st.session_state["test_df_drop_1"]

elif impute_method == "Advanced Imputation (KNN)":
    st.info("KNN imputation will encode all columns, impute, and decode. This may take some time.")

    if st.button("Apply KNN Imputation"):
        from sklearn.preprocessing import OrdinalEncoder
        from fancyimpute import KNN
        import numpy as np

        combined_df = pd.concat([st.session_state["train_df_drop_1"], st.session_state["test_df_drop_1"]], ignore_index=True)
        ordinal_enc_dict = {}

        for col in combined_df:
            ordinal_enc_dict[col] = OrdinalEncoder()
            col_data = combined_df[col]
            not_null = col_data[col_data.notnull()].values.reshape(-1, 1)
            encoded = ordinal_enc_dict[col].fit_transform(not_null)
            combined_df.loc[col_data.notnull(), col] = np.squeeze(encoded)

        combined_df_imputed = combined_df.copy(deep=True)
        knn_imputer = KNN()
        combined_df_imputed.iloc[:, :] = np.round(knn_imputer.fit_transform(combined_df))

        for col in combined_df_imputed:
            reshaped_vals = combined_df_imputed[col].values.reshape(-1, 1)
            combined_df_imputed[col] = ordinal_enc_dict[col].inverse_transform(reshaped_vals).ravel()

        st.session_state["train_df_drop_2"] = combined_df_imputed.iloc[:len(st.session_state["train_df_drop_1"])].copy()
        st.session_state["test_df_drop_2"] = combined_df_imputed.iloc[len(st.session_state["test_df_drop_1"]):].copy()

        st.success("KNN Imputation completed.")
    
    else:
        st.session_state["train_df_drop_2"] = st.session_state["train_df_drop_1"]
        st.session_state["test_df_drop_2"] = st.session_state["test_df_drop_1"]


# --- 5. Check Remaining Missing Values ---
st.header("📋 Step 5: Post-Imputation Check")
st.subheader("Train Set Missing Values")
st.dataframe(st.session_state["train_df_drop_2"].isnull().sum().to_frame("Missing Values").query("`Missing Values` > 0"))

st.subheader("Test Set Missing Values")
st.dataframe(st.session_state["test_df_drop_2"].isnull().sum().to_frame("Missing Values").query("`Missing Values` > 0"))


# --- 6. Evaluate Imputation Impact ---
st.header("📈 Step 6: Evaluate Imputation Impact")

st.markdown("Compare distributions and model performance between **original** and **imputed** data.")

# --- Distribution Comparison (Before vs After Imputation) ---
st.subheader("🔍 Compare Distributions Before vs After Imputation")

all_cols = st.session_state["train_df_drop_2"].columns.tolist()
selected_cols = st.multiselect("Select columns to compare:", all_cols, default=all_cols[:2])

def annotate_bar(ax):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10)

if selected_cols:
    n = len(selected_cols)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))  # 2 columns: before & after
    if n == 1:
        axes = [axes]  # Ensure 2D structure if only one row

    for i, col in enumerate(selected_cols):
        ax1, ax2 = axes[i]
        dtype = st.session_state["train_df_drop_2"][col].dtype

        if dtype == 'object':
            # Categorical bar plot
            st.session_state["original_train_df"][col].value_counts().plot(
                kind='bar', ax=ax1, title=f"{col} (Original - Missing Drop)")
            annotate_bar(ax1)
            ax1.set_ylabel("Count")

            st.session_state["train_df_drop_2"][col].value_counts().plot(
                kind='bar', ax=ax2, title=f"{col} (After Imputation)")
            annotate_bar(ax2)
            ax2.set_ylabel("Count")

        else:
            # Numeric histogram
            st.session_state["original_train_df"][col].dropna().plot(
                kind='hist', ax=ax1, bins=30, alpha=0.7, title=f"{col} (Original - Missing Drop)")
            ax1.set_ylabel("Frequency")

            st.session_state["train_df_drop_2"][col].dropna().plot(
                kind='hist', ax=ax2, bins=30, alpha=0.7, title=f"{col} (After Imputation)")
            ax2.set_ylabel("Frequency")

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Select at least one column to display comparison.")

# --- Cramér's V and Eta Squared ---
st.subheader("📊 Variable Strength (Cramér's V & Eta²)")

cat_cols = st.session_state["train_df_drop_2"].select_dtypes(include='object').columns.tolist()
num_cols = st.session_state["train_df_drop_2"].select_dtypes(include=['int64', 'float64']).columns.tolist()

cols_to_check = st.multiselect("Select target categorical columns to analyze:", cat_cols, default=cat_cols[:2])

from scipy.stats import chi2_contingency, kruskal
import numpy as np

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

# --- After Imputation ---
cramers_results_after = []
eta_squared_results_after = []

for target in cols_to_check:
    for col in cat_cols:
        if col != target:
            v = cramers_v(st.session_state["train_df_drop_2"][target], st.session_state["train_df_drop_2"][col])
            cramers_results_after.append({'Target': target, 'Compared With': col, "Cramér's V": round(v, 4)})

    for col in num_cols:
        groups = [st.session_state["train_df_drop_2"][st.session_state["train_df_drop_2"][target] == cat][col] for cat in st.session_state["train_df_drop_2"][target].dropna().unique()]
        if all(len(g) > 1 for g in groups):
            h_stat, _ = kruskal(*groups)
            k = len(groups)
            n = sum(len(g) for g in groups)
            eta_sq = (h_stat - k + 1) / (n - k)
            eta_squared_results_after.append({'Target': target, 'Compared With': col, 'Eta Squared': round(eta_sq, 4)})

cramers_after_df = pd.DataFrame(cramers_results_after).sort_values(by=["Target", "Cramér's V"], ascending=[True, False])
eta_after_df = pd.DataFrame(eta_squared_results_after).sort_values(by=["Target", "Eta Squared"], ascending=[True, False])

# --- Before Imputation (Original Complete Dataset) ---
cramers_results_before = []
eta_squared_results_before = []

original_df = st.session_state["original_train_df"]
cat_cols_orig = original_df.select_dtypes(include='object').columns.tolist()
num_cols_orig = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for target in cols_to_check:
    for col in cat_cols_orig:
        if col != target:
            v = cramers_v(original_df[target], original_df[col])
            cramers_results_before.append({'Target': target, 'Compared With': col, "Cramér's V": round(v, 4)})

    for col in num_cols_orig:
        groups = [original_df[original_df[target] == cat][col] for cat in original_df[target].dropna().unique()]
        if all(len(g) > 1 for g in groups):
            h_stat, _ = kruskal(*groups)
            k = len(groups)
            n = sum(len(g) for g in groups)
            eta_sq = (h_stat - k + 1) / (n - k)
            eta_squared_results_before.append({'Target': target, 'Compared With': col, 'Eta Squared': round(eta_sq, 4)})

cramers_before_df = pd.DataFrame(cramers_results_before).sort_values(by=["Target", "Cramér's V"], ascending=[True, False])
eta_before_df = pd.DataFrame(eta_squared_results_before).sort_values(by=["Target", "Eta Squared"], ascending=[True, False])

# --- Display Side-by-Side Results ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔗 Cramér's V (Before Imputation)")
    st.dataframe(cramers_before_df)

with col2:
    st.markdown("### 🔗 Cramér's V (After Imputation)")
    st.dataframe(cramers_after_df)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 📏 Eta Squared (Before Imputation)")
    st.dataframe(eta_before_df)

with col4:
    st.markdown("### 📏 Eta Squared (After Imputation)")
    st.dataframe(eta_after_df)


# --- Random Forest Performance Comparison ---
st.subheader("🌲 Model Performance: Random Forest (Before vs After Imputation)")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def build_and_evaluate_rf_model(df, label_column='income'):
    df = df.copy()
    if label_column not in df.columns:
        return None

    # Drop rows with missing label just in case
    df = df.dropna(subset=[label_column])

    # Split features and target
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Identify column types
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Pipeline: preprocessing + classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit model
    model.fit(X_train, y_train)

    # Predict & score
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Run evaluation for both datasets
    # --- Trigger evaluation only when user clicks ---
if st.button("🚀 Evaluate Random Forest Model on Before and After Datasets"):
    # Run evaluation for both datasets
    accuracy_before = build_and_evaluate_rf_model(st.session_state["original_train_df"])
    accuracy_after = build_and_evaluate_rf_model(st.session_state["train_df_drop_2"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy (Before Imputation)", f"{accuracy_before:.4f}")
    with col2:
        st.metric("Accuracy (After Imputation)", f"{accuracy_after:.4f}")


# --- 7. Save and Download Imputed Data ---
st.header("💾 Step 7: Download Final Imputed Datasets")

st.markdown("Download the latest versions of the imputed **train** and **test** datasets.")

# Ensure train and test exist in session state
if "train_df_drop_2" in st.session_state and "test_df_drop_2" in st.session_state:
    # Convert DataFrames to CSV format in memory
    train_csv = st.session_state["train_df_drop_2"].to_csv(index=False).encode("utf-8")
    test_csv = st.session_state["test_df_drop_2"].to_csv(index=False).encode("utf-8")

    # Download buttons
    st.download_button(
        label="📥 Download Imputed Train Set",
        data=train_csv,
        file_name="imputed_train.csv",
        mime="text/csv"
    )

    st.download_button(
        label="📥 Download Imputed Test Set",
        data=test_csv,
        file_name="imputed_test.csv",
        mime="text/csv"
    )
else:
    st.warning("No imputed data found in session state.")

