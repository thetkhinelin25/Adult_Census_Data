import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Model Training & Prediction", layout="wide")
st.title("🤖 Model Training and Prediction Dashboard")

# --- Mode Selection ---
mode = st.sidebar.selectbox("Select Mode", [
    "Train and Predict",
    "Make prediction with a Provided Model",
    "Make prediction with Built in model"
])

# ---------------------------
# Part 1: Train and Predict
# ---------------------------
if mode == "Train and Predict":
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0  # Initial key version

    if st.button("🔄 Reset All"):
        # List of keys used in this app (Train & Predict mode only)
        keys_to_delete = [
            "model_train_df", "model_test_df",
            "trained_model", "scaler", "label_enc",
            "numeric_cols", "raw_feature_cols", "train_columns", "category_options",
            "train_file", "test_file"
        ]

        # Dynamically include prediction input fields
        keys_to_delete += [
            key for key in st.session_state.keys()
            if key.startswith(("new_input_", "upload_num_", "upload_cat_", "built_num_", "built_cat_"))
        ]

        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]

        # Reset uploader keys
        st.session_state["uploader_key"] += 1
        st.rerun()


    # --- 1. Upload datasets ---
    st.header("📥 Step 1: Load Train and Test Datasets")

    train_file = st.file_uploader("Upload Training Set CSV", type="csv", key=f"train_file_{st.session_state['uploader_key']}")
    test_file = st.file_uploader("Upload Test Set CSV", type="csv", key=f"test_file_{st.session_state['uploader_key']}")


    if train_file:
        train_df = pd.read_csv(train_file)
        st.session_state["model_train_df"] = train_df.copy()
    elif "model_train_df" in st.session_state:
        train_df = st.session_state["model_train_df"]
    else:
        st.warning("Please upload the training dataset.")
        st.stop()

    if test_file:
        test_df = pd.read_csv(test_file)
        st.session_state["model_test_df"] = test_df.copy()
    elif "model_test_df" in st.session_state:
        test_df = st.session_state["model_test_df"]
    else:
        st.warning("Please upload the test dataset.")
        st.stop()

    # --- Encode target ---
    label_enc = LabelEncoder()
    train_df["income"] = label_enc.fit_transform(train_df["income"])
    test_df["income"] = label_enc.transform(test_df["income"])

    # Identify column types
    numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).drop("income", axis=1).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

    raw_feature_cols = numeric_cols + categorical_cols

    # Save category options for each categorical column
    category_options = {col: sorted(train_df[col].dropna().unique().tolist()) for col in categorical_cols}

    # One-hot encode
    train_df = pd.get_dummies(train_df, columns=categorical_cols)
    test_df = pd.get_dummies(test_df, columns=categorical_cols)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    # Split X and y
    X_train = train_df.drop("income", axis=1)
    y_train = train_df["income"]
    X_test = test_df.drop("income", axis=1)
    y_test = test_df["income"]

    # Scale numeric columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Store in session_state
    st.session_state["scaler"] = scaler
    st.session_state["label_enc"] = label_enc
    st.session_state["numeric_cols"] = numeric_cols
    st.session_state["raw_feature_cols"] = raw_feature_cols
    st.session_state["train_columns"] = X_train.columns
    st.session_state["category_options"] = category_options

    # --- Model Selection ---
    st.header("⚙️ Step 2: Select Model & Hyperparameters")
    model_choice = st.selectbox("Choose a model to train:", ["Gradient Boosting", "AdaBoost", "Random Forest", "ANN"])

    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report

    param_grid = {}
    if model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": st.multiselect("n_estimators", [100, 200, 300], default=[200]),
            "learning_rate": st.multiselect("learning_rate", [0.01, 0.05, 0.1], default=[0.1]),
            "max_depth": st.multiselect("max_depth", [3, 5, 7], default=[5]),
            "min_samples_split": st.multiselect("min_samples_split", [2, 5, 10], default=[2]),
            "min_samples_leaf": st.multiselect("min_samples_leaf", [1, 2, 5], default=[5]),
        }
    elif model_choice == "AdaBoost":
        model = AdaBoostClassifier(random_state=42)
        param_grid = {
            "n_estimators": st.multiselect("n_estimators", [50, 100, 200], default=[100]),
            "learning_rate": st.multiselect("learning_rate", [0.01, 0.1, 1], default=[1]),
        }
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": st.multiselect("n_estimators", [100, 200, 300], default=[200]),
            "max_depth": st.multiselect("max_depth", [None, 10, 20], default=[None]),
            "min_samples_split": st.multiselect("min_samples_split", [2, 5, 10], default=[2]),
            "min_samples_leaf": st.multiselect("min_samples_leaf", [1, 2, 5], default=[1]),
        }
    elif model_choice == "ANN":
        model = MLPClassifier(random_state=42, max_iter=500)
        param_grid = {
            "hidden_layer_sizes": st.multiselect("hidden_layer_sizes", [(50,), (100,), (50, 50)], default=[(50, 50)]),
            "activation": st.multiselect("activation", ["relu", "tanh", "logistic"], default=["relu"]),
            "alpha": st.multiselect("alpha", [0.0001, 0.001], default=[0.0001]),
        }

    if st.button("🚀 Train Model"):
        with st.spinner("Training model with GridSearchCV..."):
            grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)
            st.success("✅ Model trained successfully!")

            # Save model to session and disk
            st.session_state["trained_model"] = grid
            joblib.dump(grid.best_estimator_, "trained_model.pkl")
            st.download_button(
                label="📥 Download Trained Model",
                data=open("trained_model.pkl", "rb").read(),
                file_name="trained_model.pkl",
                mime="application/octet-stream",
            )

            # Predict and evaluate
            preds = grid.predict(X_test)
            st.subheader("📊 Classification Report")
            report = classification_report(y_test, preds, output_dict=False)
            st.text(report)

if "trained_model" in st.session_state:
    st.subheader("🔎 Predict New Samples")
    st.markdown("Enter values for each feature (not one-hot encoded). Fill according to the correct data type.")
    new_sample = {}
    cols = st.columns(4)
    for i, col in enumerate(st.session_state["raw_feature_cols"]):
        with cols[i % 4]:
            dtype = "Numeric" if col in st.session_state["numeric_cols"] else "Categorical"
            if dtype == "Numeric":
                new_sample[col] = st.text_input(f"{col} ({dtype})", key=f"new_input_{col}")
            else:
                options = st.session_state["category_options"].get(col, [])
                new_sample[col] = st.selectbox(f"{col} ({dtype})", options, key=f"new_input_{col}")

    if st.button("🔍 Predict"):
        try:
            input_df = pd.DataFrame([new_sample])
            for col in st.session_state["numeric_cols"]:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=st.session_state["train_columns"], fill_value=0)
            input_df[st.session_state["numeric_cols"]] = st.session_state["scaler"].transform(input_df[st.session_state["numeric_cols"]])
            pred = st.session_state["trained_model"].predict(input_df)
            decoded = st.session_state["label_enc"].inverse_transform(pred)
            st.write("Prediction:", decoded[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------
# Part 2: Predict with Uploaded Model
# ---------------------------
elif mode == "Make prediction with a Provided Model":
    st.header("📦 Upload a Saved Model Bundle")

    st.info(
        "**Reminder:** Please upload a model bundle (`.pkl`) that includes the following keys:\n\n"
        "- `model` or `model_path`\n"
        "- `scaler`\n"
        "- `label_encoder`\n"
        "- `numeric_cols`\n"
        "- `raw_categorical_cols`\n"
        "- `all_columns`\n"
        "- `category_options`"
    )

    model_file = st.file_uploader("Upload your model_bundle.pkl", type="pkl")

    if model_file:
        # Load the model bundle
        bundle = joblib.load(model_file)

        # Determine model type: sklearn or ANN
        if "model" in bundle:
            model = bundle["model"]
            model_type = "sklearn"
        elif "model_path" in bundle:
            model_path = bundle["model_path"]
            if not os.path.exists(model_path):
                st.error(f"❌ ANN model file not found at: {model_path}")
                st.stop()
            model = tf.keras.models.load_model(model_path)
            model_type = "ann"
        else:
            st.error("❌ Invalid model bundle: no model or model_path found.")
            st.stop()

        # Extract preprocessing details
        scaler = bundle.get("scaler")
        label_enc = bundle.get("label_encoder")
        numeric_cols = bundle.get("numeric_cols", [])
        categorical_cols = bundle.get("raw_categorical_cols", [])
        all_columns = bundle.get("all_columns", [])
        category_options = bundle.get("category_options", {})

        # Input UI
        st.success("✅ Model loaded successfully. Provide feature values for prediction.")
        user_input = {}
        cols = st.columns(4)
        for i, col in enumerate(numeric_cols + categorical_cols):
            with cols[i % 4]:
                if col in numeric_cols:
                    user_input[col] = st.text_input(f"{col} (Numeric)", key=f"upload_num_{col}")
                else:
                    options = category_options.get(col, [])
                    if options:
                        user_input[col] = st.selectbox(f"{col} (Categorical)", options, key=f"upload_cat_{col}")
                    else:
                        user_input[col] = st.text_input(f"{col} (Categorical)", key=f"upload_cat_{col}")

        # Prediction
        if st.button("🔍 Predict", key="uploaded_predict"):
            try:
                input_df = pd.DataFrame([user_input])
                for col in numeric_cols:
                    input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=all_columns, fill_value=0)
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

                if model_type == "ann":
                    input_array = input_df.to_numpy().astype(np.float32)
                    preds = (model.predict(input_array) > 0.5).astype(int).flatten()
                else:
                    preds = model.predict(input_df)

                decoded = label_enc.inverse_transform(preds)
                st.write("Prediction:", decoded[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ---------------------------
# Part 3: Predict with Built-in Model
# ---------------------------
elif mode == "Make prediction with Built in model":
    st.header("🏗️ Select Built-in Model")
    model_name = st.selectbox("Choose Model", ["GBM", "AdaBoost", "Random Forest", "ANN"])

    model_paths = {
        "GBM": "models/gbm_model_bundle.pkl",
        "AdaBoost": "models/adaboost_model_bundle.pkl",
        "Random Forest": "models/random_forest_model_bundle.pkl",
        "ANN": "models/ann_model_bundle.pkl",
    }

    import os
    bundle = joblib.load(model_paths[model_name])

    if model_name == "ANN":
        ann_path = bundle.get("model_path")
        if not os.path.exists(ann_path):
            st.error(f"❌ ANN model file not found at: {ann_path}")
            st.stop()
        model = tf.keras.models.load_model(ann_path)
        model_type = "ann"
    else:
        model = bundle["model"]
        model_type = "sklearn"

    scaler = bundle["scaler"]
    label_enc = bundle["label_encoder"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["raw_categorical_cols"]
    all_columns = bundle["all_columns"]
    category_options = bundle.get("category_options", {})

    st.success(f"✅ {model_name} model loaded.")
    st.markdown("### 🔢 Enter Feature Values")

    user_input = {}
    cols = st.columns(4)
    for i, col in enumerate(numeric_cols + categorical_cols):
        with cols[i % 4]:
            if col in numeric_cols:
                user_input[col] = st.text_input(f"{col} (Numeric)", key=f"built_num_{col}")
            else:
                options = category_options.get(col, [])
                if options:
                    user_input[col] = st.selectbox(f"{col} (Categorical)", options, key=f"built_cat_{col}")
                else:
                    user_input[col] = st.text_input(f"{col} (Categorical)", key=f"built_cat_{col}")

    if st.button("🔍 Predict", key="builtin_predict"):
        try:
            input_df = pd.DataFrame([user_input])
            for col in numeric_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=all_columns, fill_value=0)
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            if model_type == "ann":
                input_array = input_df.to_numpy().astype(np.float32)
                preds = (model.predict(input_array) > 0.5).astype(int).flatten()
            else:
                preds = model.predict(input_df)

            decoded = label_enc.inverse_transform(preds)
            st.write("Prediction:", decoded[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
