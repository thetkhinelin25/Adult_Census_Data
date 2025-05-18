import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import io

st.set_page_config(page_title="Data Exploration", layout="wide")
st.title("\U0001F4CA Data Exploration with AI Insights")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0  # Initial key version

if st.button("🔄 Reset All"):
    keys_to_delete = [
        "exploration_df",
        "filtered_df",
        "active_blocks",
        "block_counter",
        "exploration_data",
    ]

    # Also dynamically remove keys like "features_0", "range_0", "cat_0_1", etc.
    keys_to_delete += [key for key in st.session_state.keys() if key.startswith(("features_", "range_", "cat_", "plot_", "download_filtered_df_block_", "ai_button_", "delete_block_", "add_block_"))]

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

# --- Helper functions ---
def get_dtype(col):
    return df[col].dtype

def suggest_plot_types(selected):
    if len(selected) == 1:
        dtype = get_dtype(selected[0])
        return ["Histogram", "KDE", "Box"] if pd.api.types.is_numeric_dtype(dtype) else ["Bar", "Pie"]
    elif len(selected) == 2:
        col1, col2 = selected
        dtype1, dtype2 = get_dtype(col1), get_dtype(col2)
        if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
            return ["Scatter", "Hexbin", "2D Histogram"]
        elif pd.api.types.is_numeric_dtype(dtype1) != pd.api.types.is_numeric_dtype(dtype2):
            return ["Box", "Violin", "Strip", "Histogram (Grouped)"]
        else:
            return ["Grouped Bar"]
    return []

def plot_selected_features(plot_choice, selected_features, filtered_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = selected_features[0]
    y = selected_features[1] if len(selected_features) == 2 else None
    try:
        if plot_choice == "Histogram":
            sns.histplot(filtered_df[x], kde=True, bins=30, ax=ax)
        elif plot_choice == "KDE":
            sns.kdeplot(filtered_df[x], ax=ax)
        elif plot_choice == "Box":
            if y:
                sns.boxplot(x=y, y=x, data=filtered_df, ax=ax)
            else:
                sns.boxplot(x=filtered_df[x], ax=ax)
        elif plot_choice == "Bar":
            filtered_df[x].value_counts().plot(kind="bar", ax=ax)
        elif plot_choice == "Pie":
            filtered_df[x].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
        elif plot_choice == "Scatter" and y:
            sns.scatterplot(data=filtered_df, x=x, y=y, ax=ax)
        elif plot_choice == "Hexbin" and y:
            ax.hexbin(filtered_df[x], filtered_df[y], gridsize=30, cmap='Blues')
        elif plot_choice == "2D Histogram" and y:
            h = ax.hist2d(filtered_df[x], filtered_df[y], bins=30, cmap="Blues")
            plt.colorbar(h[3], ax=ax)
        elif plot_choice == "Violin" and y:
            sns.violinplot(x=x, y=y, data=filtered_df, ax=ax)
        elif plot_choice == "Strip" and y:
            sns.stripplot(x=x, y=y, data=filtered_df, ax=ax)
        elif plot_choice == "Grouped Bar" and y:
            ctab = pd.crosstab(filtered_df[x], filtered_df[y])
            if not ctab.empty:
                ctab.plot(kind="bar", stacked=True, ax=ax)
            else:
                st.warning("No data to display for this combination after filtering.")
        elif plot_choice == "Histogram (Grouped)" and y:
            num_col = x if pd.api.types.is_numeric_dtype(df[x]) else y
            cat_col = y if num_col == x else x
            for group in filtered_df[cat_col].dropna().unique():
                sns.histplot(filtered_df[filtered_df[cat_col] == group][num_col], kde=True, bins=30, label=str(group), element="step", ax=ax)
            ax.legend(title=cat_col)
        ax.set_title(f"{plot_choice} of {' vs '.join(selected_features)}")
        st.pyplot(fig)

        # --- Add download option ---
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        st.download_button(
            label="📥 Download Plot as PNG",
            data=buf,
            file_name=f"{plot_choice.replace(' ', '_').lower()}_plot.png",
            mime="image/png",
            key=f"download_{plot_choice}_{'_'.join(selected_features)}"  # ✅ Prevent ID conflict
        )

    except Exception as e:
        st.error(f"Error generating plot: {e}")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["exploration_df"] = df.copy()
    st.success("✅ Dataset loaded successfully.")
elif "exploration_df" in st.session_state:
    df = st.session_state["exploration_df"]
else:
    st.warning("Please upload the dataset.")
    st.stop()


# --- Initialize session state ---
if "active_blocks" not in st.session_state:
    st.session_state["active_blocks"] = [0]
if "block_counter" not in st.session_state:
    st.session_state["block_counter"] = 1
if "exploration_data" not in st.session_state:
    st.session_state["exploration_data"] = {}

with st.expander("🔍 Filter Your Dataset"):
    df = st.session_state["exploration_df"]

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Step 1: Choose which features to filter
    selected_features = st.multiselect("Select features to filter by:", df.columns.tolist())

    filters = {}

    # Step 2: For each selected feature, define a filter condition
    for feature in selected_features:
        if feature in numeric_cols:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            selected_range = st.slider(
                f"Select range for {feature}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                step=(max_val - min_val) / 100
            )
            filters[feature] = (df[feature] >= selected_range[0]) & (df[feature] <= selected_range[1])

        elif feature in categorical_cols:
            unique_values = df[feature].dropna().unique().tolist()
            selected_categories = st.multiselect(f"Select values for {feature}", unique_values, default=unique_values)
            filters[feature] = df[feature].isin(selected_categories)

    # Step 3: Apply filters
    if selected_features:
        combined_filter = pd.Series(True, index=df.index)
        for condition in filters.values():
            combined_filter &= condition

        # Keep only selected rows and columns
        filtered_df = df.loc[combined_filter, selected_features]
        st.session_state["filtered_df"] = filtered_df

        original_rows = df.shape[0]
        filtered_rows = filtered_df.shape[0]
        original_cols = df.shape[1]
        filtered_cols = filtered_df.shape[1]

        # Step 4: Display results
        if filtered_rows < original_rows or filtered_cols < original_cols:
            st.success(f"📊 Filtered dataset: {filtered_rows:,} rows and {len(selected_features):,} columns (from {original_rows} rows and {original_cols} columns).")
            st.dataframe(filtered_df)

            # Step 5: Download button
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv",
                key="download_filtered"
            )
        else:
            st.warning("⚠️ Filter did not reduce both rows and columns.")
            st.dataframe(df[selected_features])
    else:
        filters = {}
        st.info("ℹ️ Select one or more features to begin filtering.")


# --- Exploration Blocks ---
for i, block_id in enumerate(st.session_state["active_blocks"]):
    block_key = f"Exploration Block {block_id}"
    features = df.columns.tolist()

    with st.expander(f"\U0001F50D {block_key}", expanded=(block_id == 0)):
        # --- Selected Features ---
        if f"features_{i}" not in st.session_state:
            st.session_state[f"features_{i}"] = st.session_state["exploration_data"].get(block_key, {}).get("selected_features", [])

        selected_features = st.multiselect(
            f"Select 1 or 2 features ({block_key})",
            features,
            key=f"features_{i}",
            max_selections=2
        )

        exploration_state = st.session_state["exploration_data"].setdefault(block_key, {})
        exploration_state["selected_features"] = selected_features

        if not selected_features:
            continue

        # --- Filtering ---
        filters = {}
        exploration_state = st.session_state["exploration_data"].setdefault(block_key, {})

        if len(selected_features) == 1:
            col = selected_features[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                if f"range_{i}" not in st.session_state:
                    st.session_state[f"range_{i}"] = exploration_state.get(f"{col}_range", (min_val, max_val))
                selected_range = st.slider(
                    f"Select range for {col}",
                    min_val,
                    max_val,
                    key=f"range_{i}"
                )
                exploration_state[f"{col}_range"] = selected_range
                filters[col] = (df[col] >= selected_range[0]) & (df[col] <= selected_range[1])
            else:
                categories = df[col].dropna().unique().tolist()
                if f"cat_{i}" not in st.session_state:
                    st.session_state[f"cat_{i}"] = exploration_state.get(f"{col}_categories", categories)
                selected_categories = st.multiselect(
                    f"Select categories for {col}",
                    categories,
                    key=f"cat_{i}"
                )
                exploration_state[f"{col}_categories"] = selected_categories
                filters[col] = df[col].isin(selected_categories)

        elif len(selected_features) == 2:
            col1, col2 = selected_features
            col_a, col_b = st.columns(2)
            with col_a:
                if pd.api.types.is_numeric_dtype(df[col1]):
                    min_val, max_val = float(df[col1].min()), float(df[col1].max())
                    if f"range_{i}_0" not in st.session_state:
                        st.session_state[f"range_{i}_0"] = exploration_state.get(f"{col1}_range", (min_val, max_val))
                    selected_range = st.slider(
                        f"Select range for {col1}",
                        min_val,
                        max_val,
                        key=f"range_{i}_0"
                    )
                    exploration_state[f"{col1}_range"] = selected_range
                    filters[col1] = (df[col1] >= selected_range[0]) & (df[col1] <= selected_range[1])
                else:
                    categories = df[col1].dropna().unique().tolist()
                    if f"cat_{i}_0" not in st.session_state:
                        st.session_state[f"cat_{i}_0"] = exploration_state.get(f"{col1}_categories", categories)
                    selected_categories = st.multiselect(
                        f"Select categories for {col1}",
                        categories,
                        key=f"cat_{i}_0"
                    )
                    exploration_state[f"{col1}_categories"] = selected_categories
                    filters[col1] = df[col1].isin(selected_categories)
            with col_b:
                if pd.api.types.is_numeric_dtype(df[col2]):
                    min_val, max_val = float(df[col2].min()), float(df[col2].max())
                    if f"range_{i}_1" not in st.session_state:
                        st.session_state[f"range_{i}_1"] = exploration_state.get(f"{col2}_range", (min_val, max_val))
                    selected_range = st.slider(
                        f"Select range for {col2}",
                        min_val,
                        max_val,
                        key=f"range_{i}_1"
                    )
                    exploration_state[f"{col2}_range"] = selected_range
                    filters[col2] = (df[col2] >= selected_range[0]) & (df[col2] <= selected_range[1])
                else:
                    categories = df[col2].dropna().unique().tolist()
                    if f"cat_{i}_1" not in st.session_state:
                        st.session_state[f"cat_{i}_1"] = exploration_state.get(f"{col2}_categories", categories)
                    selected_categories = st.multiselect(
                        f"Select categories for {col2}",
                        categories,
                        key=f"cat_{i}_1"
                    )
                    exploration_state[f"{col2}_categories"] = selected_categories
                    filters[col2] = df[col2].isin(selected_categories)

        # --- Filter data ---
        combined_filter = pd.Series(True, index=df.index)
        for cond in filters.values():
            combined_filter &= cond

        filtered_df = df[combined_filter]
        exploration_state["filtered_df"] = filtered_df

        # --- Download filtered DataFrame ---
        csv = exploration_state["filtered_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv",
            key=f"download_filtered_df_block_{i}"
        )

        # --- Plotting ---
        plot_options = suggest_plot_types(selected_features)
        prev_choice = exploration_state.get("plot_choice", plot_options[0] if plot_options else None)
        plot_choice = st.selectbox(
            f"Choose a plot type ({block_key})",
            plot_options,
            index=plot_options.index(prev_choice) if prev_choice in plot_options else 0,
            key=f"plot_{i}"
        ) if plot_options else None
        exploration_state["plot_choice"] = plot_choice

        plot_selected_features(plot_choice, selected_features, filtered_df)

        if st.button(f"\U0001F9E0 Generate or Refresh AI Insight for {block_key}", key=f"ai_button_{i}"):
            base_url = "https://api.aimlapi.com/v1"
            api_key = st.secrets["AIML_API_KEY"]
            client = OpenAI(api_key=api_key, base_url=base_url)
            feature_text = " vs ".join(selected_features)
            prompt = (
                f"You are a data analyst. You will analyze Adult Census Data. "
                f"The user generated a {plot_choice} plot of {feature_text}. "
                f"Please describe key insights from the plot, including any patterns, trends, outliers, or relationships. "
                f"Make it clear and understandable for a non-technical audience."
            )
            with st.spinner("Asking AI for insights..."):
                response = client.chat.completions.create(
                    model="mistralai/Mistral-7B-Instruct-v0.2",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                ai_message = response.choices[0].message.content
                exploration_state["ai_insight"] = ai_message

        if exploration_state.get("ai_insight"):
            st.markdown("### \U0001F9E0 AI Insight:")
            st.info(exploration_state["ai_insight"])

        if st.button(f"➕ Add Another Exploration After {block_key}", key=f"add_block_{i}"):
            new_id = st.session_state["block_counter"]
            st.session_state["active_blocks"].append(new_id)
            st.session_state["block_counter"] += 1
            st.rerun()

        if st.button(f"\U0001F5D1 Delete {block_key}", key=f"delete_block_{i}"):
            st.session_state["active_blocks"].remove(block_id)
            st.session_state["exploration_data"].pop(block_key, None)
            st.session_state.pop(f"features_{i}", None)
            st.rerun()
