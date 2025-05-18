# 🧠 Adult Census Income Prediction Project

This project is a part of my MSc Data Science coursework and focuses on analyzing the **Adult Census Income** dataset from the UCI Machine Learning Repository. The objective is to develop predictive models that determine whether an individual's income exceeds $50K per year, based on demographic and employment-related attributes.

## 📁 Dataset

The dataset consists of various features such as age, work class, education, occupation, hours worked per week, and more. The target variable is `income`, classified as `<=50K` or `>50K`.

---

## 🔍 Project Tasks

### 1. 🧹 Missing Data Exploration and Imputation

- Identified and visualized missing values using Nullity Matrix and heatmap.
- Applied various imputation strategies:
  - **K-Nearest Neighbors (KNN)** for numerical features.
- Assessed impact of imputation on data quality and model performance.

### 2. 📊 Exploratory Data Analysis (EDA) for Feature Selection

- Conducted univariate and multivariate analyses to assess distributions and relationships.
- Used statistical tests such as:
  - **Chi-Square Test** for categorical features.
  - **Kruskal-Wallis H Test** for numerical features.
- Applied feature importance techniques (e.g., Random Forest importance scores) to identify relevant predictors.

### 3. 📈 EDA for Data Insights

- Explored demographic patterns, gender-income disparities, and education vs. income trends.
- Created visualizations like histograms, box plots, KDEs, and bar charts for deep insights.

### 4. 🤖 Model Development

Developed multiple classification models to predict income:

- **Gradient Boosting Machine (GBM)**
- **AdaBoost Classifier**
- **Random Forest Classifier**
- **Artificial Neural Network (ANN)** using TensorFlow/Keras

Applied:
- Hyperparameter tuning using `GridSearchCV`.
- Regularization (L1/L2) and early stopping (for ANN).
- Stratified train-test split for maintaining class balance.

### 5. 📐 Model Evaluation

- Evaluated models using key metrics:
  - **Accuracy**
  - **Precision, Recall, F1-Score**
  - **Confusion Matrix**
  - **ROC-AUC Curve**
- Performed cross-validation and compared model performances.
- Identified overfitting/underfitting and refined model architectures accordingly.

### 6. 🖥️ Interactive Dashboard

Developed a **Streamlit**-based interactive dashboard including:

- **Missing Data Handler**: Upload CSV, explore missingness, choose imputation methods.
- **Data Exploration Module**: Dynamic feature selection, filter options, and AI-generated insights.
- **Model Prediction**: Train new models or upload pretrained ones, perform income prediction.
- **Clustering and Dimensionality Reduction**: Explore data structure using PCA, FAMD, and clustering.
- **Export and Download Options**: Save insights, feature importances, and predictions.

---

```bash
streamlit run app.py
