# Day Topics - Data Science Learning Path (Days 1-10)

Structured 10-day curriculum covering data science fundamentals to intermediate ML. Each day includes Jupyter notebooks, code demos, datasets, and Streamlit apps for hands-on practice. Track progress with daily commits.[1]

## 📋 Table of Contents (Days 1-10)

- [Day 1: Python & NumPy](#day-1-python--numpy)
- [Day 2: Pandas & Data Cleaning](#day-2-pandas--data-cleaning)
- [Day 3: EDA & Visualization](#day-3-eda--visualization)
- [Day 4: ML Libraries Intro](#day-4-ml-libraries-intro)
- [Day 5: Preprocessing Pipelines](#day-5-preprocessing-pipelines)
- [Day 6: Classification Algorithms](#day-6-classification-algorithms)
- [Day 7: Regression Models](#day-7-regression-models)
- [Day 8: Clustering & Dimensionality](#day-8-clustering--dimensionality)
- [Day 9: Model Evaluation](#day-9-model-evaluation)
- [Day 10: Streamlit Deployment](#day-10-streamlit-deployment)

## Day 1: Python & NumPy {#day-1-python--numpy}

**Focus**: Array operations, broadcasting, linear algebra  
**Files**: `numpy_demo.py`, `arrays_intro.ipynb`
**Run**: `python Day1/numpy_intro.py`

## Day 2: Pandas & Data Cleaning {#day-2-pandas--data-cleaning}

**Focus**: DataFrames, missing values, merging, grouping  
**Files**: `pandas_eda.ipynb`, Mall_Customers.csv  
**Key**: `df.fillna()`, `pd.merge()`

## Day 3: EDA & Visualization {#day-3-eda--visualization}

**Focus**: Histograms, scatter plots, correlation heatmaps[1]
**Files**: `eda_workflow.ipynb`, Seaborn templates  
**Libs**: Matplotlib, Seaborn, Plotly

## Day 4: ML Libraries Intro {#day-4-ml-libraries-intro}

**Focus**: scikit-learn basics, train-test split  
**Files**: `sklearn_intro.ipynb`  
**Code**: `train_test_split(X, y, test_size=0.2)`

## Day 5: Preprocessing Pipelines {#day-5-preprocessing-pipelines}

**Focus**: StandardScaler, LabelEncoder, ColumnTransformer[2]
**Files**: `preprocessing_pipeline.py`  
**Demo**: Housing dataset scaling

## Day 6: Classification Algorithms {#day-6-classification-algorithms}

**Focus**: Logistic Regression, KNN, Decision Trees
**Files**: `classification_models.ipynb`  
**Metrics**: Accuracy, Precision, Recall

## Day 7: Regression Models {#day-7-regression-models}

**Focus**: Linear Regression, Random Forest Regressor[3]
**Files**: `house_price_predictor.py` (R²=0.98)  
**Dataset**: Home_final.csv

## Day 8: Clustering & Dimensionality {#day-8-clustering--dimensionality}

**Focus**: K-Means, Elbow method (k=5), PCA[4][5]
**Files**: `k_means_streamlit.py`  
**Demo**: Mall customer segments + centroids

## Day 9: Model Evaluation {#day-9-model-evaluation}

**Focus**: Confusion Matrix, Cross-Validation, ROC curves
**Files**: `model_metrics.ipynb`  
**Advanced**: Gini vs Entropy impurity

## Day 10: Streamlit Deployment {#day-10-streamlit-deployment}

**Focus**: Interactive apps, requirements.txt, Cloud deploy[6]
**Files**: `requirements.txt`, full apps folder  
**Run**: `streamlit run Day10/final_app.py`

## 🏃‍♂️ Complete Setup

```bash
cd Day_topics
python -m venv .venv
source .venv/bin/activate  # Linux
pip install numpy pandas scikit-learn streamlit plotly seaborn matplotlib opencv-python jupyter
jupyter lab  # Edit all notebooks
```

**Per-Day Requirements**: Check each folder's `requirements.txt`

## 📊 Progress Tracker

| Day | Status | Key Achievement | Streamlit Demo |
|-----|--------|-----------------|---------------|
| 1 | ✅ | NumPy 3D arrays | - |
| 2 | ✅ | Pandas pipeline | [View CSV EDA](Day2) |
| 3 | ✅ | Plotly dashboards | [Day3](Day3) |
| 4-10 | 🔄 | In progress | Deploy after completion |

