 Data Scientist Learning Repository

This repository documents a comprehensive journey through data science fundamentals, machine learning algorithms, and practical projects. It features daily topics with Jupyter notebooks, Streamlit applications, and assignments covering NumPy, scikit-learn, clustering, regression, and computer vision.[1]

Organized as a 2nd-year Information Systems student's hands-on portfolio for skill-building and placement preparation.

 📁 Repository Structure

- **Day_topics/**: Daily Jupyter notebooks and code examples
  - Day 3: EDA, preprocessing, visualization, Python ML libraries
  - Clustering (K-Means on Mall Customers), model evaluation[2][3]
- **Assignments/**: Coursework and practical exercises
- Root files: Standalone demos like `app.py`, `demo.py`[1]

🛠️ Key Technologies & Skills Demonstrated

- **Languages**: Python (primary), SQL
- **Libraries**:
  | Category | Tools |
  |----------|-------|
  | Data Manipulation | NumPy, pandas  |
  | ML Algorithms | scikit-learn (KNN, SVM, Decision Trees, Random Forest, Logistic Regression, Naive Bayes, K-Means) [4] |
  | Visualization | Matplotlib, Seaborn, Plotly, Streamlit [2] |
  | Computer Vision | OpenCV  |
- **Projects**:
  - Mall Customer Segmentation (K-Means, elbow method, centroids)[3][2]
  - Random Forest House Price Predictor (Streamlit app)[5]
  - Face Recognition (Virat Kohli dataset, 162 images)
  - Image Classification

🚀 Quick Start

1. Clone the repo:  
   ```bash
   git clone https://github.com/DHARANEESHGEK-02/Data_Scientist.git
   cd Data_Scientist
   ```

2. Create virtual environment (Ubuntu/Linux):  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # If provided per project
   ```

3. Run examples:
   - Streamlit apps: `streamlit run app.py` or `streamlit run k_means.py`
   - Jupyter: `jupyter notebook Day_topics/DayX/*.ipynb`

📊 Example: K-Means Clustering (Mall Customers)

Interactive Streamlit app for customer segmentation:
- Upload CSV → EDA → Elbow plot (optimal k=5) → Visualize clusters + centroids[6]
- Fixed scaling: `scaler.inverse_transform(kmeans.cluster_centers_)` for accurate plots[3]

 📈 Model Evaluation Highlights

- Classification: Confusion matrices, Gini/Entropy impurity
- Regression: R² scores (e.g., 0.98 on house prices)[5]
- Preprocessing: StandardScaler, feature engineering

🤝 Contributing

Fork, create a branch, add your notebooks/apps, and submit a PR. Focus on daily topics or new ML projects.

