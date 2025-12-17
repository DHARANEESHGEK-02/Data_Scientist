import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Student GPA Regression Demo")

# 1. Load data
df = pd.read_csv("student_details.csv")
st.subheader("Raw Data")
st.dataframe(df)

# 2. Train–test split
X = df.drop("GPA", axis=1)
y = df["GPA"]
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.4, random_state=50
)

# 3. Scaling
scaler = StandardScaler()
X_train_ss = scaler.fit_transform(X_train)
X_test_ss = scaler.transform(X_test)

# 4. Train model
model = LinearRegression()
model.fit(X_train_ss, y_train)

# 5. Predict and metrics
y_pred = model.predict(X_test_ss)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"MSE: {mse:.4f}")
st.write(f"R² Score: {r2:.4f}")

# 6. Correlation (numeric only)
st.subheader("Correlation (Age, Credits, GPA)")
st.write(df[["Age", "Credits_Completed", "GPA"]].corr())

# 7. Coefficients table
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": model.coef_
}).sort_values("coef", key=np.abs, ascending=False)

st.subheader("Top 10 Features by Coefficient")
st.dataframe(coef_df.head(10))

# 8. Plot Actual vs Predicted
st.subheader("Actual vs Predicted GPA")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_test, y_pred, color="blue")
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
ax.set_xlabel("Actual GPA")
ax.set_ylabel("Predicted GPA")
ax.set_title("Actual vs Predicted GPA")
ax.grid(True)

st.pyplot(fig)
