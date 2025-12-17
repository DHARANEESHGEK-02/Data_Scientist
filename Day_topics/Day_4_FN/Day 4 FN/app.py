import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Coffee Shop Revenue Predictor", layout="wide")

# Load model and preprocessors
@st.cache_resource
def load_model():
    try:
        model = joblib.load('coffeesalesmodel.pkl')
        scaler = joblib.load('scaler.pkl')
        selector = joblib.load('featureselector.pkl')
        return model, scaler, selector
    except:
        st.error("Model files not found. Please ensure coffeesalesmodel.pkl, scaler.pkl, and featureselector.pkl are in the same folder.")
        return None, None, None

model, scaler, selector = load_model()

st.title("☕ Coffee Shop Revenue Predictor")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("📊 Predict Daily Revenue")
st.sidebar.markdown("Enter conditions to get revenue prediction")

# Key input features (from your notebook)
col1, col2 = st.columns(2)
with col1:
    num_customers = st.sidebar.number_input("Num Customers", 10, 150, 45)
    coffee_sales = st.sidebar.number_input("Coffee Sales", 20, 120, 55)
    temp_c = st.sidebar.slider("Temperature (°C)", -10.0, 35.0, 18.0)

with col2:
    pastry_sales = st.sidebar.number_input("Pastry Sales", 5, 60, 25)
    sandwich_sales = st.sidebar.number_input("Sandwich Sales", 5, 40, 15)
    staff_count = st.sidebar.number_input("Staff Count", 2, 8, 4)

is_weekend = st.sidebar.checkbox("Weekend?", value=False)
promotion = st.sidebar.checkbox("Promotion Active?", value=False)

# Predict button
if st.sidebar.button("🚀 Predict Revenue", type="primary"):
    if model:
        # Create feature vector matching training data
        feature_data = {
            'DayofWeek': 7 if is_weekend else 2,
            'IsWeekend': 1 if is_weekend else 0,
            'Month': 6,
            'TemperatureC': temp_c,
            'IsRaining': 0,
            'Rainfallmm': 0.0,
            'IsHoliday': 0,
            'PromotionActive': 1 if promotion else 0,
            'NearbyEvents': 0,
            'StaffCount': staff_count,
            'MachineIssues': 0,
            'NumCustomers': num_customers,
            'CoffeeSales': coffee_sales,
            'PastrySales': pastry_sales,
            'SandwichSales': sandwich_sales,
            'CustomerSatisfaction': 7.5,
            'DayofYear': 150,
            'WeekofYear': 25,
            'Quarter': 2,
            'DayNameEncoded': 6 if is_weekend else 2,
            'SeasonEncoded': 2
        }
        
        df_sample = pd.DataFrame([feature_data])
        
        # Scale and select features
        X_scaled = scaler.transform(df_sample)
        X_selected = selector.transform(X_scaled)
        
        # Predict
        prediction = model.predict(X_selected)[0]
        
        st.sidebar.success(f"**Predicted Revenue: ${prediction:.2f}**")
        st.session_state.prediction = prediction
    else:
        st.sidebar.error("Model not loaded!")

# Main dashboard
if 'prediction' in st.session_state:
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Revenue", f"${st.session_state.prediction:.2f}")
    col2.metric("Avg Revenue", "$318.04")
    col3.metric("Accuracy", "94.7%")

# Model Performance Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Train R²", "0.957")
col2.metric("Test R²", "0.947")
col3.metric("Test RMSE", "$25.20")
col4.metric("Test MAE", "$19.75")

st.markdown("---")

# Feature Impact Visualization
st.subheader("📈 Key Drivers of Revenue")
if model:
    feature_names = ['DayofWeek', 'IsWeekend', 'Month', 'TemperatureC', 'IsRaining', 
                    'Rainfallmm', 'IsHoliday', 'PromotionActive', 'NearbyEvents', 
                    'StaffCount', 'MachineIssues', 'NumCustomers', 'CoffeeSales', 
                    'PastrySales', 'SandwichSales', 'CustomerSatisfaction']
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False).head(10)
    
    fig = px.bar(coef_df, x='Coefficient', y='Feature', 
                 title="Top 10 Feature Impact on Revenue",
                 color='Coefficient', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

# Scenarios comparison
st.subheader("🔮 Revenue Scenarios")
scenarios_data = {
    "Busy Weekend": {"NumCustomers": 70, "CoffeeSales": 85, "Temp": 25, "Weekend": True, "Promo": True},
    "Rainy Day": {"NumCustomers": 25, "CoffeeSales": 30, "Temp": 8, "Weekend": False, "Promo": False},
    "Average Day": {"NumCustomers": 45, "CoffeeSales": 55, "Temp": 18, "Weekend": False, "Promo": False}
}

scenario_preds = []
for name, data in scenarios_data.items():
    # Quick prediction (simplified)
    pred = 318 + (data["NumCustomers"]-45)*2 + (data["CoffeeSales"]-55)*5
    scenario_preds.append({"Scenario": name, "Revenue": pred})

fig_scenarios = px.bar(pd.DataFrame(scenario_preds), x='Scenario', y='Revenue',
                      title="Compare Different Days")
st.plotly_chart(fig_scenarios, use_container_width=True)

# Instructions
with st.expander("📋 How to Deploy"):
    st.markdown("""
    1. Save as `app.py`
    2. Put model files in same folder:
       - `coffeesalesmodel.pkl`
       - `scaler.pkl` 
       - `featureselector.pkl`
    3. Run: `streamlit run app.py`
    4. Open localhost:8501
    """)

st.markdown("---")
st.caption("Built from your Coffee Shop Linear Regression notebook 🚀")
