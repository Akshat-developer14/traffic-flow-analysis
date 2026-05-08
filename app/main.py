import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Set page configuration
st.set_page_config(
    page_title="Traffic Flow Analysis & Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e2130 0%, #2b313e 100%);
        border: 1px solid #3e4451;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    base_path = os.path.join(os.getcwd(), 'models')
    models = {
        "predictor": joblib.load(os.path.join(base_path, 'xgboost_traffic_pipeline_predictor.joblib')),
        "classifier": joblib.load(os.path.join(base_path, 'congestion_classification_pipeline.joblib')),
        "clusterer": joblib.load(os.path.join(base_path, 'traffic_clustering_pipeline.joblib')),
        "anomaly": joblib.load(os.path.join(base_path, 'traffic_anomaly_pipeline.joblib'))
    }
    return models

try:
    models = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure model files exist in the 'models' directory.")
    st.stop()

# Header
st.title("🚦 Traffic Flow Intelligence Dashboard")
st.markdown("---")

# Sidebar for Inputs
st.sidebar.header("📍 Input Parameters")

with st.sidebar:
    use_current = st.checkbox("Use Current Time & Date", value=True)
    
    # Define user timezone
    user_tz = ZoneInfo("Asia/Kolkata")
    
    if use_current:
        now = datetime.now(user_tz)
        st.session_state.selected_date = now.date()
        st.session_state.selected_time = now.time()
        st.info(f"Using: {st.session_state.selected_date} {st.session_state.selected_time.strftime('%H:%M')} (IST)")
    else:
        # Initialize session state if not present
        now = datetime.now(user_tz)
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = now.date()
        if 'selected_time' not in st.session_state:
            st.session_state.selected_time = now.time()
            
        st.session_state.selected_date = st.date_input("Select Date", st.session_state.selected_date)
        st.session_state.selected_time = st.time_input("Select Time", st.session_state.selected_time)

    selected_date = st.session_state.selected_date
    selected_time = st.session_state.selected_time

    
    st.subheader("🌦️ Weather Conditions")
    weather_main = st.selectbox("Weather Main", 
                                ['Clouds', 'Clear', 'Mist', 'Rain', 'Snow', 'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall'])
    
    temp = st.number_input("Temperature (K)", min_value=0.0, max_value=350.0, value=288.0)
    rain_1h = st.number_input("Rain in last hour (mm)", min_value=0.0, max_value=100.0, value=0.0)
    snow_1h = st.number_input("Snow in last hour (mm)", min_value=0.0, max_value=100.0, value=0.0)
    clouds_all = st.slider("Cloud Coverage (%)", 0, 100, 40)
    
    st.subheader("📅 Context")
    is_holiday = st.checkbox("Is it a Holiday?")

# Data Preprocessing Logic
def get_features(date, time, temp, rain, snow, clouds, weather, holiday):
    dt = datetime.combine(date, time)
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    holiday_val = 1 if holiday else 0
    
    # Base features for Predictor
    features_pred = pd.DataFrame({
        'temp': [temp],
        'rain_1h': [rain],
        'snow_1h': [snow],
        'clouds_all': [clouds],
        'weather_main': [weather],
        'is_holiday': [holiday_val],
        'day_of_week': [day_of_week],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'month_sin': [month_sin],
        'month_cos': [month_cos]
    })
    
    # Features for Classifier (based on notebook, it dropped weather_main)
    features_clf = features_pred.drop(columns=['weather_main'])
    
    return features_pred, features_clf, hour_sin, hour_cos, day_of_week

features_pred, features_clf, h_sin, h_cos, dow = get_features(
    selected_date, selected_time, temp, rain_1h, snow_1h, clouds_all, weather_main, is_holiday
)

# Main Application Logic
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Real-time Traffic Analysis")
    
    if st.button("🚀 Run Analysis"):
        # 1. Predict Traffic Volume
        vol_pred = models['predictor'].predict(features_pred)[0]
        
        # 2. Predict Congestion Level
        # Ensure we only pass features expected by the classifier
        # We need to see if the classifier pipeline was trained on specific columns
        # Based on notebook, X was dropped traffic_volume, congestion_level, weather_main
        congestion_pred = models['classifier'].predict(features_clf)[0]
        congestion_labels = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
        
        # 3. Predict Anomaly
        # Anomaly detector expects: ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume', 'is_holiday', 'day_of_week', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        features_anomaly = features_clf.copy()
        features_anomaly.insert(4, 'traffic_volume', [vol_pred])
        anomaly_pred = models['anomaly'].predict(features_anomaly)[0]
        anomaly_status = "⚠️ Anomaly Detected" if anomaly_pred == -1 else "✅ Normal Traffic"
        
        # 4. Clustering
        # Clusterer expects: ['traffic_volume', 'hour_sin', 'hour_cos', 'day_of_week']
        features_cluster = pd.DataFrame({
            'traffic_volume': [vol_pred],
            'hour_sin': [h_sin],
            'hour_cos': [h_cos],
            'day_of_week': [dow]
        })
        cluster_pred = models['clusterer'].predict(features_cluster)[0]
        
        cluster_names = {
            0: "🌙 Late Night Flow",
            1: "🌆 Afternoon Rush",
            2: "📉 Off-Peak / Holiday",
            3: "🌅 Morning Rush Hour",
            4: "🏙️ Weekend Mid-day",
            5: "🌃 Evening Flow"
        }

        cluster_name = cluster_names.get(cluster_pred, f"Cluster {cluster_pred}")
        
        # Display Results
        st.markdown(f"""
        <div class="prediction-card">
            <h2 style='text-align: center; color: #ff4b4b; margin-bottom: 20px;'>Prediction Results</h2>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; text-align: center;'>
                <div>
                    <p style='font-size: 0.9em; color: #888; margin: 0;'>Traffic Volume</p>
                    <h3 style='font-size: 2em; margin: 5px 0;'>{int(vol_pred)}</h3>
                    <p style='font-size: 0.8em; margin: 0;'>Vehicles/Hour</p>
                </div>
                <div>
                    <p style='font-size: 0.9em; color: #888; margin: 0;'>Congestion Level</p>
                    <h3 style='font-size: 2em; margin: 5px 0;'>{congestion_labels[congestion_pred]}</h3>
                </div>
                <div>
                    <p style='font-size: 0.9em; color: #888; margin: 0;'>Traffic Pattern</p>
                    <h3 style='font-size: 1.5em; margin: 5px 0;'>{cluster_name}</h3>
                </div>
                <div>
                    <p style='font-size: 0.9em; color: #888; margin: 0;'>System Status</p>
                    <h3 style='font-size: 1.5em; margin: 5px 0;'>{anomaly_status}</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("Analysis complete!")
    else:
        st.info("Configure the parameters in the sidebar and click 'Run Analysis' to see predictions.")


with col2:
    st.subheader("💡 Key Insights")
    st.write("This application uses a multi-model pipeline to analyze urban traffic flow:")
    st.markdown("""
    - **XGBoost Regressor**: For high-accuracy volume prediction.
    - **Random Forest Classifier**: Categorizes congestion into Low, Medium, or High.
    - **K-Means Clustering**: Identifies patterns in traffic behavior.
    - **Isolation Forest**: Detects unusual traffic anomalies.
    """)
    

# Footer & Project Details
st.markdown("---")
st.subheader("🔍 Project Documentation")

tab1, tab2, tab3 = st.tabs(["📖 About Project", "🤖 Model Details", "🛠️ Feature Engineering"])

with tab1:
    st.markdown("""
    ### Project Overview
    This project, **Urban Traffic Flow Intelligence**, aims to provide actionable insights into city traffic patterns using historical data. 
    By leveraging multi-model machine learning, we can not only predict how many vehicles will be on the road but also identify 
    unusual events (anomalies) and classify the severity of congestion.

    **Dataset**: The models are trained on the *Metro Interstate Traffic Volume* dataset, which includes hourly traffic volume 
    on I-94 between Minneapolis and St Paul, Minnesota, coupled with weather and holiday information.
    """)
    st.info("💡 **Use Case**: This dashboard can be used by city planners to simulate traffic under different weather conditions or holidays.")

with tab2:
    st.markdown("""
    ### Machine Learning Pipeline
    The application integrates four distinct models to provide a comprehensive analysis:
    
    1. **Traffic Volume Predictor (XGBoost)**: 
       - **Type**: Regression
       - **Purpose**: Predicts the exact number of vehicles per hour.
       - **Performance**: High accuracy using gradient-boosted decision trees.
       
    2. **Congestion Classifier (Random Forest)**:
       - **Type**: Multi-class Classification
       - **Purpose**: Categorizes traffic into *Low*, *Medium*, or *High* congestion levels.
       
    3. **Pattern Clusterer (K-Means)**:
       - **Type**: Unsupervised Clustering
       - **Purpose**: Groups traffic states into clusters based on volume and time patterns (e.g., Weekday Morning Peak).
       
    4. **Anomaly Detector (Isolation Forest)**:
       - **Type**: Unsupervised Anomaly Detection
       - **Purpose**: Flags data points that deviate significantly from typical patterns (e.g., extreme weather or accidents).
    """)

with tab3:
    st.markdown(r"""
    ### Feature Engineering & Preprocessing
    To achieve high predictive performance, the raw data undergoes several transformations:
    
    *   **Cyclical Encoding**: Time-based features (Hour, Month) are transformed using Sine and Cosine functions. 
        This preserves the relationship that hour 23 is close to hour 0.
        - $Hour_{sin} = \sin(2\pi \cdot \frac{Hour}{24})$
        - $Hour_{cos} = \cos(2\pi \cdot \frac{Hour}{24})$
    *   **One-Hot Encoding**: Categorical variables like `weather_main` are converted into numerical format for model compatibility.
    *   **Standard Scaling**: Numerical features are scaled to a standard normal distribution to ensure all features contribute equally to the distance-based models (like K-Means).
    """)

    
    with st.expander("🛠️ View Current Feature Vector"):
        st.write("This is the exact data being passed to the models based on your sidebar inputs:")
        st.dataframe(features_pred)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #555;'>Traffic Flow Analysis Project &copy; 2026</p>", unsafe_allow_html=True)

