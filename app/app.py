import streamlit as st
import pandas as pd
import requests
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# -------------------------------
# 🔧 Load trained model and scaler
# -------------------------------
model = pickle.load(open(r"C:\Users\divye\solar_predic_app\model.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\divye\solar_predic_app\scaler.pkl", "rb"))

# -------------------------------
# 🌤️ Function: Fetch weather data
# -------------------------------
def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        cloud = data["clouds"]["all"]
        wind = data["wind"]["speed"]
        return temp, cloud, wind
    else:
        return None

# -------------------------------
# 🖥️ Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Solar Energy Predictor", page_icon="☀️", layout="centered")

st.markdown(
    """
    <style>
        .main {
            background-color: blue;
        }
        h1 {
            color: #ff9900;
            text-align: center;
        }
        .stMetric {
            text-align: center !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("☀️ Solar Energy Prediction Dashboard")
st.markdown("### Predict your solar power generation using **live weather data** for your area.")

st.divider()

# Input section
col1, col2 = st.columns(2)
with col1:
    city = st.text_input("🌍 Enter City", "Alwal,IN")
with col2:
    api_key = "812287a71bfd86a8c22a8e2233655ebb"

if st.button("🚀 Get Prediction", use_container_width=True):
    weather = get_weather(city, api_key)

    if weather:
        temp, cloud, wind = weather

        # --- Weather Info Section ---
        st.markdown("### 🌤️ Current Weather Conditions")
        wcol1, wcol2, wcol3 = st.columns(3)
        wcol1.metric("Temperature (°C)", f"{temp:.1f}")
        wcol2.metric("Cloud Cover (%)", f"{cloud}")
        wcol3.metric("Wind Speed (m/s)", f"{wind:.1f}")

        # --- Model Input ---
        today = datetime.today()
        X_input = pd.DataFrame(
            [[temp, cloud, wind, today.month, today.timetuple().tm_yday]],
            columns=['Temp', 'Cloud', 'Wind', 'Month', 'DayOfYear']
        )

        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]

        st.divider()
        st.markdown("### 🔆 Predicted Solar Energy Output")
        st.success(f"**{prediction:.2f} kWh** expected today in {city}.")
    else:
        st.error("❌ Failed to fetch weather data. Check city name or API key.")
