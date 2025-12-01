import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

st.title("🌲 Forest Fire Prediction App")

st.write("This app predicts the **Forest Fire Area** using Machine Learning.")

# ---------- Load Dataset ----------
df = pd.read_csv("forestfires.csv")  # rename if file name is different
st.subheader("Dataset Preview")
st.write(df.head())

# ---------- Train Model ----------
st.subheader("Model Training")
st.write("RandomForest model is being used.")

X = df[['temp', 'RH', 'wind', 'rain']]
y = df['area']

model = RandomForestRegressor()
model.fit(X, y)

st.success("Model trained!")

# ---------- User Input ----------
st.subheader("Predict Fire Area")

temp = st.number_input("Temperature", value=20)
rh = st.number_input("Relative Humidity", value=30)
wind = st.number_input("Wind Speed", value=4)
rain = st.number_input("Rain", value=0)

if st.button("Predict"):
    input_data = np.array([[temp, rh, wind, rain]])
    prediction = model.predict(input_data)[0]
    st.success(f"🔥 Predicted Fire Area: {prediction:.2f}")
