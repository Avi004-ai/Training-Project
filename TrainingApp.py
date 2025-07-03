import streamlit as st
import numpy as np
import joblib
model= joblib.load('heart_model.pkl')
scaler=joblib.load('scaler.pkl')
st.set_page_config(page_title='Heart Disease Predictor',page_icon='❤️',layout='wide')
st.title("Heart Disease Risk Prediction App")
st.write("This app predicts the risk of heart disease based on user input features.")
age=st.number_input('Age',min_value=10,max_value=100,value=30)
sex=st.selectbox('Sex[1=Male,0=Female]',[1,0])
cp=st.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps=st.number_input('Resting Blood Pressure',min_value=80,max_value=200,value=120)
chol=st.number_input('Serum Cholestrol (mg/dl)',min_value=100,max_value=400,value=200)
fbs=st.selectbox('Fasting Blood Sugar > 120mg/dl (1=True,0=False)',[1,0])
restecg=st.selectbox('Resting ECG (0-2)',[0,1,2])
thalach=st.number_input('Max Heart Rate Achieved',min_value=60,max_value=250,value=150)
exang=st.selectbox('Exercise Induced Angina (1=Yes,0=No)',[1,0])
oldpeak=st.number_input('ST Depression Induced by Exercise Relative to Rest',min_value=0.0,max_value=10.0,value=1.0)
slope=st.selectbox('Slope of peak exercise ST segment (0-2)',[0,1,2])
ca=st.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-3)',[0,1,2,3])
thal=st.selectbox('Thalassemia (1=Normal,2=Fixed Defect,3=Reversible Defect)',[1,2,3])

input_data =np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
if st.button('Predict'):
    input_scaled=scaler.transform(input_data)
    prediction=model.predict(input_scaled)
    probability=model.predict_proba(input_scaled)[0][1]
    if prediction== 1:
        st.success(f"Heart Disease Risk Detected! Probability: {probability:.2f}")
        st.write("Please consult a healthcare professional for further evaluation and advice.")
    else:
        st.success(f"No Heart Disease Risk Detected. Probability: {probability:.2f}")
        st.write("Keep maintaining a healthy lifestyle to reduce the risk of heart disease.")

st.caption('Developed by Avinab Smanta | 9378130602   | Healthy Heart | 03/07/25')        