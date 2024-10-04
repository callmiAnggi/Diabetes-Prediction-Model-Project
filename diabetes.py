import streamlit as st
import pandas as pd
import random
import xgboost as xgb
import numpy as np
import pickle
import joblib

st.title("Diabetes Predictor")
xg = pickle.load(open("diabetes_model.pkl", 'rb'))
dict = {
    0 : "Not Diabetic",
    1 : "Pre-Diabetes",
    2 : "Diabetes"
}

#Predicting X:

col1, col2 = st.columns(2)

# Make sure height and weight are positive floats
BMI = 0.0
height = col1.number_input("Height in cm", 0)
weight = col1.number_input("Mass in kg", 0)

if (weight > 0 and height  > 0):
    BMI = float(weight)/(height*height)*10000
    BMI = round(BMI, 1)

Age = col1.number_input("Age", 0)

c1 = st.container()
col1, col2 = c1.columns(2)

# Collect number of features

# Collect user inputs for each feature
GenHlth = st.slider("From 1 to 5, how would you rate your general health? (1 = excellent, 5 = poor)", 1, 5)
PhysHlth = st.slider("How many days during the past 30 days was your physical health not good?", 1, 30)
MentHlth = st.slider("How many days during the past 30 days was your mental health not good?", 1, 30)
HighBP = st.radio("Do you have high blood pressure?", ('Yes', 'No'))
if HighBP == 'Yes':
    HighBP = 1
else:
    HighBP = 0

HighChol = st.radio("Do you have high cholesterol?", ('Yes', 'No'))
if HighChol == 'Yes':
    HighChol = 1
else:
    HighChol = 0

Stroke = st.radio("Have you ever had a stroke?", ('Yes', 'No'))
if Stroke == 'Yes':
    Stroke = 1
else:
    Stroke = 0

HeartDiseaseorAttack = st.radio("Have you ever had a heart attack or heart disease?", ('Yes', 'No'))
if HeartDiseaseorAttack == 'Yes':
    HeartDiseaseorAttack = 1
else:
    HeartDiseaseorAttack = 0

DiffWalk = st.radio("Do you have difficulty walking or climbing stairs?", ('Yes', 'No'))
if DiffWalk == 'Yes':
    DiffWalk = 1
else:
    DiffWalk = 0

sc = joblib.load('std_scaler.bin')
input_array = [GenHlth, HighBP, BMI, DiffWalk, HighChol, Age, HeartDiseaseorAttack, PhysHlth, Stroke, MentHlth]
input_array = np.array(input_array).reshape(1, -1)
# sc.transform the input array for index 0, 2, 5, 7, 9 of input_array
input_array[:, [0, 2, 5, 7, 9]] = sc.transform(input_array[:, [0, 2, 5, 7, 9]])


#input_array = []
#for i in range(21):
   # value = st.number_input(f'Enter value for feature {i}:', value=0.0)
    #input_array.append(value)
#input_array = np.array(input_array)
input_array = np.array(input_array).reshape(1, -1)
# # Display input array
st.write('### Input Array:', input_array)

# Button to make prediction
if st.button('Submit'):
    prediction = xg.predict(input_array)
    st.success(f'Prediction: {dict[prediction[0]]}')
    st.success(f'Probability of being diabetic: {xg.predict_proba(input_array)[0][1]}')
