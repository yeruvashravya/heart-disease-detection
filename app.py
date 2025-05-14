import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Prepare data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(X_train.values, Y_train.values)

# Streamlit app starts here
st.title("Heart Disease Prediction App")

st.markdown("""
This app predicts whether a person has **heart disease** based on medical attributes.
""")

# Input sliders
age = st.slider('Age', 29, 77, 50)
sex = st.selectbox('Sex (1 = male, 0 = female)', [1, 0])
cp = st.selectbox('Chest Pain Type (0-3)', [0,1,2,3])
trestbps = st.slider('Resting Blood Pressure', 94, 200, 120)
chol = st.slider('Serum Cholesterol (mg/dl)', 126, 564, 240)
fbs = st.selectbox('Fasting Blood Sugar >120 mg/dl (1 = yes, 0 = no)', [1, 0])
restecg = st.selectbox('Resting ECG (0-2)', [0,1,2])
thalach = st.slider('Max Heart Rate Achieved', 71, 202, 150)
exang = st.selectbox('Exercise Induced Angina (1 = yes, 0 = no)', [1,0])
oldpeak = st.slider('ST Depression Induced', 0.0, 6.2, 1.0)
slope = st.selectbox('Slope of Peak Exercise ST (0-2)', [0,1,2])
ca = st.selectbox('Number of Major Vessels (0-3)', [0,1,2,3])
thal = st.selectbox('Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)', [1,2,3])

# Predict button
if st.button('Predict Heart Disease'):

    # Create input array
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, 
                  thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # Make prediction
    prediction = model.predict(input_data_reshaped)

    # Display result
    if prediction[0] == 0:
        st.success('The person **does not** have Heart Disease.')
    else:
        st.error('The person **has** Heart Disease.')

# Show model performance
st.subheader('Model Accuracy')
X_train_pred = model.predict(X_train)
train_acc = accuracy_score(Y_train, X_train_pred)
X_test_pred = model.predict(X_test)
test_acc = accuracy_score(Y_test, X_test_pred)

st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Test Accuracy: {test_acc:.2f}")
