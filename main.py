import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
import joblib as jb
import pandas as pd
import sklearn

# Use a raw string literal for the file path
model_path = r'C:\Users\Admin\model.pkl'

# Load the model
model = jb.load(model_path)


# print("Model loaded successfully.")

# Define your Streamlit app
def main():
    st.title("Diabetes Onset Model Prediction App")

    # Add input fields for features
    feature1 = st.number_input("Pregnancies")
    feature2 = st.number_input("Glucose")
    feature3 = st.number_input("BloodPressure")
    feature4 = st.number_input("SkinThickness")
    feature5 = st.number_input("Insulin")
    feature6 = st.number_input("BMI")
    feature7 = st.number_input("DiabetesPedigreeFunction")
    feature8 = st.number_input("Age")
    # Add more inputs as needed

    # Create a DataFrame from inputs
    input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]],
                              columns=['Feature1', 'Feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                                       'feature7', 'feature8'])

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write(f"Prediction: {prediction[0]}")


if __name__ == "__main__":
    main()
