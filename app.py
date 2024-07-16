# app.py (your Streamlit app)
import streamlit as st
import pandas as pd
import joblib as jb

# Load the saved model
model = jb.load('diabetesModel.pkl')


def main():
    st.title("Diabetes Prediction App")

    # Add input fields for features
    feature1 = st.number_input("Pregnancies", min_value=0, value=0, step=1, format="%d")
    feature2 = st.number_input("Glucose", min_value=0, value=0, step=1, format="%d")
    feature3 = st.number_input("Blood Pressure", min_value=0, value=0, step=1, format="%d")
    feature4 = st.number_input("Skin Thickness", min_value=0, value=0, step=1, format="%d")
    feature5 = st.number_input("Insulin", min_value=0, value=0, step=1, format="%d")
    feature6 = st.number_input("BMI", min_value=0, value=0, step=1, format="%d")
    feature7 = st.number_input("DiabetesPedigreeFunction")
    feature8 = st.number_input("Enter your age", min_value=0, max_value=120, value=0, step=1, format="%d")

    # Create a DataFrame from inputs
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]],
                              columns=['Feature1', 'Feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                                       'feature7', 'feature8'])

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.write("You might be having diabetes.")
        else:
            st.write("You have a low chance of having diabetes.")


if __name__ == "__main__":
    main()
