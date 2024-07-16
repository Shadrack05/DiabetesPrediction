import pandas as pd
import streamlit as st

# import matplotlib as plt
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

df = pd.read_csv(r'C:\Users\Admin\Downloads\diabetes.csv')
df.groupby(['Outcome']).mean()

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df)
df_robust = pd.DataFrame(df_robust)
df_robust.head()


def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


# Remove outliers from the 'Glucose' column
columns_to_remove_outliers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction',
                              'Insulin', 'BMI', 'Age']

# Remove outliers from the specified columns
df_cleaned = remove_outliers(df, columns_to_remove_outliers)

# Separate features and target variable
X = df.drop(columns='Outcome')
y = df['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0, stratify=y)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Logistic Regression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train, y_train)

y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Random Forest Evaluation
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Gradient Boosting Evaluation
y_pred_gb = gb_clf.predict(X_test)
print("Gradient Boosting")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))
print("Classification Report:\n", classification_report(y_test, y_pred_gb))


def main():
    st.title("Diabetes Onset Model Prediction App")

    # Add input fields for features
    feature1 = st.number_input("Pregnancies", min_value=0, value=0, step=1, format="%d")
    feature2 = st.number_input("Glucose", min_value=0, value=0, step=1, format="%d")
    feature3 = st.number_input("Blood Pressure", min_value=0, value=0, step=1, format="%d")
    feature4 = st.number_input("Skin Thickness", min_value=0, value=0, step=1, format="%d")
    feature5 = st.number_input("Insulin", min_value=0, value=0, step=1, format="%d")
    feature6 = st.number_input("BMI", min_value=0, value=0, step=1, format="%d")
    feature7 = st.number_input("DiabetesPedigreeFunction")
    feature8 = st.number_input("Enter your age", min_value=0, max_value=120, value=0, step=1, format="%d")
    # Add more inputs as needed

    # Create a DataFrame from inputs
    input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]],
                              columns=['Feature1', 'Feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                                       'feature7', 'feature8'])

    # Make prediction
    if st.button("Predict"):
        prediction = gb_clf.predict(input_data)
        if prediction[0] == 1:
            st.write("You might be having diabetes.")
        else:
            st.write("You have a low chance of having diabetes.")


if __name__ == "__main__":
    main()
