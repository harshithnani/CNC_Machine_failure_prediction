# Importing necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Title of the app
st.title("CNC Machine Failure Prediction")

# Link to GitHub repository
st.markdown("[GitHub Repository](https://github.com/your-username/cnc-machine-failure-prediction)")

# Upload dataset
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    try:
        # Read the uploaded file
        data = pd.read_excel(uploaded_file)
        
        # Display the dataset
        st.write("Dataset:")
        st.write(data.head())
        
        # Select target column
        target_column = st.selectbox("Select the target column", data.columns)
        
        # Splitting the dataset into features and target variable
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create and train the model
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Display the evaluation metrics
        st.write(f"Accuracy: {accuracy}")
        st.write("Classification Report:")
        st.text(report)
    except Exception as e:
        st.error(f"An error occurred: {e}")
