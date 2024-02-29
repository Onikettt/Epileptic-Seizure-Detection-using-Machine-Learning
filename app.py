# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to train a machine learning model
def train_classifier(X_train, y_train, model):
    # Train your model
    model.fit(X_train, y_train)
    return model

# Function to make predictions using the trained model
def make_predictions(X_test, model):
    # Make predictions
    y_pred = model.predict(X_test)
    return y_pred

# Function to evaluate the model's performance
def evaluate_model(y_true, y_pred):
    # Evaluate model accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Streamlit app
def main():
    # Title of the app
    st.title('Epileptic Seizures Detection App')

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load and preprocess the uploaded data
        data = pd.read_csv(uploaded_file)

        # Map y target column for binary classification
        data['y'] = data['y'].map({5: 0, 4: 0, 3: 0, 2: 0, 1: 1})

        # Remove unnecessary column
        data = data.drop('Unnamed: 0', axis=1)  # Updated column name to match

        # Shuffle the data
        data = shuffle(data)

        # Normalize the data
        X_normalized = normalize(data.drop('y', axis=1))

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, data['y'], test_size=0.3, random_state=42)

        # Train a RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_features='sqrt')
        trained_model = train_classifier(X_train, y_train, model)

        # Make predictions on new data
        st.header("Make a Prediction")

        # Collect user input for prediction
        feature_values = []

        for feature in data.drop('y', axis=1).columns:
            value = st.slider(f"Select {feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
            feature_values.append(value)

        # Normalize user input
        user_input_normalized = normalize([feature_values])

        # Make a prediction
        prediction = make_predictions(user_input_normalized, trained_model)

        # Display the prediction result
        st.write(f"Prediction: {'Symptoms Detected' if prediction[0] else 'No Symptoms Detected'}")

        # Evaluate model accuracy on test data
        y_pred_test = make_predictions(X_test, trained_model)
        accuracy = evaluate_model(y_test, y_pred_test)
        st.write(f'Model Accuracy on Test Data: {accuracy * 100:.2f}%')

