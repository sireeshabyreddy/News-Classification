import streamlit as st
import joblib
import pandas as pd

# Load the trained model and vectorizer
model = joblib.load('text_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Title of the app
st.title('Text Classification App')

# Input text from the user
text_input = st.text_area("Enter the text to classify:")

# If the user clicks the classify button
if st.button('Classify'):
    if text_input:
        # Convert the input text into the same format used during training
        text_tfidf = vectorizer.transform([text_input])
        
        # Predict the category
        prediction = model.predict(text_tfidf)
        
        # Display the result
        st.write(f'Predicted Category: {prediction[0]}')
    else:
        st.write('Please enter some text.')
