import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # Updated import for joblib

# Load your trained Logistic Regression model
model = joblib.load('logistic_regression_model.pkl')
# Assume the TF-IDF Vectorizer is also saved and needs to be loaded
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(lyrics):
    # Transform the input using the loaded TF-IDF vectorizer
    transformed_lyrics = tfidf_vectorizer.transform([lyrics])
    # Predict using the logistic regression model
    prediction = model.predict(transformed_lyrics)
    return "Positive" if prediction[0] == 'Positive' else "Negative"

# Set up the title of the dashboard
st.title('Lyrics Sentiment Analysis')

# Creating a text area for user input
user_input = st.text_area("Enter lyrics here:")

# When the user clicks the 'Analyze' button
if st.button('Analyze Sentiment'):
    sentiment = predict_sentiment(user_input)
    st.write(f'The predicted sentiment is: {sentiment}')
