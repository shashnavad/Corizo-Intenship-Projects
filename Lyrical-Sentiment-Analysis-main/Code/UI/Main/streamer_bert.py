import streamlit as st
import joblib
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import text

# Function to load the classical model and vectorizer
@st.cache(allow_output_mutation=True)
def load_classical_model():
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Function to load the BERT model from the same directory as the script
@st.cache(allow_output_mutation=True)
def load_bert_model():
    script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
    model_path = os.path.join(script_dir, 'cleaned_model.tflite')  # Path to the model file
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = text.TextClassifierOptions(base_options=base_options)
    classifier = text.TextClassifier.create_from_options(options)
    return classifier

# Load models
lr_model, tfidf_vectorizer = load_classical_model()
bert_classifier = load_bert_model()

# Define classification functions
def classify_text_classical(input_text):
    transformed_text = tfidf_vectorizer.transform([input_text])
    prediction = lr_model.predict(transformed_text)
    return prediction[0]

def classify_text_bert(input_text):
    classification_result = bert_classifier.classify(input_text)
    top_category = classification_result.classifications[0].categories[0]
    return top_category.category_name, top_category.score

# Streamlit UI setup
st.title('Lyrics Sentiment Analysis')
model_choice = st.radio("Choose the model for analysis:", ('Classical Logistic Regression', 'BERT Model'))
user_input = st.text_area("Enter lyrics here:")
if st.button('Analyze Sentiment'):
    if model_choice == 'Classical Logistic Regression':
        result = classify_text_classical(user_input)
        st.write(f'Sentiment: {result}')
    else:
        sentiment, confidence = classify_text_bert(user_input)
        st.write(f'Sentiment: {sentiment}, Confidence: {confidence:.2f}')
