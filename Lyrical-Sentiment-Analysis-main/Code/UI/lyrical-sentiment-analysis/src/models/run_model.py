import joblib
import json
import sys
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import text

# Function to load the classical model and vectorizer
def load_classical_model():
	script_dir = os.path.dirname(__file__)
	model = joblib.load(os.path.join(script_dir, 'logistic_regression_model.pkl'))
	vectorizer = joblib.load(os.path.join(script_dir, 'tfidf_vectorizer.pkl'))
	return model, vectorizer

# Function to load the BERT model from the same directory as the script
def load_bert_model():
	script_dir = os.path.dirname(__file__)
	model_path = os.path.join(script_dir, 'cleaned_model.tflite')
	base_options = python.BaseOptions(model_asset_path=model_path)
	options = text.TextClassifierOptions(base_options=base_options)
	classifier = text.TextClassifier.create_from_options(options)
	return classifier

# Predict using classical model
def classify_text_classical(input_text):
	transformed_text = tfidf_vectorizer.transform([input_text])
	prediction = lr_model.predict(transformed_text)
	return prediction[0]

# Predict using bert model
def classify_text_bert(input_text):
	classification_result = bert_classifier.classify(input_text)
	top_category = [(x.category_name, x.score) for x in classification_result.classifications[0].categories]
	return top_category

# Load models
lr_model, tfidf_vectorizer = load_classical_model()
bert_classifier = load_bert_model()

if __name__ == "__main__":
	input_json = sys.stdin.read()
	input_dict = json.loads(input_json)

	if input_dict['model'] == "Classical":
		result = classify_text_classical(input_dict['lyrics'])
	elif input_dict['model'] == "Deep Learning":
		result = classify_text_bert(input_dict['lyrics'])
	else:
		exit(0)

	print(json.dumps(result))