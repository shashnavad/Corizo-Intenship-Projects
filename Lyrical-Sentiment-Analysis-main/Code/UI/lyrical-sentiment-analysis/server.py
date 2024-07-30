from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import text

app = Flask(__name__)
port = 3000

# Function to load the classical model and vectorizer
def load_classical_model():
	script_dir = os.path.dirname(__file__)
	model = joblib.load(os.path.join(script_dir, 'models/logistic_regression_model.pkl'))
	vectorizer = joblib.load(os.path.join(script_dir, 'models/tfidf_vectorizer.pkl'))
	return model, vectorizer

# Function to load the BERT model from the same directory as the script
def load_bert_model():
	script_dir = os.path.dirname(__file__)
	model_path = os.path.join(script_dir, 'models/cleaned_model.tflite')
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

# Server static files
@app.route('/static/<path:path>')
def serve_static(path):
	return send_from_directory('build', path)

# JSON parsing middleware equivalent
@app.before_request
def json_middleware():
	if request.method == 'POST':
		if request.is_json:
			request.data_json = request.get_json()
		else:
			return jsonify(error="Invalid JSON"), 400

# Handle endpoint
@app.route('/classify', methods=['POST'])
def classify_text():
	try:
		data = request.get_json()
		if 'model' not in data or 'lyrics' not in data:
			print("Invalid parameters")
			return jsonify(error="Invalid parameters"), 400
		
		if data['model'] == "Classical":
			result = classify_text_classical(data['lyrics'])
		elif data['model'] == "Deep Learning":
			result = classify_text_bert(data['lyrics'])
		else:
			return jsonify({'error': 'Invalid model type'}), 400

		return jsonify(result)
	
	except Exception as e:
		print('Error executing query', str(e))
		return jsonify(error="Internal server error"), 500

if __name__ == '__main__':
	app.run(port=port)
