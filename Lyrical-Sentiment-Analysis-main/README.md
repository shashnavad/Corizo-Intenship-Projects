# Lyrical Sentiment Analysis
[![Python Version](https://img.shields.io/badge/Python-3.7.6-red)](https://www.python.org/downloads/release/python-3717/)
[![Tensorflow Version](https://img.shields.io/badge/tensorflow-2.3.0-lime)](https://www.tensorflow.org/)
[![Keras Version](https://img.shields.io/badge/keras-2.4.3-orange)](https://keras.io/)
[![Pypi Version](https://img.shields.io/badge/pypi-20.0.2-yellow)](https://pypi.org/)
[<img src="https://github.com/simple-icons/simple-icons/assets/63730759/902f5f08-2056-436a-8536-22c0dea221d8" width="100">](https://developers.google.com/mediapipe)
[![Selenium](https://img.shields.io/badge/-selenium-%43B02A?style=for-the-badge&logo=selenium&logoColor=white)](https://pypi.org/project/selenium/)

## Overview
Lyrical Sentiment Analysis in Python using Natural Language Processing(NLP) with the help of [Mediapipe](https://developers.google.com/mediapipe). This project focuses on sentiment analysis of song lyrics to identify positive and negative sentiments, employing advanced NLP and machine learning techniques. We have incorporated Google's MediaPipe framework specifically for enhancing the processing capabilities of our BERT model, previously leveraging only TensorFlow and Transformers from Hugging Face.

## Key Features
- **Data Collection**: Scraping the top songs from various online databases.
- **Preprocessing**: Standardizing data by removing special characters and tokenizing text.
- **Sentiment Analysis**: Using BERT model enhanced with MediaPipe for improved feature extraction and faster processing.
- **Visualization**: Displaying data insights through interactive charts and graphs.

## MediaPipe Integration
MediaPipe offers state-of-the-art machine learning solutions for media processing. In this project, we use MediaPipe to optimize our BERT model, enhancing its ability to process and analyze lyrical content efficiently. This integration allows for real-time analysis and increased accuracy in sentiment classification.

## Requirements
 To start the scraping process we need to get the requirement for the scarping.
 ```
pip install -r requirements.txt
```

To update the webdriver after we import all the required files with the command bellow:

```
pip install --upgrade webdriver-manager
```

## UI

### Classical Machine Learning Model UI

Download these file from the project:
- [logistic_regression_model.pkl](https://github.com/prathamgupta36/Lyrical-Sentiment-Analysis/blob/main/Code/UI/Classical%20ML/logistic_regression_model.pkl)
- [streamer.py](https://github.com/prathamgupta36/Lyrical-Sentiment-Analysis/blob/main/Code/UI/Classical%20ML/streamer.py)
- [tfidf_vectorizer.pkl](https://github.com/prathamgupta36/Lyrical-Sentiment-Analysis/blob/main/Code/UI/Classical%20ML/tfidf_vectorizer.pkl)

Then in the folder where the files are saved then do:
```
streamlit run streamer.py
```
This should open up a webui on your machine where you can input lyrics and see the sentiment.

### UI for Both
Download these file from the project:
- [logistic_regression_model.pkl](https://github.com/prathamgupta36/Lyrical-Sentiment-Analysis/blob/main/Code/UI/Classical%20ML/logistic_regression_model.pkl)
- [tfidf_vectorizer.pkl](https://github.com/prathamgupta36/Lyrical-Sentiment-Analysis/blob/main/Code/UI/Classical%20ML/tfidf_vectorizer.pkl)
- [streamer_bert.py](https://github.com/prathamgupta36/Lyrical-Sentiment-Analysis/blob/main/Code/UI/Main/streamer_bert.py)
- [cleaned_model.tflite](https://github.com/prathamgupta36/Lyrical-Sentiment-Analysis/blob/main/Code/UI/Main/cleaned_model.tflite)

Then in the folder where the files are saved then do:
```
streamlit run streamer_bert.py
```
This should open up a webui on your machine where you can input lyrics and see the sentiment.

## Contributors
- [Josue Cortez](https://github.com/jgcortez)
- [Pratham Gupta](https://github.com/prathamgupta36)
- [Shashank Navad](https://github.com/shashnavad)
- [Ryan Skabelund](https://github.com/ryan-skabelund)
- [Race Musgrave](https://github.com/R-a-c-e)
- [Tanooj Reddy Seelam](https://github.com/TanoojSeelam)

## Acknowledgements
- Arizona State University, School of Computing and Augmented Intelligence
- TensorFlow, Hugging Face for the initial model frameworks
- Google MediaPipe for the enhanced processing tools

## References
- [MediaPipe](https://google.github.io/mediapipe/)
- [TensorFlow](https://www.tensorflow.org/)
- [Transformers by Hugging Face](https://huggingface.co/transformers/)
