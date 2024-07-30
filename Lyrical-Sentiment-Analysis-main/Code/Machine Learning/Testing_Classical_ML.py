import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import label_binarize
from collections import Counter

def predict_sentiment(lyrics):
    # Load the trained model and vectorizer
    model = joblib.load('Data/Model/classical_ml_model.pkl')
    vectorizer = joblib.load('Data/Model/classical_ml_tfidf_vectorizer.pkl')
    
    # Transform the lyrics using the loaded vectorizer
    lyrics_tfidf = vectorizer.transform([lyrics])
    
    # Predict sentiment using the loaded model
    prediction = model.predict(lyrics_tfidf)
    
    return prediction[0]

def load_and_predict_new_data(file_path, filter_column=None, filter_value=None):
    # Load the new data
    new_data = pd.read_csv(file_path)

    # Filter data if filter_column and filter_value are specified
    if filter_column and filter_value:
        new_data = new_data[new_data[filter_column] == filter_value]

    # Remove rows where the Label is 'Neutral'
    new_data = new_data[new_data['Label'] != 'Neutral']
    
    # Load the trained model and vectorizer
    model = joblib.load('Data/Model/classical_ml_model.pkl')
    vectorizer = joblib.load('Data/Model/classical_ml_tfidf_vectorizer.pkl')
    
    # Transform the new data using the loaded vectorizer
    new_data_tfidf = vectorizer.transform(new_data['Lyrics'])
    
    # Predict sentiment
    predictions = model.predict(new_data_tfidf)
    predicted_probs = model.predict_proba(new_data_tfidf)[:, 1]
    
    return new_data, predictions, predicted_probs

def calculate_metrics(new_data, predictions, predicted_probs):
    # Mapping predictions back to numeric if necessary
    label_map = {'Negative': 0, 'Positive': 1}
    numeric_predictions = np.array([label_map[pred] for pred in predictions])

    # True labels are already assumed to be numeric based on the map you use in predictions
    true_labels = new_data['Label'].map({'Negative': 0, 'Positive': 1}).values

    # Calculate Accuracy
    accuracy = accuracy_score(true_labels, numeric_predictions)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(true_labels, predicted_probs)
    
    # Calculate Precision, Recall, F1-Score for each class and weighted
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, numeric_predictions, average=None, labels=[0, 1], zero_division=1)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(true_labels, numeric_predictions, average='weighted', zero_division=1)
    
    # Calculate Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, numeric_predictions)
    
    # Print Metrics
    print(f"\nAccuracy: {round(accuracy, 2)}")
    print(f"ROC AUC Score: {round(roc_auc, 2)}")
    print("Negative: Precision: {}, Recall: {}, F1-Score: {}".format(round(precision[0], 2), round(recall[0], 2), round(f1[0], 2)))
    print("Positive: Precision: {}, Recall: {}, F1-Score: {}".format(round(precision[1], 2), round(recall[1], 2), round(f1[1], 2)))
    print("Weighted: Precision: {}, Recall: {}, F1-Score: {}".format(round(weighted_precision, 2), round(weighted_recall, 2), round(weighted_f1, 2)))
    print(f"Confusion Matrix:\n{conf_matrix}")

def count_predictions(predictions, column_name=None, column_value=None):
    if column_name and column_value:
        print(f"\nCalculating Positives & Negatives for {column_name} = {column_value}")
    else:
        print(f"\nCalculating Positives & Negatives")

    # Count occurrences of each label in the predictions array
    counts = Counter(predictions)
    total_predictions = len(predictions)

    # Print the count for each label
    print(f"Total Amount of Predictions: {total_predictions}")
    print(f"Total 'Positive' predictions: {counts['Positive']} ({round(100*counts['Positive']/total_predictions, 2)}%)")
    print(f"Total 'Negative' predictions: {counts['Negative']} ({round(100*counts['Negative']/total_predictions, 2)}%)")



# Path to the Validating dataset CSV file
file_path = 'Data/Cleaned/testCleanedLabeled.csv'
columnNameFilter = 'Top100Year'
columnValueFilterArray = range(2014, 2024)

# Evaluate whole tresting data set.
new_data, predictions, predicted_probs = load_and_predict_new_data(file_path)
count_predictions(predictions)
calculate_metrics(new_data, predictions, predicted_probs)

# Predict data for filtered values
for columnValueFilter in columnValueFilterArray:
    new_data, predictions, predicted_probs = load_and_predict_new_data(file_path, filter_column=columnNameFilter, filter_value=columnValueFilter)
    count_predictions(predictions, columnNameFilter, str(columnValueFilter))
    # calculate_metrics(new_data, predictions, predicted_probs)


# Example Positive Lyrics
lyrics = "Well met couple hours ago last night town hey would not know would get hooked girl blue diamond eyes Mexico oh walking asking dance Smilin' smile reaching hand Well move two would like show still got chance got soul know use Put hand hip know lose got heart racing like nothing Fallin' love beat music Oh-oh would not want stay Yeah Oh-oh let plane fly away away away got soul know use Put hand hip know lose got heart racing like nothing Fallin' love beat music beat music think could get used steel drum playing Wakin' beach know saying One night would alright hold baby got soul know use Put hand hip know lose got heart racing like nothing Falling love beat music falling love beat music Ooh-ooh Beat music Yay-ay-ay Yay-ay-ay ay Yay-ay-ay"
lyrics = "see wearing nothing beneath Forgive staring forgive breathing might not know might not know baby tonight beautiful beautiful beautiful might not know might not know baby tonight beautiful light sky open clouds baby tonight beautiful beautiful beautiful Wherever going going chase left moment not going waste Stranded together worlds collided not forever try fight beautiful beautiful might not know might not know baby tonight beautiful light sky open clouds baby tonight beautiful beautiful beautiful Let us live tonight like fireflies one one light sky disappear pass crown beautiful beautiful beautiful beautiful beautiful beautiful beautiful beautiful beautiful beautiful beautiful beautiful beautiful beautiful"
lyrics = "Diamond rings old One's queens one's fools One's future one's past One's forever one not last not like midnight cigarette smoke not like watered whiskey coke guess things not mix like hoped Like diamond rings old wrongs rights highs lows ""I love you's "" ""I told Past miles wherever home Another morning waking alone not like midnight cigarette smoke not like watered whiskey coke guess things not mix like hoped Like diamond rings old not like midnight cigarette smoke Nothing like watered whiskey coke guess things not mix like hoped Like diamond rings old"
lyrics = "young first saw close eyes flashback starts standing balcony summer air See lights see party ball gowns See make way crowd say 'Hello' Little know Romeo throwing pebbles daddy said 'Stay away Juliet' crying staircase Begging 'Please not got said Romeo take somewhere alone waiting left run prince princess love story baby say 'Yes' sneak garden see keep quiet dead knew close eyes Escape town little oh oh Romeo scarlet letter daddy said 'Stay away Juliet' everything begging 'Please not got said Romeo take somewhere alone waiting left run prince princess love story baby say 'Yes' Romeo save trying tell feel love difficult real not afraid make mess love story baby say 'Yes' Oh oh got tired waiting Wondering ever coming around faith fading met outskirts town said 'Romeo save feeling alone keep waiting never come head not know think knelt ground pulled ring said 'Marry Juliet never alone love really know talked dad go pick white dress love story baby say 'Yes'' Oh oh oh Oh oh oh young first saw"

# Predict and print the sentiment
sentiment = predict_sentiment(lyrics)
print(f"\nPredicted Sentiment: {sentiment}")
