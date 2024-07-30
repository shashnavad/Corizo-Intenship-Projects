import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_final_model(X, y):
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_df=0.95, min_df=5)
    X_tfidf = vectorizer.fit_transform(X)  # Transform the entire dataset

    # Setup the Logistic Regression Model with a GridSearchCV to find the best parameters
    model = LogisticRegression(random_state=7, max_iter=1000)
    param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1)
    grid_search.fit(X_tfidf, y)

    # Save the trained model and vectorizer to disk for later use
    joblib.dump(grid_search.best_estimator_, 'Data/Model/classical_ml_model.pkl')
    joblib.dump(vectorizer, 'Data/Model/classical_ml_tfidf_vectorizer.pkl')

    print("Model and vectorizer saved successfully!")

# Load the data
df = pd.read_csv('Data/Cleaned/trainCleanedLabeled.csv')

# Remove rows where the Label is 'Neutral' to focus on binary classification
df = df[df['Label'] != 'Neutral']

# Prepare data
X = df['Lyrics']
y = df['Label']

# Train the final model on the full dataset
train_final_model(X, y)
