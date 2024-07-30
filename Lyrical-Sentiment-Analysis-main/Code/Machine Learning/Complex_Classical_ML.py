import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Load the data
df = pd.read_csv('Data/Cleaned/completeCleanedLabeled.csv')

# Remove rows where the Label is 'Neutral'
df = df[df['Label'] != 'Neutral']

def train_and_evaluate(X, y, random_state, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_df=0.95, min_df=5)
    
    class_labels = np.unique(y)
    metrics = {label: {'precision': [], 'recall': [], 'f1': []} for label in class_labels}
    weighted_metrics = {'precision': [], 'recall': [], 'f1': []}
    auc_scores = []
    accuracies = []
    confusion_matrices = []


    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LogisticRegression(random_state=random_state)
        param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
        grid_search = GridSearchCV(model, param_grid, cv=5, verbose=0)
        grid_search.fit(X_train_tfidf, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test_tfidf)
        y_prob = best_model.predict_proba(X_test_tfidf)[:, 1]

        accuracies.append(accuracy_score(y_test, y_pred))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        prf_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        weighted_metrics['precision'].append(prf_weighted[0])
        weighted_metrics['recall'].append(prf_weighted[1])
        weighted_metrics['f1'].append(prf_weighted[2])
        if len(np.unique(y)) == 2:
            auc_scores.append(roc_auc_score(y_test, y_prob))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=class_labels)
        for idx, label in enumerate(class_labels):
            metrics[label]['precision'].append(prec[idx])
            metrics[label]['recall'].append(rec[idx])
            metrics[label]['f1'].append(f1[idx])

    avg_metrics = {label: {} for label in class_labels}
    for label, scores in metrics.items():
        avg_metrics[label]['precision'] = np.mean(scores['precision'])
        avg_metrics[label]['recall'] = np.mean(scores['recall'])
        avg_metrics[label]['f1'] = np.mean(scores['f1'])
    
    avg_accuracy = np.mean(accuracies)
    avg_auc = np.mean(auc_scores) if auc_scores else None
    avg_conf_matrix = np.sum(confusion_matrices, axis=0)
    avg_weighted_metrics = {k: np.mean(v) for k, v in weighted_metrics.items()}


    return avg_accuracy, avg_auc, avg_metrics, avg_weighted_metrics, avg_conf_matrix

def evaluate_model_multiple_random_states(X, y, num_runs, base_random_state):
    results = []
    for i in range(num_runs):
        accuracy, auc, metrics, weighted_metrics, conf_matrix = train_and_evaluate(X, y, random_state=base_random_state+i, n_splits=5)
        results.append({'accuracy': accuracy, 'auc': auc, 'metrics': metrics, 'weighted_metrics': weighted_metrics, 'conf_matrix': conf_matrix})
        print(f"Run {i+1}:")
        print(f"Accuracy: {round(accuracy, 2)}, ROC AUC Score: {round(auc, 2) if auc is not None else 'N/A'}")
        for label, scores in metrics.items():
            print(f"{label}: Precision: {round(scores['precision'], 2)}, Recall: {round(scores['recall'], 2)}, F1-Score: {round(scores['f1'], 2)}")
        print(f"Weighted: Precision: {round(weighted_metrics['precision'], 2)}, Recall: {round(weighted_metrics['recall'], 2)}, F1-Score: {round(weighted_metrics['f1'], 2)}")
        print(f"Confusion Matrix:\n{conf_matrix}\n")

    # Calculate averages
    avg_accuracy = np.mean([res['accuracy'] for res in results])
    avg_auc = np.mean([res['auc'] for res in results if res['auc'] is not None])
    avg_conf_matrix = np.sum([res['conf_matrix'] for res in results], axis=0)
    avg_metrics = {label: {'precision': np.mean([res['metrics'][label]['precision'] for res in results]),
                           'recall': np.mean([res['metrics'][label]['recall'] for res in results]),
                           'f1': np.mean([res['metrics'][label]['f1'] for res in results])}
                   for label in results[0]['metrics']}
    avg_weighted_metrics = {k: np.mean([res['weighted_metrics'][k] for res in results]) for k in results[0]['weighted_metrics'].keys()}


    print("Average Results Across All Runs:")
    print(f"Average Accuracy: {round(avg_accuracy, 2)}")
    print(f"Average ROC AUC Score: {round(avg_auc, 2) if avg_auc is not None else 'N/A'}")
    for label, scores in avg_metrics.items():
        print(f"{label}: Average Precision: {round(scores['precision'], 2)}, Average Recall: {round(scores['recall'], 2)}, Average F1 Score: {round(scores['f1'], 2)}")
    print(f"Weighted: Average Precision: {round(avg_weighted_metrics['precision'], 2)}, Average Recall: {round(avg_weighted_metrics['recall'], 2)}, Average F1 Score: {round(avg_weighted_metrics['f1'], 2)}")
    print(f"Aggregated Confusion Matrix:\n{avg_conf_matrix}")

# Example usage
evaluate_model_multiple_random_states(df['Lyrics'], df['Label'], num_runs=5, base_random_state=7)
