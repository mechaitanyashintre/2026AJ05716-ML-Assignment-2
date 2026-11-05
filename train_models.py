"""train_models.py - trains the five classifiers and saves their results.

Run after dataset preparation: python train_models.py
This script calls download_dataset.prepare() before reading dataset.csv.
It splits data into train/test sets, trains all five models, and saves them.
Outputs: test_data.csv, model/*.joblib, model/model_info.json and results/ files.
Run this again whenever the random seed or model settings are changed.
"""

from pathlib import Path  # Work with project folders and file names.
import json  # Store the model details in JSON format.

import joblib  # Save trained models to disk.
import pandas as pd  # Read data and build result tables.
from sklearn.impute import SimpleImputer  # Fill missing values if needed.
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier.
from sklearn.metrics import (  # Scoring functions for model evaluation.
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # Make the train/test split.
from sklearn.naive_bayes import GaussianNB  # Naive Bayes classifier.
from stable_knn import StableKNNClassifier  # KNN with fixed tie-breaking for equally close rows.
from sklearn.pipeline import Pipeline  # Combine preprocessing and model into one pipeline.
from sklearn.preprocessing import StandardScaler  # Standardize features for distance-sensitive models.
from sklearn.svm import SVC  # Support Vector Machine classifier.
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier.

from download_dataset import prepare  # Reuse the dataset preparation function.

PROJECT_DIR = Path(__file__).resolve().parent  # Folder containing these scripts.
MODEL_DIR = PROJECT_DIR / 'model'  # Folder for saved models.
RESULT_DIR = PROJECT_DIR / 'results'  # Folder for metric results.
MODEL_DIR.mkdir(exist_ok=True)  # Create the model folder if needed.
RESULT_DIR.mkdir(exist_ok=True)  # Create the result folder if needed.
RANDOM_SEED = 565  # Use a fixed seed based on our course code (AIMLZG565).


def scaled_model(classifier):  # Pipeline with scaling for Logistic Regression, KNN and SVM.
    # StandardScaler is fitted on TRAIN rows, not the test rows.
    return Pipeline([  # Run these steps in the given order.
        ('missing_values', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most common value.
        ('scaler', StandardScaler()),  # Scale features using training-set statistics.
        ('classifier', classifier),  # Fit the classifier after preprocessing.
    ])


def basic_model(classifier):  # Pipeline without scaling for Tree and Naive Bayes.
    # Tree and Naive Bayes do not require feature scaling in this example.
    return Pipeline([  # Handle missing values the same way.
        ('missing_values', SimpleImputer(strategy='most_frequent')),
        ('classifier', classifier),  # Add the chosen classifier.
    ])


def get_models():  # Set up the five models required in the assignment.
    # Simple settings rather than automatic hyperparameter search.
    return {  # Each model has a filename and a pipeline.
        # Allow more iterations for convergence.
        'Logistic Regression': ('logistic_regression.joblib', scaled_model(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))),
        # Limit tree depth to 8.
        'Decision Tree': ('decision_tree.joblib', basic_model(DecisionTreeClassifier(max_depth=8, random_state=RANDOM_SEED))),
        'KNN': ('knn.joblib', scaled_model(StableKNNClassifier(n_neighbors=5))),  # Five neighbors, fixed tie-breaking.
        'Naive Bayes': ('naive_bayes.joblib', basic_model(GaussianNB())),  # Gaussian probability model.
        # RBF SVM; probabilities are needed for AUC.
        'Support Vector Machine': ('svm.joblib', scaled_model(SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_SEED))),
    }


def evaluate(model, features, actual):  # Calculate the six scores on the test set.
    predicted = model.predict(features)  # Predict whether each record is phishing or legitimate.
    phishing_index = list(model.classes_).index(1)  # Find the probability column for class 1 (phishing).
    phishing_probabilities = model.predict_proba(features)[:, phishing_index]  # Take the phishing probabilities for AUC.
    return {
        'Accuracy': accuracy_score(actual, predicted),  # Proportion of correct predictions.
        'AUC': roc_auc_score(actual, phishing_probabilities),  # Area under the ROC curve.
        # Correct phishing alerts divided by all phishing alerts.
        'Precision': precision_score(actual, predicted, pos_label=1, zero_division=0),
        # Detected phishing cases divided by actual phishing cases.
        'Recall': recall_score(actual, predicted, pos_label=1, zero_division=0),
        'F1': f1_score(actual, predicted, pos_label=1, zero_division=0),  # Balance between precision and recall.
        'MCC': matthews_corrcoef(actual, predicted),  # Correlation score using all four confusion-matrix counts.
    }, predicted


def main():  # Main training steps.
    prepare()  # Create the dataset if it is not already there.
    website_data = pd.read_csv(PROJECT_DIR / 'dataset.csv')  # Open the prepared CSV.
    if 'target' not in website_data.columns or website_data.shape != (11055, 31):  # Check rows, columns and the target.
        raise ValueError('dataset.csv must have 11,055 rows, 30 features and target.')

    X = website_data.drop(columns=['target'])  # X contains the 30 input features.
    y = website_data['target'].astype(int)  # y contains the actual labels.
    if set(y.unique()) != {0, 1}:  # Both classes should be present.
        raise ValueError('target must contain both 0 (legitimate) and 1 (phishing).')

    # All five algorithms use exactly the same 80/20 STRATIFIED split.
    X_train, X_test, y_train, y_test = train_test_split(  # Keep 20% for testing, with similar class proportions.
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y,
    )
    test_data = X_test.copy()  # Copy the test features for Streamlit.
    test_data['target'] = y_test  # Include the actual labels in the test CSV.
    test_data.to_csv(PROJECT_DIR / 'test_data.csv', index=False)  # Save the labelled test data.

    classifiers = get_models()  # Get all five model pipelines.
    metric_records = []  # Collect metrics for the comparison table.
    for name, (filename, model) in classifiers.items():  # Train and test each model using the same split.
        # Fitting happens ONCE on training rows only.
        model.fit(X_train, y_train)  # Train the pipeline on training data.
        scores, predicted = evaluate(model, X_test, y_test)  # Evaluate on the held-out test set.
        metric_records.append({'ML Model Name': name, **scores})  # Save this model's six scores in the table.
        joblib.dump(model, MODEL_DIR / filename)  # Store the fitted pipeline for Streamlit.
        print('\n' + name)
        print('Confusion matrix (0=legitimate, 1=phishing):')
        print(confusion_matrix(y_test, predicted, labels=[0, 1]))  # Print the actual-vs-predicted counts.

    scores = pd.DataFrame(metric_records)  # Make a results table from all five models.
    scores.to_csv(RESULT_DIR / 'metrics.csv', index=False, float_format='%.6f')  # Save the scores to a CSV.

    # The app reads the real feature names and saved-model mapping from this file.
    model_details = {  # Store information needed by app.py.
        'dataset_name': 'UCI Phishing Websites (ID 327)',
        'dataset_url': 'https://archive.ics.uci.edu/dataset/327/phishing',
        'feature_names': list(X.columns),  # Keep the exact feature order for later predictions.
        'target_column': 'target',  # Name of the label column in test_data.csv.
        'class_names': {'0': 'Legitimate', '1': 'Phishing'},  # Meaning of the two target values.
        'training_rows': len(X_train),
        'test_rows': len(X_test),
        'random_state': RANDOM_SEED,
        'model_files': {name: file for name, (file, _) in classifiers.items()},  # Map each model name to its saved file.
    }
    # Save the model settings in JSON format.
    (MODEL_DIR / 'model_info.json').write_text(json.dumps(model_details, indent=2), encoding='utf-8')

    # Create a Markdown table for the README.
    header = '| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |\n'
    header += '|---|---:|---:|---:|---:|---:|---:|\n'
    # Format the metric values to four decimal places.
    body = ''.join('| ' + str(row['ML Model Name']) + ' | ' + ' | '.join(
        f'{row[col]:.4f}' for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    ) + ' |\n' for _, row in scores.iterrows())
    # Save the formatted table as a text file.
    (RESULT_DIR / 'metrics_table.md').write_text(header + body, encoding='utf-8')

    best_classifier = scores.sort_values(['F1', 'MCC'], ascending=False).iloc[0]  # Rank models by F1, then MCC if tied.
    print('\nSIX METRICS ON THE SAME UNSEEN TEST DATA:')  # Print the results and winner.
    print(scores.to_string(index=False, float_format=lambda number: f'{number:.4f}'))
    print('\nWinner by test F1 (MCC as tie-breaker):', best_classifier['ML Model Name'])
    print('Ready: model/, results/, test_data.csv')


if __name__ == '__main__':  # Start training when this script is run directly.
    main()
