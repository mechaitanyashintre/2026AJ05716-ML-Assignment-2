"""verify_project.py - checks the trained models and saved evaluation scores.

Run after train_models.py: python verify_project.py
It opens model/*.joblib, test_data.csv and results/metrics.csv.
It predicts using the saved models and confirms that all six scores match.
This file does not retrain any models or download the dataset.
"""
from pathlib import Path  # Find the project folder and saved files.
import json  # Read the saved metadata.

import joblib  # Load the saved trained models.
import pandas as pd  # Read test data and stored scores.
from sklearn.metrics import (  # Import the six evaluation metrics.
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parent  # Project folder.
INFO_FILE = PROJECT_DIR / 'model/model_info.json'  # JSON file describing saved models.
TEST_FILE = PROJECT_DIR / 'test_data.csv'  # Held-out test CSV.
METRICS_FILE = PROJECT_DIR / 'results/metrics.csv'  # Scores saved during training.

for path in [INFO_FILE, TEST_FILE, METRICS_FILE]:  # Check that the required output files exist.
    if not path.exists():
        raise FileNotFoundError(f'{path.name} missing. First run download_dataset.py and train_models.py.')

model_details = json.loads(INFO_FILE.read_text(encoding='utf-8'))  # Read the original model settings.
test_dataset = pd.read_csv(TEST_FILE)  # Open the labelled test data.
true_labels = test_dataset['target'].astype(int)  # Get the actual target labels.
X = test_dataset[model_details['feature_names']]  # Select inputs in the same order used for training.
saved_metrics = pd.read_csv(METRICS_FILE).set_index('ML Model Name')  # Look up expected scores by model name.

assert X.shape == (2211, 30), f'Expected (2211, 30) test features, got {X.shape}'  # Expect 2,211 records with 30 features.
assert len(model_details['model_files']) == 5, 'Expected 5 classifiers'  # Confirm that all five models were saved.
assert {'0', '1'} == set(model_details['class_names']), 'Unexpected target coding'  # Confirm both target labels are documented.

for name, filename in model_details['model_files'].items():  # Check the saved models one by one.
    model = joblib.load(PROJECT_DIR / 'model' / filename)  # Load a fitted pipeline from disk.
    predicted_labels = model.predict(X)  # Predict labels on the test data.
    # Get probability scores for phishing (class 1).
    phishing_probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    recalculated_metrics = {  # Recalculate the six metrics.
        'Accuracy': accuracy_score(true_labels, predicted_labels),  # Check overall accuracy.
        'AUC': roc_auc_score(true_labels, phishing_probabilities),  # Check ROC-AUC from class probabilities.
        'Precision': precision_score(true_labels, predicted_labels, zero_division=0),  # Check precision for phishing.
        'Recall': recall_score(true_labels, predicted_labels, zero_division=0),  # Check recall for phishing.
        'F1': f1_score(true_labels, predicted_labels, zero_division=0),  # Check the F1 score.
        'MCC': matthews_corrcoef(true_labels, predicted_labels),  # Check MCC using the full confusion matrix.
    }
    for key, value in recalculated_metrics.items():  # Compare recalculated scores with metrics.csv.
        # Allow for rounding in saved decimal values.
        assert abs(value - float(saved_metrics.loc[name, key])) < 0.000002, f'{name}: {key} differs from metrics.csv'
    print('PASS:', name)  # Display PASS after this model matches.

# Display final confirmation after all models pass.
print('\nPASS: all five models, all six metrics, all 2,211 test records.')
