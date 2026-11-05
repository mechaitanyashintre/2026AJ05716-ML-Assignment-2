"""download_dataset.py - prepares dataset.csv for model training.

Run this first: python download_dataset.py
train_models.py also calls prepare() automatically if dataset.csv is missing.
This file reads a local ARFF copy if available, otherwise downloads from UCI.
It converts the target to 0 (legitimate) and 1 (phishing), then saves dataset.csv.
"""

from pathlib import Path  # Read file paths without hardcoding a Windows folder.
import pandas as pd  # Read and save datasets as tables.

PROJECT_DIR = Path(__file__).resolve().parent  # Project folder.
DATASET_FILE = PROJECT_DIR / 'dataset.csv'  # Where the prepared CSV is saved.
DATASET_URL = 'https://archive.ics.uci.edu/dataset/327/phishing'


def load_from_arff(arff_path):  # Alternative loader when UCI download is unavailable.
    """Read the original UCI ARFF file if it is already downloaded."""
    from scipy.io import arff  # Read the original ARFF file.

    records, _ = arff.loadarff(arff_path)  # Get the records from the ARFF file.
    website_data = pd.DataFrame(records)  # Convert records into a table.
    # Values are stored as bytes in some ARFF files, e.g. b'-1'.
    for column in website_data.columns:  # Convert each original column into numbers.
        website_data[column] = pd.to_numeric(  # Convert text values to numeric values.
            website_data[column].map(lambda value: value.decode('utf-8') if isinstance(value, bytes) else value),
            errors='raise',
        )
    return website_data.drop(columns=['Result']), website_data['Result']  # Separate inputs from the original class column.


def load_from_uci():  # Download directly from UCI.
    """Use the official UCI client (requires internet just for this step)."""
    from ucimlrepo import fetch_ucirepo

    uci_dataset = fetch_ucirepo(id=327)  # 327 is the UCI Phishing Websites dataset ID.
    X = uci_dataset.data.features.copy()  # Keep the 30 feature columns.
    y = uci_dataset.data.targets.iloc[:, 0].copy()  # Read the original target column.
    return X, y


def prepare():  # Prepare the dataset once.
    if DATASET_FILE.exists():  # Reuse the existing CSV if it is already available.
        print('Existing dataset.csv found; keeping the same original data.')
        return DATASET_FILE

    # Offline backup for a restricted virtual lab.
    # Look for an offline ARFF copy in either folder.
    offline_dataset_paths = [PROJECT_DIR / 'data' / 'Training Dataset.arff', PROJECT_DIR / 'Training Dataset.arff']
    local_arff = next((file for file in offline_dataset_paths if file.exists()), None)  # Use the first ARFF file that exists.

    if local_arff:  # Prefer the local file when one is found.
        X, original_labels = load_from_arff(local_arff)
        print('Source: original ARFF file:', local_arff)
    else:  # Download from UCI if no local file is present.
        print('Retrieving UCI Phishing Websites dataset (ID 327)...')
        try:  # Catch download problems and explain the fallback.
            X, original_labels = load_from_uci()
            print('Source: UCI ucimlrepo client')
        except Exception as error:
            raise RuntimeError(
                'Could not download from UCI. Visit '+ DATASET_URL +
                ', download the dataset, extract Training Dataset.arff, put it in '
                'data/, then rerun this script.'
            ) from error

    # The real dataset has 30 input features and 11,055 records.
    X = X.copy()  # Work on a copy of the original features.
    X.columns = [str(column).strip() for column in X.columns]  # Trim spaces from column names.
    X = X.apply(pd.to_numeric, errors='raise')  # Check that feature values are numeric.
    source_labels = pd.to_numeric(original_labels, errors='raise')  # Read the target values as numbers too.

    if X.shape != (11055, 30):  # The original data should have 11,055 rows and 30 features.
        raise ValueError(f'Unexpected UCI dimensions {X.shape}; expected (11055, 30).')
    if set(source_labels.unique()) != {-1, 1}:  # The original classes must be -1 and 1.
        raise ValueError(f'Unexpected target labels: {sorted(source_labels.unique())}; expected -1, 1.')
    if X.isna().any().any() or source_labels.isna().any():  # Stop if any values are missing.
        raise ValueError('Missing input/target values found; check your original dataset.')
    if not set(pd.unique(X.to_numpy().ravel())).issubset({-1, 0, 1}):  # Feature values may be -1, 0 or 1.
        raise ValueError('Unexpected feature encoding; expected -1, 0, 1.')

    # Original UCI Result: -1 = phishing, +1 = legitimate.
    # For metric calculations, make phishing the POSITIVE (1) class.
    y = (source_labels == -1).astype(int)  # Change phishing (-1) to class 1 and legitimate (+1) to class 0.
    prepared_data = X.copy()  # Start the output table with the feature columns.
    prepared_data['target'] = y.to_numpy()  # Add the converted target column.
    prepared_data.to_csv(DATASET_FILE, index=False)  # Save the table without a separate row-number column.

    print('Created:', DATASET_FILE)  # Show the location of the saved file.
    print('Records:', len(prepared_data), '| Features:', X.shape[1])
    print('Target: 1 = phishing, 0 = legitimate')
    print('Class totals:', y.value_counts().sort_index().to_dict())  # Show how many records belong to each class.
    return DATASET_FILE  # Return the output CSV path.


if __name__ == '__main__':  # Run this part only when executing this file directly.
    prepare()
