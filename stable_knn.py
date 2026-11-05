"""
Stable KNN classifier for handling equal-distance neighbors.

During testing, the original KNN gave slightly different results
between the local testing environment and Streamlit Cloud.

Scikit-learn documents that when the kth and (k+1)th neighbors
have equal distances but different labels, predictions can depend
on the ordering of training data.

To keep results consistent, this implementation uses stable sorting
to select earlier training rows when distances are equal.

Reference:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""

import numpy as np  # Calculate distances and sort neighboring rows.
from sklearn.neighbors import KNeighborsClassifier  # Start from scikit-learn's KNN model.


class StableKNNClassifier(KNeighborsClassifier):
    """Five-neighbor KNN with consistent tie-breaking on all computers."""

    def __init__(self, n_neighbors=5):
        # The original assignment used five neighbors with equal votes.
        super().__init__(n_neighbors=n_neighbors, algorithm='brute')

    def fit(self, X, y):
        # Fit the usual KNN model and remember the training labels.
        super().fit(X, y)
        self.training_labels_ = np.asarray(y)
        return self

    def predict_proba(self, X):
        # Work with the already-fitted training rows. No retraining here.
        test_rows = np.asarray(X, dtype=np.float64)
        training_rows = np.asarray(self._fit_X, dtype=np.float64)
        if test_rows.ndim == 1:
            test_rows = test_rows.reshape(1, -1)
        if test_rows.shape[1] != training_rows.shape[1]:
            raise ValueError('Test data must have the same number of features as training data.')

        probabilities = np.zeros((len(test_rows), len(self.classes_)))
        # Process a few test records at a time to keep memory usage small.
        for start in range(0, len(test_rows), 64):
            batch = test_rows[start:start + 64]
            distances = np.zeros((len(batch), len(training_rows)))

            # Squared Euclidean distance: adding the squared difference
            # of each feature gives the same nearest-neighbor order.
            for column in range(training_rows.shape[1]):
                difference = batch[:, column, None] - training_rows[None, :, column]
                distances += difference * difference

            # Round tiny floating-point differences before sorting.
            # Stable sorting chooses the earlier training row when tied.
            distances = np.round(distances, decimals=10)
            neighbors = np.argsort(distances, axis=1, kind='stable')[:, :self.n_neighbors]
            neighbor_labels = self.training_labels_[neighbors]

            # Each of the five neighbors gives one equal vote.
            for class_index, class_label in enumerate(self.classes_):
                probabilities[start:start + len(batch), class_index] = (
                    neighbor_labels == class_label
                ).mean(axis=1)

        return probabilities

    def predict(self, X):
        # Choose the class receiving most votes (smallest class if tied).
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
