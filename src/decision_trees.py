import numpy as np
from typing import List

class Node:
    def __init__(self, feature_idx: int, threshold: float, left, right):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = self._grow_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth=0) -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if depth == self.max_depth or n_labels == 1 or n_samples < 2:
            return Node(None, None, None, None)

        best_feature_idx, best_threshold = self._find_best_split(X, y, n_labels)

        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_threshold)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature_idx, best_threshold, left, right)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, n_labels: int) -> List[float]:
        best_feature_idx, best_threshold = None, None
        best_gini = 1.0

        for feature_idx in range(X.shape[1]):
            thresholds, classes = self._thresholds_and_classes(X[:, feature_idx], y)

            for threshold, class_left, class_right in zip(thresholds, classes[:-1], classes[1:]):
                gini = self._gini_index(y, class_left, class_right, n_labels)
                if gini < best_gini:
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_gini = gini

        return best_feature_idx, best_threshold

    def _split(self, feature_column: np.ndarray, threshold: float) -> List[np.ndarray]:
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]

        return left_idxs, right_idxs

    def _thresholds_and_classes(self, feature_column: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        sorted_idxs = np.argsort(feature_column)
        sorted_features = feature_column[sorted_idxs]
        sorted_labels = y[sorted_idxs]

        classes = np.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1
        thresholds = (sorted_features[classes - 1] + sorted_features[classes]) / 2

        return thresholds, classes

    def _gini_index(self, y: np.ndarray, class_left: np.ndarray, class_right: np.ndarray, n_labels: int) -> float:
        n_left, n_right = len(class_left), len(class_right)
        n_total = n_left + n_right

        if n_left == 0 or n_right == 0:
            return 0.0

        p_left = np.array([
            np.count_nonzero(y[class_left] == label) / n_left for label in range(n_labels)
        ])
        p_right = np.array([
            np.count_nonzero(y[class_right] == label) / n_right for label in range(n_labels)
        ])

        gini_left = 1.0 - np.sum(p_left ** 2)
        gini_right = 1.0 - np.sum(p_right ** 2)

        return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right