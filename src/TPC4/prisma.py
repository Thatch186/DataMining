import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class PRISM:
    def __init__(self):
        self.rules = []

    def fit(self, X, y):
        self.rules = []
        discrete_mask = self._get_discrete_mask(X)
        while len(np.unique(y)) > 1:
            best_rule, best_score = None, -np.inf
            for feature_idx, is_discrete in enumerate(discrete_mask):
                if is_discrete:
                    unique_values = np.unique(X[:, feature_idx])
                    covered = np.equal.outer(X[:, feature_idx], unique_values)
                    for i, value in enumerate(unique_values):
                        rule = (feature_idx, value)
                        score = self._evaluate_rule(X, y, rule, covered[:, i])
                        if score > best_score:
                            best_rule, best_score = rule, score
                else:
                    unique_values = np.unique(X[:, feature_idx])
                    for i, value in enumerate(unique_values):
                        covered = X[:, feature_idx] <= value
                        rule = (feature_idx, value, True)
                        score = self.evaluate_rule(X, y, rule, covered)
                        if score > best_score:
                            best_rule, best_score = rule, score
                        covered = X[:, feature_idx] > value
                        rule = (feature_idx, value, False)
                        score = self.evaluate_rule(X, y, rule, covered)
                        if score > best_score:
                            best_rule, best_score = rule, score
            if best_rule is None:
                break
            self.rules.append(best_rule)
            X, y = self._remove_covered_examples(X, y, best_rule)

    def predict(self, X):
        predictions = np.zeros(X.shape[0], dtype=int)
        for rule in self.rules:
            feature_idx, value = rule[:2]
            if len(rule) == 3:
                is_less_than_or_equal = rule[2]
                if is_less_than_or_equal:
                    covered = X[:, feature_idx] <= value
                else:
                    covered = X[:, feature_idx] > value
            else:
                covered = X[:, feature_idx] == value
            if np.any(covered):
                target_covered = self._get_target_covered(covered)
                most_common, _ = mode(target_covered, keepdims=True)
                predictions[covered] = most_common[0]

        return predictions

    def _get_discrete_mask(self, X):
        discrete_mask = []
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            discrete_mask.append(len(unique_values) <= int(np.sqrt(X.shape[0])))
        return discrete_mask

    def evaluate_rule(self, X, y, rule, covered):

        target_covered = y[covered]

        if len(target_covered) == 0:
            return 0.0

        most_common, _ = mode(target_covered, nan_policy='omit')
        correct = np.sum(target_covered == most_common)

        return correct / len(target_covered)

    def _remove_covered_examples(self, X, y, rule):
        feature_idx, value = rule[:2]
        if len(rule) == 3:
            is_less_than_or_equal = rule[2]
            if is_less_than_or_equal:
                not_covered = X[:, feature_idx] > value
            else:
                not_covered = X[:, feature_idx] <= value
        else:
            not_covered = X[:, feature_idx] != value
        return X[not_covered], y[not_covered]

    def _get_target_covered(self, covered):
            df_covered = pd.DataFrame({'covered': covered})
            most_common = df_covered.mode()['covered'].iloc[0]
            return most_common


    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


if __name__ == '__main__':
    # Create or load your dataset using pandas
    df = pd.read_csv("src/TPC4/iris.csv")  # Replace with the path to your dataset

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert class column to numeric values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Create an instance of the PRISM classifier
    prism = PRISM()

    # Fit the PRISM classifier to the training set
    prism.fit(X_train, y_train)

    # Generate predictions on the test set
    predictions_encoded = prism.predict(X_test)

    # Convert predictions back to original class labels
    predictions = label_encoder.inverse_transform(predictions_encoded)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)