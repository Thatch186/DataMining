from collections import Counter
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.feature_probs = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = np.zeros(len(self.classes))
        self.feature_probs = []

        # Compute class priors
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.sum(y == c) / len(y)

        # Compute feature probabilities
        for i in range(X.shape[1]):
            feature_values = np.unique(X[:, i])
            feature_probs = []
            for c in self.classes:
                class_indices = np.where(y == c)[0]
                feature_counts = Counter(X[class_indices, i])
                feature_probs.append({val: (count + 1) / (len(class_indices) + len(feature_values)) for val, count in feature_counts.items()})
            self.feature_probs.append(feature_probs)

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = []
            for i, c in enumerate(self.classes):
                score = np.log(self.class_priors[i])
                for j, feature in enumerate(sample):
                    score += np.log(self.feature_probs[j][i][feature])
                class_scores.append(score)
            predictions.append(self.classes[np.argmax(class_scores)])
        return predictions


if __name__ == '__main__':
    # Create a sample dataset
    # Sample training data
    X_train = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
                        [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
                        [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    y_train = np.array(['No', 'No', 'Yes', 'Yes', 'No',
                        'No', 'No', 'Yes', 'Yes', 'Yes',
                        'Yes', 'Yes', 'Yes', 'Yes', 'No'])

    # Sample test data
    X_test = np.array([[2, 'S'], [1, 'M'], [3, 'L']])

    # Instantiate and fit Naive Bayes classifier
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # Predict class labels for test data
    predictions = nb.predict(X_test)

    # Print the predictions
    print(predictions)  # Output: ['No', 'Yes', 'Yes']