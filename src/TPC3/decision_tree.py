import numpy as np
from scipy.stats import chi2_contingency

class DecisionTree:
    def __init__(self, criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None,
                 max_leaf_nodes=None, class_threshold=0.5, pre_pruning=None, post_pruning=None, min_size=None):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_threshold = class_threshold
        self.pre_pruning = pre_pruning
        self.post_pruning = post_pruning
        self.min_size = min_size
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0, n_nodes=0)

    def _predict_sample(self, x, node):
        if 'label' in node:
            return node['label']

        if x[node['feature_idx']] < node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
        predictions = [self._predict_sample(x, self.tree) for x in X]
        return np.array(predictions)


    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _gain_ratio(self, gain, y, y_left, y_right):
        split_info = -((len(y_left) / len(y)) * np.log2(len(y_left) / len(y)) + (len(y_right) / len(y)) * np.log2(len(y_right) / len(y)))
        gain_ratio = gain / split_info
        return gain_ratio

    def _best_split(self, X, y):
        best_value = 0
        best_feature_idx = -1
        best_threshold = None
        n_features = X.shape[1]

        if self.criterion == 'entropy':
            impurity = self._calculate_entropy(y)
        elif self.criterion == 'gini':
            impurity = self._calculate_gini(y)
        else:
            raise ValueError(f"Invalid criterion '{self.criterion}', use 'entropy' or 'gini'")

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            for threshold in np.unique(feature_values):
                mask = feature_values < threshold
                y_left = y[mask]
                y_right = y[~mask]

                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                left_impurity = self._calculate_entropy(y_left) if self.criterion == 'entropy' else self._calculate_gini(y_left)
                right_impurity = self._calculate_entropy(y_right) if self.criterion == 'entropy' else self._calculate_gini(y_right)
                weighted_impurity = (len(y_left) * left_impurity + len(y_right) * right_impurity) / len(y)

                gain = impurity - weighted_impurity

                if self.splitter == 'gain_ratio':
                    value = self._gain_ratio(gain, y, y_left, y_right)
                else:
                    value = gain

                if value > best_value:
                    best_value = value
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, depth, n_nodes):
        # Condição de paragem para criação de um nó folha
        if depth == self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split or n_nodes == self.max_leaf_nodes or (self.min_size is not None and len(y) < self.min_size):
            majority_class = self._majority_voting_with_threshold(y)
            return {'label': majority_class if majority_class is not None else np.argmax(np.bincount(y))}
        
        feature_idx, threshold = self._best_split(X, y)
        
        if threshold is None:
            return {'label': np.argmax(np.bincount(y))}
        
        if self.pre_pruning == 'independence':
            p_value = self._chi_squared_test(X, y, feature_idx, threshold)
            if p_value > self.class_threshold:
                return {'label': np.argmax(np.bincount(y))}

        mask = X[:, feature_idx] < threshold
        left = self._build_tree(X[mask], y[mask], depth + 1, n_nodes + 1)
        right = self._build_tree(X[~mask], y[~mask], depth + 1, n_nodes + 1)

        return {'feature_idx': feature_idx, 'threshold': threshold, 'left': left, 'right': right}
                                
    def _majority_voting_with_threshold(self, y):
        class_counts = np.bincount(y)
        max_count = np.max(class_counts)
        majority_class = np.argmax(class_counts)
        return majority_class if max_count / len(y) > self.class_threshold else None

    def _chi_squared_test(self, X, y, feature_idx, threshold):
        contingency_table = np.zeros((2, len(np.unique(y))))
        mask = X[:, feature_idx] < threshold
        for i, class_label in enumerate(np.unique(y)):
            contingency_table[0, i] = np.sum(y[mask] == class_label)
            contingency_table[1, i] = np.sum(y[~mask] == class_label)
        
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        return p_value

    def _reduced_error_pruning(self, node, X, y):
        if 'label' in node:
            return node

        feature_idx = node['feature_idx']
        threshold = node['threshold']
        mask = X[:, feature_idx] < threshold
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        node['left'] = self._reduced_error_pruning(node['left'], X_left, y_left)
        node['right'] = self._reduced_error_pruning(node['right'], X_right, y_right)

        if 'label' in node['left'] and 'label' in node['right']:
            y_pred = self.predict(X)
            node_label = {'label': np.argmax(np.bincount(y))}
            self.tree = node_label
            y_pred_pruned = self.predict(X)

            if np.sum(y_pred != y) >= np.sum(y_pred_pruned != y):
                return node_label

        self.tree = node
        return node
    
    def _pessimistic_error_pruning(self, node, X, y, n):
        if 'label' in node:
            node['error'] = np.sum(y != node['label'])
            return node

        feature_idx = node['feature_idx']
        threshold = node['threshold']
        mask = X[:, feature_idx] < threshold
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        node['left'] = self._pessimistic_error_pruning(node['left'], X_left, y_left, n)
        node['right'] = self._pessimistic_error_pruning(node['right'], X_right, y_right, n)

        node_error = node['left']['error'] + node['right']['error']
        node['error'] = node_error
        leaf_error = np.sum(y != np.argmax(np.bincount(y)))

        if node_error + np.sqrt(node_error / n) >= leaf_error:
            return {'label': np.argmax(np.bincount(y)), 'error': leaf_error}

        return node

    def prune(self, X, y):
        if self.post_pruning == 'reduced_error_pruning':
            self.tree = self._reduced_error_pruning(self.tree, X, y)
        elif self.post_pruning == 'pessimistic_error_pruning':
            self.tree = self._pessimistic_error_pruning(self.tree, X, y)
        else:
            raise ValueError(f"Invalid post_pruning '{self.post_pruning}', use 'reduced_error_pruning' or 'pessimistic_error_pruning'")