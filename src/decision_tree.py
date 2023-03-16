from dataset import Dataset
import numpy as np
import math
import copy


class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.tree
            while node['left'] != None and node['right'] != None:
                if sample[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['class'])
        return np.array(predictions)

    # Attribute selection methods
    def __entropy(self, y):
        entropy = 0
        classes = np.unique(y)
        for c in classes:
            p = np.sum(y == c) / len(y)
            entropy += -p * np.log2(p)
        return entropy

    def __gini(self, y):
        gini = 1
        classes = np.unique(y)
        for c in classes:
            p = np.sum(y == c) / len(y)
            gini -= p ** 2
        return gini

    def __gain_ratio(self, X, y):
        def info(D):
            return -np.sum([p * np.log2(p) for p in D if p != 0])

        def split_info(D, Xj):
            n = len(D)
            Dj = [D[X[:, j] == Xj] for Xj in np.unique(X[:, j])]
            return np.sum([(len(Dj[k]) / n) * np.log2(len(Dj[k]) / n) for k in range(len(Dj)) if len(Dj[k]) > 0])

        info_D = info([np.sum(y == c) / len(y) for c in np.unique(y)])
        gain_ratios = []
        for j in range(X.shape[1]):
            Xj = X[:, j]
            A = np.unique(Xj)
            gain = info_D
            split_info_D = split_info([y[X[:, j] == Aj] for Aj in A], A)
            for Aj in A:
                gain -= (np.sum(Xj == Aj) / len(y)) * info([np.sum(y[X[:, j] == Aj] == c) / np.sum(X[:, j] == Aj) for c in np.unique(y)])
            gain_ratio = gain / split_info_D if split_info_D != 0 else 0
            gain_ratios.append(gain_ratio)
        return np.argmax(gain_ratios)

    # Conflict resolution methods
    def __prune(self, tree, X_val, y_val):
        # Recursive pruning
        for val in tree.keys():
            if isinstance(tree[val], dict):
                tree[val] = self.__prune(tree[val], X_val, y_val)

        # Check accuracy before and after pruning
        acc_before = self.score(X_val, y_val)

        # Check if node is a leaf
        if not isinstance(tree, dict):
            return tree

        # Calculate accuracy after replacing subtree with majority class label
        y_pred = []
        for i in range(X_val.shape[0]):
            instance = X_val[i]
            y_pred.append(self.predict_one(instance, tree))
        majority = self.__majority_voting(y_pred)
        tree_acc = (y_pred == y_val).sum() / len(y_val)
        acc_after = tree_acc if tree_acc > majority else majority

        # If accuracy improves, replace subtree with majority class label
        if acc_after >= acc_before:
            return majority

        return tree


    def __majority_voting(self, y):
        unique, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return unique[index]
    
    # Pre-prunning
    def __size_prepruning(self, n_samples):
        if n_samples < self.min_samples_split:
            return True
        return False

    def __maximum_depth_prepruning(self, depth):
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    
   # Post-prunning
    def __pessimistic_error_pruning(self, node, X, y):
        if node is None:
            return

        if node['left'] is not None or node['right'] is not None:
            left_X, left_y, right_X, right_y = self.__split(X, y, node['index'], node['value'])

            self.__pessimistic_error_pruning(node['left'], left_X, left_y)
            self.__pessimistic_error_pruning(node['right'], right_X, right_y)

            left_acc, left_label = self.__predict_node(node['left'], left_X)
            right_acc, right_label = self.__predict_node(node['right'], right_X)

            error = (1 - max(left_acc, right_acc)) / len(X)

            n = len(X)
            delta = np.sqrt((1 - error) * error / n + 0.25 * np.log(2 / 0.05) / n)

            if error + delta <= node['error']:
                node['left'] = None
                node['right'] = None
                node['class'] = left_label if left_acc > right_acc else right_label

                self.__calculate_node_stats(node, X, y)

    def __reduced_error_pruning(self, node, X_val, y_val):
        if node is None:
            return

        if node['left'] is not None or node['right'] is not None:
            left_X, left_y, right_X, right_y = self.__split(X_val, y_val, node['index'], node['value'])

            self.__reduced_error_pruning(node['left'], left_X, left_y)
            self.__reduced_error_pruning(node['right'], right_X, right_y)

            acc_before = self.__accuracy(X_val, y_val)

            node_copy = copy.deepcopy(node)

            node['left'] = None
            node['right'] = None
            node['class'] = self.__majority_voting(y_val)

            self.__calculate_node_stats(node, X_val, y_val)

            acc_after = self.__accuracy(X_val, y_val)

            if acc_after >= acc_before:
                return
            else:
                node['left'] = node_copy['left']
                node['right'] = node_copy['right']
                node['class'] = node_copy['class']
                self.__calculate_node_stats(node, X_val, y_val)

    # Métodos auxiliares
    def _build_tree(self, X, y, depth):
        # Check for leaf node
        if len(set(y)) == 1:
            return y[0]

        # Check for pre-pruning conditions
        if self.size_prepruning is not None and len(X) <= self.size_prepruning:
            return self.__majority_voting(y)

        if self.maximum_depth_prepruning is not None and depth >= self.maximum_depth_prepruning:
            return self.__majority_voting(y)

        # Choose attribute and split data
        best_attr = self.__choose_attribute(X, y)
        tree = {best_attr: {}}

        for val in np.unique(X[:, best_attr]):
            val_X = X[X[:, best_attr] == val]
            val_y = y[X[:, best_attr] == val]

            if len(val_y) == 0:
                tree[best_attr][val] = self.__majority_voting(y)
            else:
                subtree = self._build_tree(val_X, val_y, depth+1)
                tree[best_attr][val] = subtree

        return tree