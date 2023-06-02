from typing import Callable

import numpy as np


from dataset import Dataset
from f_classif import f_classification



class SelectKBest:
    """
    Select features according to the k highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    k: int, default=10
        Number of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        Select features according to the k highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.
        """
        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        It fits SelectKBest to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
        A labeled dataset

        Returns
        -------
        dataset: Dataset
        A labeled dataset with the k highest scoring features.
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        discrete_features = [feat for feat in features if feat in dataset.get_discrete_mask()]
        numeric_features = [feat for feat in features if feat not in dataset.get_discrete_mask()]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), discrete_features=discrete_features, numeric_features=numeric_features, label=dataset.label)


    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':
    # Create a sample dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    features = ['feature1', 'feature2', 'feature3']
    
    # Specify the numeric features
    numeric_features = ['feature1', 'feature2', 'feature3']
    
    # Create an instance of Dataset with the specified numeric features
    dataset = Dataset(X=X, y=y, features=features, numeric_features=numeric_features)

    # Create an instance of SelectKBest
    k_best = SelectKBest(k=2)

    # Fit and transform the dataset
    transformed_dataset = k_best.fit_transform(dataset)

    # Print the selected features and transformed data
    print("Selected Features:", transformed_dataset.get_features())
    print("Transformed Data:\n", transformed_dataset.get_X())
