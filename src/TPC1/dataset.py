from typing import Tuple, Sequence

import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, X: np.ndarray, 
                 y: np.ndarray = None, 
                 features: Sequence[str] = None,
                 discrete_features: Sequence[str] = None,
                 numeric_features: Sequence[str] = None, 
                 label: str = None):
        """
        Dataset represents a machine learning tabular dataset.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        discrete_features : list of str (n_features)
            The features names of discrete features
        numeric_features : list of str (n_features)
            The features names of numeric features
        label: str (1)
            The label name
        """

        if X is None:
            raise ValueError("X cannot be None")

        if features is None:
            features = [str(i) for i in range(X.shape[1])]
        else:
            features = list(features)
        if discrete_features is None and numeric_features is None:
            raise ValueError("At least one of discrete_features or numeric_features must be provided")
        elif discrete_features is None:
            self.discrete_mask = np.isin(features, numeric_features)
        elif numeric_features is None:
            self.discrete_mask = np.isin(features, discrete_features, invert=True)
        else:
            self.discrete_mask = np.zeros(X.shape[1], dtype=bool)
            self.discrete_mask[np.isin(features, discrete_features)] = True
            self.discrete_mask[np.isin(features, numeric_features)] = False

        if y is not None and label is None:
            label = "y"

        self.X = X
        self.y = y
        self.features = features
        self.label = label
        self.to_numeric()

        if all(~self.discrete_mask): 
            self.all_numeric = True
        else: 
            self.all_numeric = False

    def to_numeric(self):
        """
        Ensures that numeric features have a numeric type.

        """
        discrete_mask = self.get_discrete_mask()
        if any(~discrete_mask):
            self.X[:, ~discrete_mask] = self.X[:, ~discrete_mask].astype(np.float)

    def get_X(self):
        """
        Getter for X array.

        Returns
        -------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        """
        return self.X
    
    def get_y(self):
        """
        Getter for y array.

        Returns
        -------
        y: numpy.ndarray (n_samples, 1)
            The label vector
        """
        return self.y
    
    def get_features(self):
        """
        Getter for features array.

        Returns
        -------
        features: list of str (n_features)
            The feature names
        """
        return self.features
    
    def get_label(self):
        """
        Getter for label name.

        Returns
        -------
        label: str (1)
            The label name
        """
        return self.label

    def get_discrete_mask(self) -> np.ndarray:
        """
        Returns the boolean mask indicating which columns in X correspond to discrete features.

        Returns
        -------
        numpy.ndarray (n_features,)
            Boolean mask indicating which columns in X correspond to discrete features
        """
        return self.discrete_mask


    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset.

        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape



    def get_mean(self) -> np.ndarray:
        """
        Computes the mean for each numeric feature in the dataset, and returns an array with the results. 
        For discrete features, the corresponding value in the array is set to np.nan.

        Returns
        -------
        numpy.ndarray (n_features,)
            An array with the mean for each numeric feature. If a feature is discrete, the corresponding 
            value in the array is np.nan.
        """
        discrete_mask = self.get_discrete_mask()

        # Calculate the mean of each numeric feature
        numeric_means = np.nanmean(self.X[:, ~discrete_mask], axis=0)

        # Create a result array with NaN values for discrete features
        result = np.empty(self.X.shape[1])
        result.fill(np.nan)

        # Assign the numeric means to the result array
        result[~discrete_mask] = numeric_means

        return result
    
    

    def get_median(self) -> np.ndarray:
        """
        Computes the median for each numeric feature in the dataset, and returns an array with the results. 
        For discrete features, the corresponding value in the array is set to np.nan.

        Returns
        -------
        numpy.ndarray (n_features,)
            An array with the median for each numeric feature. If a feature is discrete, the corresponding 
            value in the array is np.nan.
        """
        discrete_mask = self.get_discrete_mask()

        # Calculate the var of each numeric feature
        numeric_median = np.nanmedian(self.X[:, ~discrete_mask], axis=0)

        # Create a result array with NaN values for discrete features
        result = np.empty(self.X.shape[1])
        result.fill(np.nan)

        # Assign the numeric median to the result array
        result[~discrete_mask] = numeric_median

        return result
    
 
   
    def replace_nulls(self, method='mean'):
        """
        Replace all NaN values of each numeric feature using the specified method.

        Parameters
        ----------
        method : str, optional (default='mean')
            Method of replacing
        """
        discrete_mask = self.get_discrete_mask()

        if method == 'mean':
            means = np.nanmean(self.X[:, ~discrete_mask], axis=0)
            self.X[:, ~discrete_mask] = np.where(np.isnan(self.X[:, ~discrete_mask].astype(np.float32)), means, self.X[:, ~discrete_mask])
        elif method == 'median':
            medians = np.nanmedian(self.X[:, ~discrete_mask], axis=0)
            self.X[:, ~discrete_mask] = np.where(np.isnan(self.X[:, ~discrete_mask].astype(np.float32)), medians, self.X[:, ~discrete_mask])
        else:
            raise ValueError("Invalid method: {}".format(method))

    def count_nulls(self) -> np.ndarray:
        """
        Counts the number of null values in each numeric feature of X.

        Returns
        -------
        numpy.ndarray (n_features,)
            Array containing the number of null values in each feature.
        """
        discrete_mask = self.get_discrete_mask()
        bool_array = np.isnan(self.X[:, ~discrete_mask].astype(np.float32))
        nulls = np.count_nonzero(bool_array, axis = 0)
        return nulls
    
    def get_numeric_features(self):
        numeric_features = []
        for i, feat in enumerate(self.features):
            if np.issubdtype(self.X[:, i].dtype, np.number):
                numeric_features.append(feat)
        return numeric_features
    
if __name__ == "__main__":
    # Importar a função read_csv do seu código
    from to_csv import read_csv

    # Carregar o conjunto de dados iris_missing_data
    dataset = read_csv("iris_missing_data.csv", features=True, label=True)

    # Exibir as informações do conjunto de dados
    print("Shape:", dataset.shape())
    print("Features:", dataset.get_features())
    print("Label:", dataset.get_label())

    # Exibir o número de valores nulos em cada feature numérica
    null_counts = dataset.count_nulls()
    numeric_features = dataset.get_numeric_features()
    for feat, null_count in zip(numeric_features, null_counts):
        print("Feature:", feat, "Null Count:", null_count)

    # Calcular a média das features numéricas
    mean_values = dataset.get_mean()
    print("Mean Values:", mean_values)

    # Calcular a mediana das features numéricas
    median_values = dataset.get_median()
    print("Median Values:", median_values)

    # Substituir os valores nulos pelas médias
    dataset.replace_nulls(method='mean')

    # Exibir o número de valores nulos em cada feature numérica após a substituição
    null_counts_after = dataset.count_nulls()
    for feat, null_count_after in zip(numeric_features, null_counts_after):
        print("Feature:", feat, "Null Count (After):", null_count_after)
