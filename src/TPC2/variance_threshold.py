import numpy as np
from dataset import Dataset

class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.
    Parameters
    ----------
    threshold: float
        The threshold value to use for feature selection. Features with a
        training-set variance lower than this threshold will be removed.
    Attributes
    ----------
    variance: array-like, shape (n_features,)
        The variance of each feature.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.
        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.
        Returns
        -------
        self : object
        """
        #VER SE É TUDO NUMERICO
        self.variance = np.var(dataset.X, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold.
        Parameters
        ----------
        dataset: Dataset
        Returns
        -------
        dataset: Dataset
        """
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        numeric_features = list(dataset.get_numeric_features())
        discrete_features = []
        print("NF: ",numeric_features)
        return Dataset(X=X, y=dataset.y, features=list(features),discrete_features=[],numeric_features=numeric_features, label=dataset.label)
    
if __name__ == '__main__':
    # Criar um objeto Dataset de exemplo
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    features = ["feature1", "feature2", "feature3"]
    numeric_features = ["feature1", "feature2"]  # Exemplo de numeric_features
    dataset = Dataset(X=X, y=y, features=features, numeric_features=numeric_features)

    # Criar um objeto VarianceThreshold e ajustá-lo ao conjunto de dados
    threshold = 1.0
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(dataset)

    # Transformar o conjunto de dados de acordo com o limiar de variância
    transformed_dataset = vt.transform(dataset)

    # Imprimir o resultado
    print("Features selecionadas:", transformed_dataset.features)
    print("Dados transformados:\n", transformed_dataset.X)