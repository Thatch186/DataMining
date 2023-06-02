import numpy as np

class LinearRegression:
    def __init__(self, epochs=1000, lr=0.001, ldb=1, gd=False):
        """
        Linear regression model.

        Parameters:
        - epochs: int, number of epochs for gradient descent (GD).
        - lr: float, learning rate for GD.
        - ldb: float, lambda for regularization. If non-positive, L2 regularization is not applied.
        - gd: bool, if True uses gradient descent (GD) to train the model, otherwise uses closed form linear algebra.
        """
        self.theta = None
        self.epochs = epochs
        self.lr = lr
        self.ldb = ldb
        self.gd = gd
        self.is_fitted = False

    def fit(self, X, y):
        X = self.add_intersect(X)
        self.X = X
        self.y = y
        # Closed form or GD
        self.train_gd(X, y) if self.gd else self.train_closed(X, y)
        self.is_fitted = True

    def train_closed(self, X, y):
        if self.ldb > 0:
            n = X.shape[1]
            identity = np.eye(n)
            identity[0, 0] = 0
            self.theta = np.linalg.inv(X.T @ X + self.ldb * identity) @ X.T @ y
        else:
            self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def train_gd(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        if self.ldb > 0:
            lbds = np.full(m, self.ldb)
            lbds[0] = 0
        for epoch in range(self.epochs):
            if self.ldb > 0:
                grad = (X.dot(self.theta) - y).dot(X)
                self.theta -= (self.lr / m) * (lbds * self.theta + grad)
            else:
                grad = 1 / m * (X @ self.theta - y) @ X
                self.theta -= self.lr * grad
            self.history[epoch] = [self.theta.copy(), self.cost()]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        _x = np.hstack(([1], x))
        return np.dot(self.theta, _x)

    def cost(self, X=None, y=None, theta=None):
        X = self.add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta
        y_pred = np.dot(X, theta)
        return np.mean((y_pred - y) ** 2)

    @staticmethod
    def add_intersect(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

if __name__ == "__main__":
    # Create a sample dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 5, 7])

    # Create an instance of the LinearRegression class
    lr = LinearRegression(epochs=1000, lr=0.001, ldb=1, gd=True)

    # Fit the model to the dataset
    lr.fit(X, y)

    # Test the predict method
    x_test = np.array([7, 8])
    prediction = lr.predict(x_test)
    print("Prediction:", prediction)

    # Test the cost method
    cost = lr.cost(X, y)
    print("Cost:", cost)