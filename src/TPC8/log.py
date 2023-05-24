import numpy as np

class LogisticRegression:
    def __init__(self, epochs=1000, lr=0.001, ldb=1, gd=False):
        """
        Logistic regression model.

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
                z = X.dot(self.theta)
                h = self.sigmoid(z)
                grad = (h - y).dot(X)
                self.theta -= (self.lr / m) * (lbds * self.theta + grad)
            else:
                z = X.dot(self.theta)
                h = self.sigmoid(z)
                grad = 1 / m * (X.T @ (h - y))
                self.theta -= self.lr * grad
            self.history[epoch] = [self.theta.copy(), self.cost()]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        _x = np.hstack(([1], x))
        return self.sigmoid(np.dot(self.theta, _x))

    def cost(self, X=None, y=None, theta=None):
        X = self.add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta
        z = X.dot(theta)
        h = self.sigmoid(z)
        return np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))

    @staticmethod
    def add_intersect(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
