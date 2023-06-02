import numpy as np
from dataset import Dataset

class MLP:

    def __init__(self, dataset, hidden_nodes=2, normalize=False):
        self.X = dataset.get_X()
        self.y = dataset.get_y()
        self.X = np.hstack((np.ones([self.X.shape[0], 1]), self.X))  # colocar 1's no X

        self.h = hidden_nodes
        self.W1 = np.zeros([hidden_nodes, self.X.shape[1]])
        self.W2 = np.zeros([1, hidden_nodes + 1])

        if normalize:
            self.normalize()
        else:
            self.normalized = False

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def setWeights(self, w1, w2):
        self.W1 = w1
        self.W2 = w2


    def predict(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])

        if self.normalized:
            if np.all(self.sigma!= 0): 
                x[1:] = (x[1:] - self.mu) / self.sigma
            else: x[1:] = (x[1:] - self.mu)

        z2 = np.dot(self.W1,x)
        a2 = np.empty([z2.shape[0]+1])
        a2[0] = 1
        a2[1:]= sigmoid(z2)
        z3 = sigmoid(np.dot(self.W2,a2))
        h = sigmoid(z3)

        return h

    def costFunction(self, weights):
        self.setWeights(weights[:self.h * (self.X.shape[1])].reshape(self.h, self.X.shape[1]),
                        weights[self.h * (self.X.shape[1]):].reshape(1, self.h + 1))

        # Forward propagation
        z1 = np.dot(self.X, self.W1.T)
        a1 = self.sigmoid(z1)
        a1 = np.hstack((np.ones([a1.shape[0], 1]), a1))
        z2 = np.dot(a1, self.W2.T)
        a2 = self.sigmoid(z2)

        # Compute the cost
        m = self.X.shape[0]
        cost = (-1 / m) * np.sum(self.y * np.log(a2) + (1 - self.y) * np.log(1 - a2))

        return cost


    def build_model(self):
        from scipy import optimize

        size = self.h * self.X.shape[1] + self.h+1

        initial_w = np.random.rand(size)        
        result = optimize.minimize(lambda w: self.costFunction(w), initial_w, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        weights = result.x
        self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
        self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h+1])

    def normalize(self):
        self.mu = np.mean(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] - self.mu
        self.sigma = np.std(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] / self.sigma
        self.normalized = True

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    # Create a sample dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])

    # Define the feature types
    discrete_features = ["feature_name_1"]
    numeric_features = ["feature_name_2", "feature_name_3"]

    # Create an instance of the Dataset class
    dataset = Dataset(X, y, discrete_features=discrete_features, numeric_features=numeric_features)

    # Create an instance of the MLP class
    mlp = MLP(dataset, hidden_nodes=2, normalize=False)

    # Build the model and set the weights
    mlp.build_model()
    w1 = np.random.rand(mlp.h, mlp.X.shape[1])
    w2 = np.random.rand(1, mlp.h+1)
    mlp.setWeights(w1, w2)

    # Test the predict method
    instance = [1, 2, 3]
    prediction = mlp.predict(instance)
    print("Prediction:", prediction)

    # Test the costFunction method
    weights = np.concatenate([mlp.W1.flatten(), mlp.W2.flatten()])
    cost = mlp.costFunction(weights)
    print("Cost:", cost)
#Make sure to replace "feature_name_1", "feature_name_2", and "feature_name_3" with the actual names of your features. This way, you're explicitly providing the discrete and numeric features, which should resolve the error.












