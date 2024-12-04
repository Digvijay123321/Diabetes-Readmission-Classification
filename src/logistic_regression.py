from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

class logistic_Regression:
    def __init__(self, X, y, learningRate, tolerance, lamda, maxIteration):
        self.X = X
        self.y = y
        self.learningRate = learningRate
        self.tolerance = tolerance
        self.lamda = lamda
        self.maxIteration = maxIteration

    def add_X0(self, X):
        return np.column_stack([np.ones([X.shape[0], 1]), X])

    def sigmoid(self, z):
        sig = 1 / (1 + np.exp(-z))  # fixed np.exp instead of math.e
        return sig

    def costFunction(self, X, y):
        pred_ = np.log(np.ones(X.shape[0]) + np.exp(X.dot(self.w))) - X.dot(self.w) * y
        cost = pred_.sum()
        return cost

    def gradient(self, X, y):
        sig = self.sigmoid(X.dot(self.w))
        grad = (sig - y).dot(X)
        return grad

    def gradientDescent(self, X, y):
        errors = []
        last = float('inf')

        for i in tqdm(range(self.maxIteration)):
            self.w = self.w - self.learningRate * (self.gradient(X, y) + (self.lamda * self.w))
            curr = self.costFunction(X, y)

            diff = last - curr
            last = curr

            errors.append(curr)
            if abs(diff) < self.tolerance:
                print('Model stopped')
                break

    def predict(self, X):
        sig = self.sigmoid(X.dot(self.w))
        return np.around(sig)

    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.w))

    def evaluate(self, y, y_hat):
        y = (y == 1)
        y_hat = (y_hat == 1)

        accuracy = (y == y_hat).sum() / len(y)
        precision = (y & y_hat).sum() / y_hat.sum()
        recall = (y & y_hat).sum() / y.sum()
        f1score = 2 * (precision * recall) / (precision + recall)

        return accuracy, precision, recall, f1score

def run_model_log_reg(self, X_train, X_test, y_train, y_test):
        
        # Add bias term (X0 = 1) for both training and testing data
        X_train = self.add_X0(X_train)
        X_test = self.add_X0(X_test)

        # Initialize weights
        self.w = np.ones(X_train.shape[1], dtype=np.float64)

        # Train the model
        self.gradientDescent(X_train, y_train)

        # Predict on the test set
        y_pred = self.predict(X_test)

        # Evaluate the model on the test data
        accuracy, precision, recall, f1score = self.evaluate(y_test, y_pred)

        print(f"Model evaluation on hold-out test data:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1score}")
