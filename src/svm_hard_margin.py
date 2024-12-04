import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class HardMarginSVM:
    def __init__(self, learning_rate: float, n_iter: int = 10000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples = X.shape[0]
        y = np.where(y <= 0, -1, 1)  # Ensure labels are either -1 or 1

        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                # Check if the point is within the margin or misclassified
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) < 1
                
                if condition:
                    # Update rule when a point is misclassified or within margin
                    self.w = self.w - self.learning_rate * (2 * np.dot(self.w, self.w) - np.dot(x_i, y[idx]))
                    self.b = self.b - self.learning_rate * (-y[idx])  # Update bias for misclassified points

    def predict(self, X):
        # Decision function to classify based on the sign of the result
        pred = np.dot(X, self.w) + self.b
        return np.sign(pred)

    def decision_function(self, X):
        # Compute the distance from the hyperplane
        return np.dot(X, self.w) + self.b

    def evaluate(self, y_test, y_pred):
        # Calculate accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print the metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:")
        print(conf_matrix)
        
        return accuracy, precision, recall, f1, conf_matrix

def run_model_svm_hard_margin(self, X_train, X_test, y_train, y_test):
        # Fit the model
        self.fit(X_train, y_train)

        # Predict on test data
        y_pred = self.predict(X_test)

        # Evaluate model performance and return metrics
        return self.evaluate(y_test, y_pred)
