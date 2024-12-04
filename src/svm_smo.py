import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

class SVMwithSMO:
    def __init__(self, X, y, C, tolerance, max_passes):
        self.X = X
        self.y = y
        self.C = C
        self.tolerance = tolerance
        self.max_passes = max_passes
        self.m = X.shape[0]
        self.alphas = np.zeros(self.m)
        self.b = 0
        self.w = np.zeros(self.X.shape[1])
        self.slack = np.zeros(self.m)

    def compute_L_H(self, alpha_i, alpha_j, y_i, y_j):
        if y_i != y_j:
            return max(0, alpha_i - alpha_j), min(self.C, self.C + alpha_i - alpha_j)
        else:
            return max(0, alpha_i + alpha_j - self.C), min(self.C, alpha_i + alpha_j)

    def calculate_eta(self, X_i, X_j):
        return 2.0 * np.dot(X_i, X_j) - np.dot(X_i, X_i) - np.dot(X_j, X_j)

    def compute_error(self, X_i, y_i, alphas, b):
        f_x_i = np.dot(self.X, X_i) @ (alphas * self.y) + b
        return f_x_i - y_i

    def update(self, i, j, L, H, eta, E_i, E_j):
        if L == H or eta >= 0:
            return False

        alpha_j_old = self.alphas[j]
        alpha_i_old = self.alphas[i]

        self.alphas[j] -= self.y[j] * (E_i - E_j) / eta
        self.alphas[j] = max(min(self.alphas[j], H), L)

        if abs(self.alphas[j] - alpha_j_old) < 1e-5:
            return False

        self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

        b1 = self.b - E_i - self.y[i] * (self.alphas[i] - alpha_i_old) * np.dot(self.X[i], self.X[i]) - self.y[j] * (self.alphas[j] - alpha_j_old) * np.dot(self.X[i], self.X[j])
        b2 = self.b - E_j - self.y[i] * (self.alphas[i] - alpha_i_old) * np.dot(self.X[j], self.X[i]) - self.y[j] * (self.alphas[j] - alpha_j_old) * np.dot(self.X[j], self.X[j])
        self.b = (b1 + b2) / 2 if self.alphas[i] > 0 and self.alphas[i] < self.C else (b1 if self.alphas[j] > 0 and self.alphas[j] < self.C else (b1 + b2) / 2)

        self.update_w()
        self.calculate_slack()

        return True

    def update_w(self):
        self.w = np.sum((self.alphas * self.y)[:, np.newaxis] * self.X, axis=0)

    def calculate_slack(self):
        for i in range(self.m):
            self.slack[i] = max(0, 1 - self.y[i] * (np.dot(self.w, self.X[i]) + self.b))

    def train(self):
        for _ in tqdm(range(self.max_passes), desc="Training progress"):
            num_changed_alphas = 0
            for i in range(self.m):  # Note: You had a missing colon here in the provided snippet
                E_i = self.compute_error(self.X[i], self.y[i], self.alphas, self.b)
                if (self.y[i] * E_i < -self.tolerance and self.alphas[i] < self.C) or \
                (self.y[i] * E_i > self.tolerance and self.alphas[i] > 0):
                    j = random.randint(0, self.m - 1)
                    while i == j:
                        j = random.randint(0, self.m - 1)
                    E_j = self.compute_error(self.X[j], self.y[j], self.alphas, self.b)
                    L, H = self.compute_L_H(self.alphas[i], self.alphas[j], self.y[i], self.y[j])
                    eta = self.calculate_eta(self.X[i], self.X[j])
                    if self.update(i, j, L, H, eta, E_i, E_j):
                        num_changed_alphas += 1

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = np.sum(predictions == y_test)
        return correct / len(y_test) * 100

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data using multiple metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Confusion Matrix
        """
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print evaluation results
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1 Score: {:.2f}".format(f1))
        print("Confusion Matrix:")
        print(conf_matrix)

        return accuracy, precision, recall, f1, conf_matrix


def run_model_svm_smo(X_train, X_test, y_train, y_test):
    """
    Run the SVM model, train it and evaluate it on test data
    """
    model = SVMwithSMO(X_train, y_train, C=1.0, tolerance=0.0001, max_passes=10)
    model.train()

    # Evaluate the model on test data
    accuracy, precision, recall, f1, conf_matrix = model.evaluate(X_test, y_test)

    return accuracy, precision, recall, f1, conf_matrix