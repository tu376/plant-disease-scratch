import numpy as np

class LogisticRegressionModel:
    def __init__(self, input_dim, num_classes, lr=0.01):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        self.lr = lr

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)  # numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        n = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / n

    def one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def fit(self, X, y, epochs=100):
        n_samples = X.shape[0]
        y_onehot = self.one_hot(y, self.b.shape[1])

        for epoch in range(epochs):
            # Forward
            logits = X @ self.W + self.b
            probs = self.softmax(logits)

            # Loss
            loss = self.cross_entropy_loss(y_onehot, probs)

            # Backward
            dZ = (probs - y_onehot) / n_samples
            dW = X.T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)

            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 10 == 0:
                acc = self.score(X, y)
                print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

    def predict_proba(self, X):
        logits = X @ self.W + self.b
        return self.softmax(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)