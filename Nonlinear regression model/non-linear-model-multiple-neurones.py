import numpy as np
import matplotlib.pyplot as plt


class MLPAdam:
    def __init__(self, n_input=1, nh1=50, nh2=30, n_output=1, lr=0.01, epochs=6000,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=None, seed=42):
        np.random.seed(seed)

        self.n_input = n_input
        self.nh1 = nh1
        self.nh2 = nh2
        self.n_output = n_output
        self.lr = lr
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size

        # Xavier initialization
        self.W1 = self.xavier_init(n_input, nh1)
        self.b1 = np.zeros((1, nh1))
        self.W2 = self.xavier_init(nh1, nh2)
        self.b2 = np.zeros((1, nh2))
        self.W3 = self.xavier_init(nh2, n_output)
        self.b3 = np.zeros((1, n_output))

        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.mW3 = np.zeros_like(self.W3); self.vW3 = np.zeros_like(self.W3)
        self.mb3 = np.zeros_like(self.b3); self.vb3 = np.zeros_like(self.b3)

        self.losses = []

    # --------------------
    # Helper methods
    # --------------------
    @staticmethod
    def xavier_init(n_in, n_out):
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(-limit, limit, (n_in, n_out))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        return 1 - np.tanh(x) ** 2

    # --------------------
    # Training
    # --------------------
    def train(self, X, Y):
        if self.batch_size is None:
            self.batch_size = X.shape[0]  # full batch if not given

        for epoch in range(1, self.epochs + 1):
            # Shuffle
            perm = np.random.permutation(X.shape[0])
            X_shuffled, Y_shuffled = X[perm], Y[perm]

            epoch_loss = 0

            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                Y_batch = Y_shuffled[i:i + self.batch_size]

                # Forward pass
                z1 = X_batch @ self.W1 + self.b1
                a1 = self.tanh(z1)

                z2 = a1 @ self.W2 + self.b2
                a2 = self.tanh(z2)

                z3 = a2 @ self.W3 + self.b3
                y_pred = self.tanh(z3)

                # Loss calculation
                loss = np.mean((y_pred - Y_batch) ** 2)
                epoch_loss += loss

                # Back propagation
                delta3 = 2 * (y_pred - Y_batch) * self.dtanh(z3) / self.batch_size
                dW3 = a2.T @ delta3
                db3 = np.sum(delta3, axis=0, keepdims=True)

                delta2 = (delta3 @ self.W3.T) * self.dtanh(z2)
                dW2 = a1.T @ delta2
                db2 = np.sum(delta2, axis=0, keepdims=True)

                delta1 = (delta2 @ self.W2.T) * self.dtanh(z1)
                dW1 = X_batch.T @ delta1
                db1 = np.sum(delta1, axis=0, keepdims=True)

                # Adam optimization
                t = epoch

                self.W1, self.mW1, self.vW1 = self.adam_update(self.W1, dW1, self.mW1, self.vW1, t)
                self.b1, self.mb1, self.vb1 = self.adam_update(self.b1, db1, self.mb1, self.vb1, t)

                self.W2, self.mW2, self.vW2 = self.adam_update(self.W2, dW2, self.mW2, self.vW2, t)
                self.b2, self.mb2, self.vb2 = self.adam_update(self.b2, db2, self.mb2, self.vb2, t)

                self.W3, self.mW3, self.vW3 = self.adam_update(self.W3, dW3, self.mW3, self.vW3, t)
                self.b3, self.mb3, self.vb3 = self.adam_update(self.b3, db3, self.mb3, self.vb3, t)

            self.losses.append(epoch_loss)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

    def adam_update(self, param, grad, m, v, t):
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param, m, v

    # --------------------
    # Prediction
    # --------------------
    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.tanh(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.tanh(z2)

        z3 = a2 @ self.W3 + self.b3
        y_pred = self.tanh(z3)
        return y_pred

    # --------------------
    # Plotting
    # --------------------
    def plot(self, X, Y, y_pred):
        plt.figure(figsize=(10, 5))
        plt.scatter(X, Y, label="Original", color='green', s=10)
        plt.plot(X, y_pred, label="Predicted", color='blue', linewidth=2)
        plt.legend()
        plt.title(f"Multiple Layer fit with tanh activation (No. neurones in h1={self.nh1}, No. neurones in h2={self.nh2})")
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    # example data
    X = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    Y = np.cos(3*X) * (1 + 0.2 * X**2) + 0.1 * np.random.randn(*X.shape)

    # Normalization
    def normalize(X): return (X - X.min()) / (X.max() - X.min()) * 2 - 1
    X_norm = normalize(X)
    Y_norm = normalize(Y)

    # Training model
    model = MLPAdam(nh1=50, nh2=30, epochs=3000, lr=0.01)
    model.train(X_norm, Y_norm)

    # Prediction
    y_pred_norm = model.predict(X_norm)

    # De-normalization
    def denormalize(X, original_min, original_max):
        return 0.5 * (X + 1) * (original_max - original_min) + original_min
    y_pred = denormalize(y_pred_norm, Y.min(), Y.max())

    model.plot(X, Y, y_pred)
