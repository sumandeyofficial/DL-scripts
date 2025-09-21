import numpy as np
import matplotlib.pyplot as plt

class CosmicRayFNN:
    def __init__(self, hidden_neurons=10, learning_rate=0.03, epochs=20000, batch_size=None, alpha=1.0):
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = 1 #batch_size  # if None -> full batch

        # Weight initialization (Xavier)
        np.random.seed(42)
        self.W1 = np.random.uniform(-np.sqrt(6/(1+hidden_neurons)),
                                    np.sqrt(6/(1+hidden_neurons)),
                                    (1, hidden_neurons))
        self.b1 = np.zeros((1, hidden_neurons))

        self.W2 = np.random.uniform(-np.sqrt(6/(hidden_neurons+1)),
                                    np.sqrt(6/(hidden_neurons+1)),
                                    (hidden_neurons, 1))
        self.b2 = np.zeros((1, 1))

    # --------------------------
    # Activation (tanh)
    # --------------------------
    def tanh(self, x):
        return np.tanh(x) #1 / (1 + np.exp(-x))

    def tanh_derivative(self, s):
        return 1 -  s**2 #s * (1 - s)

    # --------------------------
    # Normalize utilities
    # --------------------------
    def normalize(self, data, data_min, data_max):
        return 2 * (data - data_min) / (data_max - data_min) - 1

    def denormalize(self, data_norm, data_min, data_max):
        return 0.5 * (data_norm + 1) * (data_max - data_min) + data_min

    # --------------------------
    # Training loop (Batch/SGD)
    # --------------------------
    def train(self, x_data, y_data):
        x = np.array(x_data).reshape(-1, 1)
        y = np.array(y_data).reshape(-1, 1)

        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()

        # Normalize
        x_norm = self.normalize(x, self.x_min, self.x_max)
        y_norm = self.normalize(y, self.y_min, self.y_max)

        # log(x+1) normalization
        #x_norm = np.log(np.array(x_data).reshape(-1, 1) + 1.0)
        #y_norm = np.log(np.array(y_data).reshape(-1, 1) + 1.0)

        N = x.shape[0]
        batch_size = self.batch_size if self.batch_size else N

        for epoch in range(self.epochs):
            total_loss = 0
            indices = np.random.permutation(N)

            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                xb = x_norm[batch_idx]
                yb = y_norm[batch_idx]

                # Forward
                z1 = xb @ self.W1 + self.b1
                a1 = self.tanh(z1)
                z2 = a1 @ self.W2 + self.b2
                a2 = self.tanh(z2)
                y_pred = a2

                # Loss (MSE)
                loss = np.mean((y_pred - yb) ** 2)
                total_loss += loss * xb.shape[0]

                # Backward
                dL_da2 = 2 * (y_pred - yb) / xb.shape[0]
                da2_dz2 = self.tanh_derivative(a2)
                dL_dz2 = dL_da2 * da2_dz2

                dL_dW2 = a1.T @ dL_dz2
                dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

                da1 = dL_dz2 @ self.W2.T
                dz1 = da1 * self.tanh_derivative(a1)
                dL_dW1 = xb.T @ dz1
                dL_db1 = np.sum(dz1, axis=0, keepdims=True)

                # Update
                self.W2 -= self.learning_rate * dL_dW2
                self.b2 -= self.learning_rate * dL_db2
                self.W1 -= self.learning_rate * dL_dW1
                self.b1 -= self.learning_rate * dL_db1

            if epoch % 1000 == 0:
                avg_loss = total_loss / N
                print(f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}")

    # --------------------------
    # Prediction
    # --------------------------
    def predict(self, x_input):
        x_arr = np.array(x_input).reshape(-1, 1)
        x_norm = self.normalize(x_arr, self.x_min, self.x_max) #np.log(np.array(x_input).reshape(-1, 1) + 1.0)

        z1 = x_norm @ self.W1 + self.b1
        a1 = self.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.tanh(z2)

        y_pred_norm = a2
        return self.denormalize(y_pred_norm, self.y_min, self.y_max).flatten() #np.exp(y_pred_norm).flatten() - 1.0

    # --------------------------
    # Plotting
    # --------------------------
    def plot(self, x_data, y_true, x_custom=None, y_custom_pred=None):
        y_fit = self.predict(x_data)
        plt.figure(figsize=(6, 4), dpi=200)
        plt.scatter(x_data, y_true, color='blue', alpha=0.5, label="Fermi-LAT and H.E.S.S data", s=15)
        plt.plot(x_data, y_fit, color='red', label="NN Fit") #color='darkred', linewidth=2.5,
        #if x_custom is not None and y_custom_pred is not None:
            #plt.scatter(x_custom, y_custom_pred, color='black', s=60, marker='x', label="Custom Predictions")
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Flux")
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(8e-4, 7.0)
        plt.title(f"Cosmic Ray Spectrum FNN — {self.hidden_neurons} Hidden Neurons")
        plt.legend(loc='best')
        plt.grid(False)
        plt.show()


# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    x_eV = [1.7320E+09, 5.4757E+09, 1.7320E+10, 5.4757E+10,
            3.1622E+11, 5.9844E+11, 8.7842E+11, 1.2891E+12,
            1.8921E+12, 2.7775E+12, 4.0766E+12, 6.4848E+12, 1.1534E+13]
    x = [val / 1e12 for val in x_eV]  # convert to TeV

    y_true = [0.39041, 0.61625, 0.74940, 0.87277,
              0.90842, 0.78417, 0.60144, 0.51635,
              0.49360, 0.44860, 0.35228, 0.16017, 0.00099]


    model = CosmicRayFNN(hidden_neurons=5, learning_rate=0.03, epochs=80000)
    model.train(x, y_true)

    # Custom predictions
    x_custom = x
    y_custom_pred = model.predict(x_custom)

    print("\nCustom Predictions:")
    for xi, yi in zip(x_custom, y_custom_pred):
        print(f"x = {xi:.2f} TeV => y_pred = {yi:.4f}")

    model.plot(x, y_true, x_custom, y_custom_pred)
