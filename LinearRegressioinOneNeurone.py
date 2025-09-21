import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import random

plt.style.use('science')
plt.rc('text', usetex=plt.rcParamsDefault['text.usetex'])

class LinearRegressionGD:  
    def __init__(self, learning_rate=0.01, epochs=20):  # constructing a constructor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = random.uniform(-10, 10)
        self.b = random.uniform(-10, 10)
        self.loss_history = []  # Loss history

    def train(self, X, Y):
        n = len(X)
        for epoch in range(self.epochs):
            total_loss = 0
            dw, db = 0, 0

            for i in range(n):
                x = X[i]
                y = Y[i]

                # Forward pass -- y_pred
                y_pred = self.w * x + self.b

                # calculating dw and db
                error = y_pred - y
                total_loss += error ** 2
                dw += 2 * error * x
                db += 2 * error

            # Calculating Mean Squared Error (MSE)
            total_loss /= n
            self.loss_history.append(total_loss)  # Loss history

            # Updating w and b
            self.w -= self.learning_rate * dw / n
            self.b -= self.learning_rate * db / n

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}, w = {self.w:.4f}, b = {self.b:.4f}")

        print("\nTraining complete.")
        print(f"Final Model: y = {self.w:.4f} * x + {self.b:.4f}")
        print(f"Final Loss (MSE) = {total_loss:.4f}")

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        y_mean = sum(Y) / len(Y)

        ss_total = sum((y - y_mean) ** 2 for y in Y)
        ss_res = sum((Y[i] - Y_pred[i]) ** 2 for i in range(len(Y)))

        r2 = 1 - (ss_res / ss_total)
        accuracy_percent = r2 * 100

        print(f"R^2 Score: {r2:.4f}")
        print(f"Accuracy: {accuracy_percent:.2f}%")

        return r2, accuracy_percent, Y_pred

    def plot(self, X, Y, Y_pred, accuracy_percent):
        plt.figure(figsize=(6, 4), dpi=200)
        plt.scatter(X, Y, color='black', label='Actual Data')
        plt.plot(X, Y_pred, color='red', label='Predicted Line')
        plt.title(f"Linear Regression Using Single Neuron\nAccuracy: {accuracy_percent:.2f}%")
        plt.xlabel(r"$X$")
        plt.ylabel(r"$Y$ and $Y_{\text{pred}}$")
        plt.legend()
        plt.grid(False)
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(6, 4), dpi=200)
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o', color='blue')
        plt.title("Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error (Loss)")
        plt.grid(False)
        plt.show()

    def plot_rate(X, Y, learning_rates, final_losses):
        plt.figure(figsize=(6, 4), dpi=200)
        plt.plot(learning_rates, final_losses, marker='o', color='red')
        plt.title("Final Loss vs Learning Rate")
        plt.xlabel("Learning Rate")
        plt.xscale("log")
        plt.ylabel("Final Mean Squared Error (Loss)")
        plt.grid(False)
        plt.show()

# ----------------------
# Example use case
# ----------------------

if __name__ == "__main__":
    X = [-3, -1, 0.5, 1, 2, 3, 4, 5, 7, 10]
    Y = [-14, -4, 2.5, 6, 10, 16, 20, 25, 35, 51]

    model = LinearRegressionGD(learning_rate=0.01, epochs=20)
    model.train(X, Y)
    r2, accuracy_percent, Y_pred = model.evaluate(X, Y)
    
    #-------------------------------------
    # Plotting the regression fit
    #-------------------------------------
    model.plot(X, Y, Y_pred, accuracy_percent)

    #-------------------------------------
    # Plotting the loss v epoch
    #-------------------------------------
    model.plot_loss()

    #-------------------------------------
    # Plotting the loss v learning rates
    #-------------------------------------
    learning_rates = np.arange(0.0004, 0.012, 0.002)
    final_losses = []

    for lr in learning_rates:
        model = LinearRegressionGD(learning_rate=lr, epochs=20)
        model.train(X, Y)
        final_losses.append(model.loss_history[-1])
    #print(final_losses)

    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(learning_rates, final_losses, marker='o', color='red')
    plt.title("Final Loss vs Learning Rate")
    plt.xlabel("Learning Rate")
    #plt.xscale("log")
    plt.ylabel("Final Mean Squared Error (Loss)")
    plt.grid(False)
    plt.show()