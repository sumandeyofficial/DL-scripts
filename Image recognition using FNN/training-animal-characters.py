import numpy as np
import urllib.request
import os
import tarfile
import pickle


class CIFAR10_FNN:
    def __init__(self, input_size=3072, hidden1=250, hidden2=100, output_size=10,
                 lr=0.001, epochs=40, batch_size=128, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(1. / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(1. / hidden2)
        self.b3 = np.zeros((1, output_size))

        # Initialize Adam moment terms
        self.m = {name: np.zeros_like(param) for name, param in self.__dict__.items() if name.startswith(('W', 'b'))}
        self.v = {name: np.zeros_like(param) for name, param in self.__dict__.items() if name.startswith(('W', 'b'))}
        self.t = 0 

    # -----------------------------
    # Activation Functions
    # -----------------------------
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy(pred, label):
        return -np.sum(label * np.log(pred + 1e-8)) / pred.shape[0]

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        z3 = a2 @ self.W3 + self.b3
        output = self.softmax(z3)
        cache = (X, z1, a1, z2, a2, z3, output)
        return output, cache

    # -----------------------------
    # Backpropagation
    # -----------------------------
    def backward(self, cache, Y_batch):
        X, z1, a1, z2, a2, z3, output = cache
        batch_size = len(X)

        d3 = (output - Y_batch) / batch_size
        dW3 = a2.T @ d3
        db3 = np.sum(d3, axis=0, keepdims=True)

        d2 = d3 @ self.W3.T * self.relu_deriv(z2)
        dW2 = a1.T @ d2
        db2 = np.sum(d2, axis=0, keepdims=True)

        d1 = d2 @ self.W2.T * self.relu_deriv(z1)
        dW1 = X.T @ d1
        db1 = np.sum(d1, axis=0, keepdims=True)

        # Adam optimizer
        self.t += 1
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

        for name, grad in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            setattr(self, name, getattr(self, name) - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon))

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, X_train, Y_train):
        n_samples = X_train.shape[0]
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_train, Y_train = X_train[indices], Y_train[indices]
            epoch_loss = 0

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                Y_batch = Y_train[i:i+self.batch_size]

                output, cache = self.forward(X_batch)
                loss = self.cross_entropy(output, Y_batch)
                epoch_loss += loss * len(X_batch)
                self.backward(cache, Y_batch)

            epoch_loss /= n_samples
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    def evaluate(self, X_test, y_test):
        output, _ = self.forward(X_test)
        pred_labels = np.argmax(output, axis=1)
        acc = np.mean(pred_labels == y_test)
        print(f"Test Accuracy: {acc:.4f}")
        return acc

    # -----------------------------
    # Save Model
    # -----------------------------
    def save(self, filename='cifar10_model_adam.npz', accuracy=None):
        np.savez(filename,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 accuracy=accuracy)
        print(f"Model saved as {filename}")


# ======================================================
# Load & Prepare CIFAR-10 Dataset
# ======================================================
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
save_path = "/Users/sumandey/Downloads/Deep-Learning/cifar-10-python.tar.gz"
extract_path = "/Users/sumandey/Downloads/Deep-Learning/cifar-10-batches-py"

if not os.path.exists(extract_path):
    if not os.path.exists(save_path):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, save_path)
    print("Extracting CIFAR-10...")
    with tarfile.open(save_path, 'r:gz') as tar:
        tar.extractall(path=os.path.dirname(save_path))

# Loading batches
train_data = []
train_labels = []
for i in range(1, 6):
    batch = unpickle(f"{extract_path}/data_batch_{i}")
    train_data.append(batch[b'data'])
    train_labels += batch[b'labels']

X_train = np.concatenate(train_data)
y_train = np.array(train_labels)

test_batch = unpickle(f"{extract_path}/test_batch")
X_test = test_batch[b'data']
y_test = np.array(test_batch[b'labels'])

# Normalize and flattening
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
Y_train = np.eye(10)[y_train]
Y_test = np.eye(10)[y_test]

# ======================================================
# Train model
# ======================================================
model = CIFAR10_FNN(lr=0.001, epochs=40, batch_size=128)
print("\nTraining Fully Connected Neural Network on CIFAR-10 with Adam Optimizer...\n")
model.train(X_train, Y_train)
accuracy = model.evaluate(X_test, y_test)
model.save(filename='/Users/sumandey/Downloads/Deep-Learning/cifar10_model_adam.npz', accuracy=accuracy)
