import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class CIFAR10ModelPredictor:
    def __init__(self, model_path, data_dir):
        """Load model weights and CIFAR-10 test data."""
        self.model_path = model_path
        self.data_dir = data_dir

        # CIFAR-10 class names
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        # ---------------- Loading model ----------------
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found. Train and save the model first.")
        model = np.load(model_path)
        self.W1, self.b1 = model['W1'], model['b1']
        self.W2, self.b2 = model['W2'], model['b2']
        self.W3, self.b3 = model['W3'], model['b3']
        self.accuracy = model['accuracy']

        # ---------------- Loading CIFAR-10 test data ----------------
        test_batch = os.path.join(data_dir, 'test_batch')
        if not os.path.exists(test_batch):
            raise FileNotFoundError(f"{test_batch} not found. CIFAR-10 not extracted properly.")
        with open(test_batch, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        X_test = data_dict[b'data']
        y_test = np.array(data_dict[b'labels'])

        # Preprocess: normalize
        self.X_test = X_test / 255.0
        self.y_test = y_test

    # ---------------- Activation functions ----------------
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    # ---------------- Prediction ----------------
    def predict(self, x):
        a1 = self.relu(x @ self.W1 + self.b1)
        a2 = self.relu(a1 @ self.W2 + self.b2)
        out = self.softmax(a2 @ self.W3 + self.b3)
        return np.argmax(out, axis=1)

    # ---------------- Visualization ----------------
    def visualize_predictions(self, rows=5, cols=5):
        num_samples = rows * cols
        random_indices = np.random.choice(len(self.X_test), size=num_samples, replace=False)
        random_images = self.X_test[random_indices]
        random_labels = self.y_test[random_indices]

        predictions = self.predict(random_images)

        plt.figure(figsize=(8, 8))
        for i in range(num_samples):
            plt.subplot(rows, cols, i + 1)
            img = random_images[i].reshape(3, 32, 32).transpose(1, 2, 0)
            plt.imshow(img)
            plt.title(
                f"Pred: {self.classes[predictions[i]]}\nActual: {self.classes[random_labels[i]]}",
                fontsize=10
            )
            plt.axis('off')

        plt.suptitle(f"CIFAR-10 Prediction Visualization\nAccuracy: {self.accuracy * 100:.2f}%", fontsize=16)
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------
# Example usage
# -----------------------------------------------------
if __name__ == "__main__":
    predictor = CIFAR10ModelPredictor(
        model_path='/Users/sumandey/Downloads/Deep-Learning/cifar10_model_adam.npz',
        data_dir='/Users/sumandey/Downloads/Deep-Learning/cifar-10-batches-py'
    )
    predictor.visualize_predictions(rows=6, cols=6)
