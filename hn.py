
import numpy as np
import random
import matplotlib.pyplot as plt

# Import exemplars representing numerals 0 to 7 in a 100-element format (10x10 rasterized array).
from data import exemplars


# Define the Hopfield Network class
class HopfieldNetwork:

    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iterations=100):
        current_pattern = np.copy(pattern)
        for _ in range(max_iterations):
            idx = random.randint(0, self.size - 1)
            dot_product = np.dot(self.weights[idx], current_pattern)
            current_pattern[idx] = 1 if dot_product >= 0 else -1
        return current_pattern

def create_noisy_inputs(pattern, noise_levels):
    noisy_patterns = []
    for noise_level in noise_levels:
        noisy_pattern = np.copy(pattern)
        indices = np.random.choice(len(pattern), int(noise_level * len(pattern)), replace=False)
        noisy_pattern[indices] = -noisy_pattern[indices]
        noisy_patterns.append(noisy_pattern)
    return noisy_patterns

def display_pattern(pattern):
    reshaped_pattern = pattern.reshape(10, 10)
    plt.imshow(reshaped_pattern, cmap='gray')
    plt.show()

def accuracy(original, recalled):
    return np.sum(original == recalled) / len(original)

# Train the Hopfield network
network = HopfieldNetwork(100)
network.train(exemplars)

# Test the network with noisy inputs
num_test_sequences = 2
noise_levels = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i, exemplar in enumerate(exemplars):
    for j in range(num_test_sequences):
        noise_level = noise_levels[i]
        noisy_exemplars = create_noisy_inputs(exemplar, [noise_level])
        recalled_pattern = network.recall(noisy_exemplars[0])

        acc = accuracy(exemplar, recalled_pattern)
        print(f"Exemplar {i}, Sequence {j+1}, Validation Accuracy: {acc*100:.2f}%")

        plt.title('Original')
        display_pattern(exemplar)

        plt.title('Noisy')
        display_pattern(noisy_exemplars[0])

        plt.title('Recalled')
        display_pattern(recalled_pattern)