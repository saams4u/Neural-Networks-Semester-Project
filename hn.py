
import numpy as np
import random
import matplotlib.pyplot as plt

# Import exemplars representing numerals 0 to 7 in a 100-element format (10x10 rasterized array).
from data import exemplars


# Define Hopfield Network class
class HopfieldNetwork:

    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iterations=100):
        sequence = []
        current_pattern = np.copy(pattern)
        for _ in range(max_iterations):
            idx = random.randint(0, self.size - 1)
            dot_product = np.dot(self.weights[idx], current_pattern)
            current_pattern[idx] = 1 if dot_product >= 0 else -1
            sequence.append(np.copy(current_pattern))
        return sequence

# Accepts a pattern and a list of noise levels, and returns a list of noisy patterns with the specified noise levels
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

# Train the Hopfield network
network = HopfieldNetwork(100)
network.train(exemplars)

# Test the network with noisy inputs
num_test_sequences = 2
noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

for i, exemplar in enumerate(exemplars):
    for j in range(num_test_sequences):
        noise_level = noise_levels[i]
        noisy_exemplars = create_noisy_inputs(exemplar, [noise_level])
        recalled_sequence = network.recall(noisy_exemplars[0])

        print(f"Exemplar {i}, Sequence {j+1}\n")

        plt.title('Original')
        display_pattern(exemplar)

        plt.title('Noisy')
        display_pattern(noisy_exemplars[0])

        print("Recalled Sequence:")

        for k, pattern in enumerate(recalled_sequence):
            if k % 10 == 0:  # Display every 10th iteration
                print(f"Iteration {k}")
                plt.title(f'Recalled (Iteration {k})')
                display_pattern(pattern)

        print("-" * 20)