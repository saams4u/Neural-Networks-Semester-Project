
import numpy as np
import random
import matplotlib.pyplot as plt

# Import exemplars representing numerals 0 to 7 in a 100-element format (10x10 rasterized array).
from data import exemplars


# Define Restricted Boltzmann Machine class
class RestrictedBoltzmannMachine:

    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Train the RBM using Hinton's contrastive divergence method
    def train(self, patterns, learning_rate=0.01, n_epochs=10, batch_size=1, k=1):
        for epoch in range(n_epochs):
            np.random.shuffle(patterns)
            for batch in range(0, len(patterns), batch_size):
                data = patterns[batch:batch + batch_size]
                pos_hidden_activations = np.dot(data, self.weights) + self.hidden_bias
                pos_hidden_probs = self.sigmoid(pos_hidden_activations)
                pos_hidden_states = pos_hidden_probs > np.random.rand(*pos_hidden_probs.shape)
                pos_associations = np.dot(np.transpose(data), pos_hidden_probs)

                neg_visible_activations = np.dot(pos_hidden_states, np.transpose(self.weights)) + self.visible_bias
                neg_visible_probs = self.sigmoid(neg_visible_activations)
                neg_visible_probs[:, -1] = 1  # Fix the bias unit
                neg_hidden_activations = np.dot(neg_visible_probs, self.weights) + self.hidden_bias
                neg_hidden_probs = self.sigmoid(neg_hidden_activations)
                neg_associations = np.dot(np.transpose(neg_visible_probs), neg_hidden_probs)

                self.weights += learning_rate * (pos_associations - neg_associations) / batch_size
                self.visible_bias += learning_rate * np.mean(data - neg_visible_probs, axis=0)
                self.hidden_bias += learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    # Recall the original pattern given a noisy input using Gibbs sampling
    def recall(self, pattern, n_gibbs=100):
        visible = pattern.copy()
        for _ in range(n_gibbs):
            hidden_activations = np.dot(visible, self.weights) + self.hidden_bias
            hidden_probs = self.sigmoid(hidden_activations)
            hidden_states = hidden_probs > np.random.rand(*hidden_probs.shape)

            visible_activations = np.dot(hidden_states, np.transpose(self.weights)) + self.visible_bias
            visible_probs = self.sigmoid(visible_activations)
            visible = visible_probs > np.random.rand(*visible_probs.shape)
        return visible

# Function to create a noisy version of the input pattern
def create_noisy_input(pattern, noise_level=0.1):
    noisy_exemplars = pattern.copy()
    for idx in range(len(noisy_exemplars)):
        if np.random.random() < noise_level:
            noisy_exemplars[idx] = 1 - noisy_exemplars[idx]
    return noisy_exemplars

# Function to compute the Jaccard similarity between two binary arrays
def jaccard_similarity(a, b):
    intersection = np.sum(np.logical_and(a == 1, b == 1))
    union = np.sum(np.logical_or(a == 1, b == 1))
    return intersection / union if union > 0 else 0

def display_pattern(pattern):
    plt.imshow(pattern.reshape(10, 10), cmap='gray')
    plt.show()

def train_with_best_k():
    rbm = RestrictedBoltzmannMachine(100, 100)
    rbm.train([exemplar for exemplar, _ in exemplars_with_labels], k=best_k)
    
    for exemplar, label in exemplars_with_labels:
        for i in range(num_test_sequences):
            noisy_exemplars = create_noisy_input(exemplar, noise_level=noise_input)
            recalled_pattern = rbm.recall(noisy_exemplars, n_gibbs=num_gibbs)

            # Display the original, noisy, and recalled patterns
            print(f"Sequence {i+1} with {num_gibbs} Gibbs sampling cycles")

            print("Original pattern:")
            display_pattern(exemplar)

            print("Noisy input:")
            display_pattern(noisy_exemplars)

            print("Recalled pattern:")
            display_pattern(recalled_pattern)
            print("-" * 20)

# Add labels to the exemplars
exemplars_with_labels = list(zip(exemplars, range(len(exemplars))))

# Train the RBM with different values of k
k_values = range(1, 36)
best_k = None
best_performance = 0

correct_reconstructions = []
correct_reconstructions_count = 0
total_reconstructions_count = 0

num_test_sequences = 2
similarity_threshold = 0.15
num_gibbs = 20
noise_input = 0.1

for k in k_values:
    rbm = RestrictedBoltzmannMachine(100, 100)
    rbm.train([exemplar for exemplar, _ in exemplars_with_labels], learning_rate=0.001, n_epochs=60, batch_size=1, k=k)

    for exemplar, label in exemplars_with_labels:
        for i in range(num_test_sequences):
            noisy_exemplars = create_noisy_input(exemplar, noise_level=noise_input)
            recalled_pattern = rbm.recall(noisy_exemplars, n_gibbs=num_gibbs)
            total_reconstructions_count += 1
            similarity = jaccard_similarity(recalled_pattern, exemplar)
            if similarity >= similarity_threshold:
                correct_reconstructions_count += 1

    performance = correct_reconstructions_count / total_reconstructions_count
    correct_reconstructions.append(performance)
    
    print(f"k: {k}, Frequency of correct reconstructions: {performance:.2f}")

    if performance > best_performance:
        best_performance = performance
        best_k = k

print(f"Optimal k: {best_k}, Best performance: {best_performance:.2f}")

# Plot the relationship between k and the frequency of correct reconstructions
plt.plot(k_values, correct_reconstructions, marker='o')
plt.xlabel('Number of Gibbs sampling cycles (k)')
plt.ylabel('Frequency of correct reconstructions')
plt.title('Performance of the RBM with varying k')
plt.grid()
plt.show()

# Train the RBM with the best k value
train_with_best_k()