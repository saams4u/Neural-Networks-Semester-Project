
import numpy as np
import random
import itertools
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

    # Update the train function to include early stopping and adaptive learning rate (Adam)
    def train(self, patterns, learning_rate=0.001, n_epochs=20, batch_size=4, k=1, validation_split=0.1,
              early_stopping_rounds=5, beta1=0.9, beta2=0.999, epsilon=1e-8, gibbs=100, noise_input=0.1):
        
        # Split the dataset into training and validation sets
        n_validation = int(len(patterns) * validation_split)
        training_patterns = patterns[:-n_validation]
        validation_patterns = patterns[-n_validation:]

        best_weights = self.weights.copy()
        best_visible_bias = self.visible_bias.copy()
        best_hidden_bias = self.hidden_bias.copy()
        best_epoch = 0
        best_performance = 0
        stopping_counter = 0

        m_weights = np.zeros_like(self.weights)
        v_weights = np.zeros_like(self.weights)
        m_visible_bias = np.zeros_like(self.visible_bias)
        v_visible_bias = np.zeros_like(self.visible_bias)
        m_hidden_bias = np.zeros_like(self.hidden_bias)
        v_hidden_bias = np.zeros_like(self.hidden_bias)

        for epoch in range(n_epochs):
            np.random.shuffle(training_patterns)
            for batch in range(0, len(training_patterns), batch_size):
                data = training_patterns[batch:batch + batch_size]

                # Positive phase: compute hidden probabilities given input data (visible units)
                pos_hidden_activations = np.dot(data, self.weights) + self.hidden_bias
                pos_hidden_probs = self.sigmoid(pos_hidden_activations)
                pos_hidden_states = pos_hidden_probs > np.random.rand(*pos_hidden_probs.shape)
                pos_associations = np.dot(np.transpose(data), pos_hidden_probs)

                # Negative phase: perform k-step Gibbs sampling to generate visible units from hidden units
                neg_visible_activations = np.dot(pos_hidden_states, np.transpose(self.weights)) + self.visible_bias
                neg_visible_probs = self.sigmoid(neg_visible_activations)
                neg_visible_probs[:, -1] = 1  # Fix the bias unit
                neg_hidden_activations = np.dot(neg_visible_probs, self.weights) + self.hidden_bias
                neg_hidden_probs = self.sigmoid(neg_hidden_activations)
                neg_associations = np.dot(np.transpose(neg_visible_probs), neg_hidden_probs)

                # Update the weights and biases using Adam
                m_weights = beta1 * m_weights + (1 - beta1) * (pos_associations - neg_associations) / batch_size
                v_weights = beta2 * v_weights + (1 - beta2) * ((pos_associations - neg_associations) / batch_size)**2
                m_hat_weights = m_weights / (1 - beta1**(epoch + 1))
                v_hat_weights = v_weights / (1 - beta2**(epoch + 1))
                self.weights += learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)

                m_visible_bias = beta1 * m_visible_bias + (1 - beta1) * np.mean(data - neg_visible_probs, axis=0)
                v_visible_bias = beta2 * v_visible_bias + (1 - beta2) * np.mean(data - neg_visible_probs, axis=0)**2
                m_hat_visible_bias = m_visible_bias / (1 - beta1**(epoch + 1))
                v_hat_visible_bias = v_visible_bias / (1 - beta2**(epoch + 1))
                self.visible_bias += learning_rate * m_hat_visible_bias / (np.sqrt(v_hat_visible_bias) + epsilon)

                m_hidden_bias = beta1 * m_hidden_bias + (1 - beta1) * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)
                v_hidden_bias = beta2 * v_hidden_bias + (1 - beta2) * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)**2
                m_hat_hidden_bias = m_hidden_bias / (1 - beta1**(epoch + 1))
                v_hat_hidden_bias = v_hidden_bias / (1 - beta2**(epoch + 1))
                self.hidden_bias += learning_rate * m_hat_hidden_bias / (np.sqrt(v_hat_hidden_bias) + epsilon)

            # Evaluate the model on the validation set and implement early stopping
            total_similarity = 0
            for exemplar in validation_patterns:
                noisy_exemplars = create_noisy_input(exemplar, noise_level=noise_input)
                recalled_pattern = self.recall(noisy_exemplars, n_gibbs=gibbs)
                distance = hamming_distance(recalled_pattern, exemplar)
                total_similarity += distance

            performance = total_similarity / len(validation_patterns)
            print(f"Epoch: {epoch + 1}, Validation accuracy: {performance / 100:.2f}")

            if performance > best_performance:
                best_weights = self.weights.copy()
                best_visible_bias = self.visible_bias.copy()
                best_hidden_bias = self.hidden_bias.copy()
                best_epoch = epoch
                best_performance = performance
                stopping_counter = 0
            else:
                stopping_counter += 1

            if stopping_counter >= early_stopping_rounds:
                print(f"Early stopping at epoch {epoch + 1}. Best performance at epoch {best_epoch + 1}: {best_performance / 100:.2f}")
                self.weights = best_weights
                self.visible_bias = best_visible_bias
                self.hidden_bias = best_hidden_bias
                break

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
    noisy_exemplars = np.copy(pattern)
    for idx in range(len(noisy_exemplars)):
        if np.random.random() < noise_level:
            noisy_exemplars[idx] = 1 - noisy_exemplars[idx]
    return noisy_exemplars

# Function to compute the Hamming distance between two binary arrays
def hamming_distance(a, b):
    return np.sum(a != b)

def maxnet(input_values, epsilon=0.01, max_iterations=1000):
    values = input_values.copy()
    for _ in range(max_iterations):
        values = values - epsilon * (np.sum(values) - values)
        values[values < 0] = 0

        if np.count_nonzero(values) == 1:
            break

    return np.argmax(values)

def display_patterns(original, noisy, reconstructed):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(original.reshape(10, 10), cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(noisy.reshape(10, 10), cmap='gray')
    axes[1].set_title("Noisy")
    axes[1].axis('off')
    axes[2].imshow(reconstructed.reshape(10, 10), cmap='gray')
    axes[2].set_title("Reconstructed")
    axes[2].axis('off')
    plt.show()

# Add labels to the exemplars
exemplars_with_labels = list(zip(exemplars, range(len(exemplars))))

# Initialize search parameters
num_test_sequences = 2
hamming_threshold = 80  # Updated threshold for Hamming distance
correct_reconstructions = []
noise = 0.1
gibbs_values = range(1, 201, 20)

best_performance = 0
best_threshold = None
best_noise_level = None
best_gibbs = None

for gibbs in gibbs_values:
    rbm = RestrictedBoltzmannMachine(100, 100)
    rbm.train([exemplar for exemplar, _ in exemplars_with_labels], gibbs=gibbs)

    correct_reconstructions_count = 0
    total_reconstructions_count = 0

    for exemplar, label in exemplars_with_labels:
        for i in range(num_test_sequences):
            noisy_exemplars = create_noisy_input(exemplar, noise_level=noise)
            recalled_pattern = rbm.recall(noisy_exemplars)
            total_reconstructions_count += 1

            # Compute Hamming distances between the recalled pattern and all exemplars
            distances = np.array([hamming_distance(recalled_pattern, exemplar) for exemplar, _ in exemplars_with_labels])

            # Apply MAXNET to find the exemplar with the minimum Hamming distance
            min_distance_idx = maxnet(-distances)  # Note that we use the negative of the distances, as MAXNET selects the largest value

            # Check if the selected exemplar has a Hamming distance below the threshold
            if distances[min_distance_idx] >= hamming_threshold:
                correct_reconstructions_count += 1
                print(f"Exemplar: {label}, Test sequence: {i + 1}, Similarity: {distances[min_distance_idx] / 100:.2f}")
                display_patterns(exemplar, noisy_exemplars, recalled_pattern)

    performance = correct_reconstructions_count / total_reconstructions_count
    print(f"Gibbs cycles: {gibbs}, Noise level: {noise:.2f}, Threshold: {hamming_threshold}, Frequency of correct reconstructions: {performance:.2f} \n")
    
    correct_reconstructions.append(performance)
    
if performance > best_performance:
    best_performance = performance
    best_threshold = hamming_threshold
    best_noise_level = noise
    best_gibbs = gibbs

# Plot the relationship between the number of Gibbs cycles and the frequency of correct reconstructions
plt.plot(gibbs_values, correct_reconstructions, marker='o')
plt.xlabel('Number of Gibbs sampling cycles')
plt.ylabel('Frequency of correct reconstructions')
plt.title('Measuring the Performance of the RBM')
plt.grid()
plt.show()