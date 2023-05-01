
# Import helper functions
import os
import time
from itertools import product

# Import exemplar data
from data import exemplars

# Import plot functions
from plot_exemplars import plot_for_two_labels, plot_for_all_labels, plot_rbm_performance

# Import to train and evaluate RBM
from train_and_evaluate import train_and_evaluate_gibbs_cycles


# Implement the Restricted Boltzmann Machine class
class RBM:

    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.random.normal(scale=0.01, size=(num_visible, num_hidden))
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def train(self, data, learning_rate=0.001, epochs=20, gibbs_steps=1, validation_data=None):
        for epoch in range(epochs):
            for sample in data:
                # Initialize visible state
                initial_visible_states = np.copy(sample)

                # Perform Gibbs sampling for the specified number of steps
                for _ in range(gibbs_steps):
                    # Forward pass
                    hidden_probs = self._sigmoid(np.dot(initial_visible_states, self.weights) + self.hidden_bias)
                    hidden_states = (hidden_probs > np.random.random(self.num_hidden)).astype(int)

                    # Backward pass (Gibbs sampling)
                    visible_probs = self._sigmoid(np.dot(hidden_states, self.weights.T) + self.visible_bias)
                    visible_states = (visible_probs > np.random.random(self.num_visible)).astype(int)

                # Update weights and biases
                final_hidden_probs = self._sigmoid(np.dot(visible_states, self.weights) + self.hidden_bias)
                positive_grad = np.outer(initial_visible_states, hidden_probs)
                negative_grad = np.outer(visible_states, final_hidden_probs)

                weight_update = learning_rate * (positive_grad - negative_grad)
                visible_bias_update = learning_rate * (initial_visible_states - visible_states)
                hidden_bias_update = learning_rate * (hidden_probs - final_hidden_probs)

                self.weights += weight_update
                self.visible_bias += visible_bias_update
                self.hidden_bias += hidden_bias_update

            if validation_data is not None:
                # Adjust learning rate based on learning rate decay
                learning_rate *= self._learning_rate_decay(epoch)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def reconstruct(self, visible):
        hidden_prob = self._sigmoid(np.dot(visible, self.weights) + self.hidden_bias)
        hidden = (np.random.rand(*hidden_prob.shape) < hidden_prob).astype(int)
        visible_prob = self._sigmoid(np.dot(hidden, self.weights.T) + self.visible_bias)
        return (np.random.rand(*visible_prob.shape) < visible_prob).astype(int)

    def _reconstruction_error(self, data):
        reconstructed_data = self.reconstruct(data)
        return np.mean(np.square(data - reconstructed_data))

    def generate_samples(self, exemplars, num_samples=50, noise_factor=0.2):
        samples = []
        for exemplar in exemplars:
            for _ in range(num_samples):
                noiseless_exemplar = self.add_noise(exemplar, noise_factor)
                samples.append(noiseless_exemplar)
        return samples

    def add_noise(self, exemplar, noise_factor=0.2):
        noise = np.random.uniform(-noise_factor, noise_factor, size=exemplar.shape)
        noisy_exemplar = np.clip(exemplar + noise, -1, 1)
        return noisy_exemplar

    def plot_digit(self, digit_array):
        plt.imshow(digit_array.reshape(10, 10), cmap="gray")
        plt.axis("off")

    def split_data(self, data, test_size=None, random_seed=None, original_exemplars=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(data)
        split_index = int(len(data) * (1 - test_size))
        train_data = data[:split_index]
        test_data = data[split_index:]
        test_exemplars = [original_exemplars[idx] for _, idx in test_data]
        test_labels = [lbl for _, lbl in test_data]
        return train_data, (test_exemplars, test_labels)

    def _learning_rate_decay(self, epoch, decay_rate=0.9):
        return decay_rate ** epoch

num_classes = 8  # Given there are 8 classes (0 to 7)
num_samples_options = [120, 240, 480, 960]
test_size_options = [0.1, 0.2, 0.3, 0.4]
num_hidden_options = [5, 10, 15, 20]
noise_factor_options = [0.3, 0.5, 0.7, 0.9]
learning_rate_options = [0.001, 0.003, 0.005, 0.007]
epochs_options = [50, 100, 150, 200]
gibbs_cycle_list = [1, 5, 10, 20, 50, 100]

best_params = None
best_accuracy = -1

with open('grid_search_output.txt', 'w') as output_file:
    for num_samples, test_size, num_hidden, noise_factor, learning_rate, epochs in product(num_samples_options, 
                                                                                           test_size_options, 
                                                                                           num_hidden_options, 
                                                                                           noise_factor_options, 
                                                                                           learning_rate_options, 
                                                                                           epochs_options):

        print(f"Training with parameters: num_samples={num_samples}, test_size={test_size}, num_hidden={num_hidden}, noise_factor={noise_factor}, learning_rate={learning_rate}, epochs={epochs}", file=output_file)

        # Instantiate the RBM network
        rbm = RBM(num_visible=100, num_hidden=num_hidden)

        # Generate noisy samples to train on
        samples = rbm.generate_samples(exemplars, num_samples=num_samples, noise_factor=noise_factor)

        # Normalize the generated noisy samples
        exemplars_normalized = [sample * 0.5 + 0.5 for sample in samples]

        # Generate labels based on the index of the exemplars
        num_exemplars_per_class = len(exemplars_normalized) // num_classes
        labels = np.repeat(np.arange(num_classes), num_exemplars_per_class)

        # Combine exemplars with their labels
        exemplars_with_labels = list(zip(exemplars_normalized, labels))

        # Split data into training and testing sets using the custom function
        train_data, (test_exemplars, test_labels) = rbm.split_data(exemplars_with_labels, 
                                                                   test_size=test_size, 
                                                                   random_seed=42, 
                                                                   original_exemplars=exemplars)

        # Separate digit arrays and labels in the testing set
        test_data = test_exemplars, test_labels

        # Test the performance of the RBM with the current set of parameters
        correct_reconstructions = train_and_evaluate_gibbs_cycles(gibbs_cycle_list, 
                                                                  test_data, 
                                                                  noise_factor, 
                                                                  epochs, 
                                                                  learning_rate)
        current_accuracy = max(correct_reconstructions)

        print(f"Current accuracy: {current_accuracy}", file=output_file)

        # Update the best parameters if the current accuracy is higher than the previous best
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_params = {
                'num_samples': num_samples,
                'test_size': test_size,
                'num_hidden': num_hidden,
                'noise_factor': noise_factor,
                'learning_rate': learning_rate,
                'epochs': epochs
            }

print("Best parameters:", best_params)
print("Best accuracy:", best_accuracy)