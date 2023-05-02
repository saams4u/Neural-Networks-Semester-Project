

# Import helper functions
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Import exemplar data
from data import exemplars

# Import to train and evaluate RBM
from train_and_evaluate import train_and_evaluate_gibbs_cycles


# Implement the Restricted Boltzmann Machine class
class RBM:

    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.W = np.random.normal(0.0, 0.1, (num_visible, num_hidden))
        self.a = np.zeros(num_visible)
        self.b = np.zeros(num_hidden)

    def sample_visible_layer(self, h):
        # Calculate visible probabilities
        activation = np.vectorize(self._sigmoid)
        visible_probas = activation(np.dot(h, self.W.T) + self.a)

        # Sample visible layer output from calculated probabilties
        random_vars = np.random.uniform(size=self.num_visible) # Set to this for testing purposes - np.array([0.25, 0.72])
        return (visible_probas > random_vars).astype(int)

    def sample_hidden_layer(self, v):
        # Calculate hidden probabilities
        activation = np.vectorize(self._sigmoid)
        hidden_probas = activation(np.dot(v, self.W) + self.b)

        # Sample hidden layer output from calculated probabilties
        random_vars = np.random.uniform(size=self.num_hidden) # Set to this for testing purposes - np.array([0.87, 0.14, 0.64]) 
        return (hidden_probas > random_vars).astype(int)

    def train(self, data, learning_rate=0.01, epochs=20, gibbs_steps=None, validation_data=None):
        for epoch in range(epochs):
            for v in data:
                # Take training sample, and compute probabilties of hidden units and the
                # sample a hidden activation vector from this probability distribution
                h = self.sample_hidden_layer(v)

                # Compute the outer product of v and h and call this the positive gradient
                pos_grad = np.outer(v, h)

                # Perform gibbs sampling in for loop so that it is a tunable hyperparameter
                h_prime = h.copy()
                for _ in range(gibbs_steps):
                    # From h, sample a reconstruction v' of the visible units
                    v_prime = self.sample_visible_layer(h_prime)

                    # Then resample activations h' from this
                    h_prime = self.sample_hidden_layer(v_prime)

                # Compute the outer product of v' and h' and call this the negative gradient
                neg_grad = np.outer(v_prime, h_prime)
               
                self.W += learning_rate*(pos_grad - neg_grad) # Update to the weight matrix W, will be the positive gradient minus the negative gradient, times some learning rate
                self.a += learning_rate*(v - v_prime) # Update to the visible bias
                self.b += learning_rate*(h - h_prime) # Update to the hidden bias
            
            if validation_data is not None:
                # Adjust learning rate based on learning rate decay
                learning_rate *= self._learning_rate_decay(epoch)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def reconstruct(self, visible, iters=10):
        v = visible.copy()
        for _ in range(iters):
            h = self.sample_hidden_layer(v)
            v = self.sample_visible_layer(h)
        return v
    
    def _reconstruction_error(self, data, iters=10):
        reconstructed_data = self.reconstruct(data, iters=iters)
        return np.mean(np.square(data - reconstructed_data))

    def generate_samples(self, exemplars, num_samples=50, noise_factor=0.2):
        samples = []
        for exemplar in exemplars:
            for _ in range(num_samples):
                noisy_exemplar = self.add_noise(exemplar, noise_factor)
                samples.append(noisy_exemplar)
        return samples

    def add_noise(self, exemplar, noise_factor=0.2):
        noisy_exemplar = exemplar.copy()
        for i in range(len(noisy_exemplar)):
            proba = np.random.random()
            if proba > noise_factor:
                continue
            noisy_exemplar[i] = 0 if noisy_exemplar[i] == 1 else 1

        ## Removing label from exemplar
        # noisy_exemplar = noisy_exemplar.reshape(10, 12)
        # noisy_exemplar[:, -2:] = 0
        
        # noise = np.random.uniform(-noise_factor, noise_factor, size=exemplar.shape)
        # noisy_exemplar = np.clip(exemplar + noise, -1, 1)
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

num_classes           = 8
num_samples           = 50
test_size_options     = [0.1, 0.2]
num_hidden_options    = [20, 40, 60, 80, 100, 120]
noise_factor_options  = [0.2, 0.3, 0.5, 0.7, 0.9]
learning_rate_options = [0.005, 0.01]
epochs_options        = [100, 200, 300]
gibbs_cycle_list      = [1, 2, 3]
num_reconstructions   = [1, 5, 10]

best_params = None
best_accuracy = -1

if __name__ == '__main__':
    start_time = time.time()  # Get the start time

    # with open('grid_search/trial_01.txt', 'w') as output_file:
    for test_size, num_hidden, noise_factor, learning_rate, epochs, iters in product(test_size_options, 
                                                                              num_hidden_options, 
                                                                              noise_factor_options, 
                                                                              learning_rate_options, 
                                                                              epochs_options,
                                                                              num_reconstructions):
        print(f"Training with parameters: \n",
                f" - reconstruction_iters: {iters}\n",
                f" - num_samples: {num_samples}\n"
                f" - test_size: {test_size}\n"
                f" - num_hidden: {num_hidden}\n"
                f" - noise_factor: {noise_factor}\n"
                f" - learning_rate: {learning_rate}\n"
                f" - epochs: {epochs}\n")

        # Instantiate the RBM network
        rbm = RBM(num_visible=100, num_hidden=num_hidden)

        # Generate noisy samples to train on
        samples = rbm.generate_samples(exemplars, num_samples=num_samples, noise_factor=noise_factor)

        # Normalize the generated noisy samples
        # exemplars_normalized = [sample * 0.5 + 0.5 for sample in samples]

        # Generate labels based on the index of the exemplars
        num_exemplars_per_class = len(samples) // num_classes
        labels = np.repeat(np.arange(num_classes), num_exemplars_per_class)

        # Combine exemplars with their labels
        exemplars_with_labels = list(zip(samples, labels))

        # Split data into training and testing sets using the custom function
        train_data, (test_exemplars, test_labels) = rbm.split_data(exemplars_with_labels, 
                                                                test_size=test_size, 
                                                                random_seed=42, 
                                                                original_exemplars=exemplars)

        # Separate digit arrays and labels in the testing set
        test_data = test_exemplars, test_labels

        # Test the performance of the RBM with the current set of parameters
        correct_reconstructions = train_and_evaluate_gibbs_cycles(rbm,
                                                                iters,
                                                                num_samples,
                                                                test_size,
                                                                num_hidden,  
                                                                train_data,
                                                                test_data,
                                                                test_exemplars,
                                                                labels,
                                                                test_labels, 
                                                                noise_factor, 
                                                                epochs,
                                                                gibbs_cycle_list, 
                                                                learning_rate,
                                                                None)
        current_accuracy = max(correct_reconstructions)

        print(f"Current accuracy: {current_accuracy}\n")

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

    end_time = time.time()  # Get the end time
    time_taken = end_time - start_time  # Calculate the time taken for the grid search

    print("Best parameters:\n"
          f" - num_samples: {best_params['num_samples']}\n"
          f" - test_size: {best_params['test_size']}\n"
          f" - num_hidden: {best_params['num_hidden']}\n"
          f" - noise_factor: {best_params['noise_factor']}\n"
          f" - learning_rate: {best_params['learning_rate']}\n"
          f" - epochs: {best_params['epochs']}\n")
    print("Best accuracy:", best_accuracy)
    
    print(f"Grid search took {time_taken:.2f} seconds")