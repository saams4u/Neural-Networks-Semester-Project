
# Import helper functions
import numpy as np
import matplotlib.pyplot as plt


# Implement the Restricted Boltzmann Machine class
class RBM:

    def __init__(self, num_visible, num_hidden):
        '''
        Initialize 2 Layer Restricted Boltzmann Machine with a specified number of hidden nodes and visible nodes.

        Arguments:
          n_visible (int) - Number of nodes in the visible layer
          n_hidden  (int) - Number of nodes in the hidden layer
        '''
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

    def train(self, data, learning_rate=0.01, epochs=20, gibbs_steps=1, validation_data=None):
        '''
        Train the Restricted Boltzmann Machine utilizing Hinton's Approximation Method.
        '''
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

            # training_error = []
            
            # if validation_data is not None:
            #     # Adjust learning rate based on learning rate decay
            #     learning_rate *= self._learning_rate_decay(epoch)

    def _sigmoid(self, x):
        '''
        Activation function for output of nodes in either layer.
        '''
        return 1.0 / (1.0 + np.exp(-x))

    def reconstruct(self, inputs, iters=10):
        results = []
        for input in inputs:
            v = input.copy()
            for _ in range(iters):
                h = self.sample_hidden_layer(v)
                v = self.sample_visible_layer(h)
            results += [v]
        return results

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
