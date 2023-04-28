import numpy as np


class RestrictedBoltzmannMachine:
    def __init__(self, n_visible: int, n_hidden: int):
        '''
        Initialize 2 Layer Restricted Boltzmann Machine with a specified number of hidden nodes and visible nodes.

        Arguments:
          n_visible (int) - Number of nodes in the visible layer
          n_hidden  (int) - Number of nodes in the hidden layer
        '''
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.W = np.random.normal(0.0, 0.1, (n_visible, n_hidden)) # Set to this for testing purposes - np.array([[2, -1, 1], [-2, 0, -1]])
        self.a = np.random.normal(0.0, 0.1, n_visible)
        self.b = np.zeros((n_hidden))

        
    def sigmoid(self, x: float):
        '''
        Activation function for output of nodes in either layer.
        '''
        return 1 / (1 + np.exp(-x))
    
    def sample_hidden_layer(self, v: np.array):
        '''
        Generate unbiased sample of activated hidden units by switching on/off hidden units according to
        the conditional probabilities of h = 1 given v.
        '''
        # Calculate hidden probabilities
        activation = np.vectorize(self.sigmoid)
        hidden_probas = activation(np.dot(v, self.W) + self.b)

        # Sample hidden layer output from calculated probabilties
        random_vars = np.random.uniform(size=self.n_hidden) # Set to this for testing purposes - np.array([0.87, 0.14, 0.64]) 
        return (hidden_probas > random_vars).astype(int)

    def sample_visible_layer(self, h: np.array):
        '''
        Generate unbiased sample of activated visible units by them switching on/off according to
        the conditional probabilities of v = 1 given h.
        '''

        # Calculate visible probabilities
        activation = np.vectorize(self.sigmoid)
        visible_probas = activation(np.dot(h, self.W.T) + self.a)

        # Sample visible layer output from calculated probabilties
        random_vars = np.random.uniform(size=self.n_visible) # Set to this for testing purposes - np.array([0.25, 0.72])
        return (visible_probas > random_vars).astype(int)

    def train(self, patterns: np.array, eta: float =0.01, epochs:int = 10):
        '''
        Train the Restricted Boltzmann Machine utilizing Hinton's Approximation Method.
        '''
        for _ in range(epochs):
            for v in patterns:
                #########################################################################
                # Positive Gradient Phase
                #########################################################################

                # Sample hidden activation units from training example
                h = self.sample_hidden_layer(v) 

                # Compute the outer product of v and h and call this the positive gradient
                pos_grad = np.outer(v, h)

                #########################################################################
                # Negative Gradient Phase
                #########################################################################
                
                # From h, sample a reconstruction v' of the visible units
                v_prime = self.sample_visible_layer(h)

                # Then resample activations h' from v'
                h_prime = self.sample_hidden_layer(v_prime)

                # Compute the outer product of v' and h' and call this the negative gradient
                neg_grad = np.outer(v_prime, h_prime)

                #########################################################################
                # Weight Update
                #########################################################################
                self.W += eta*(pos_grad - neg_grad) # Update to the weight matrix W, will be the positive gradient minus the negative gradient, times some learning rate
                self.a += eta*(v - v_prime) # Update to the visible bias
                self.b += eta*(h - h_prime) # Update to the hidden bias

    def reconstruct(self, pattern: np.array, iters: int =10):
        '''
        Reconstruct noisy input until converging to one of the exemplars.
        '''
        v = pattern.copy()
        for _ in range(iters):
            h = self.sample_hidden_layer(v)
            v = self.sample_visible_layer(h)
        
        return v