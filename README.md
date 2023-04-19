
**HOPFIELD NETWORK**

A Hopfield Network is a type of recurrent artificial neural network that is designed to store and recall patterns. It can be used as an associative memory, which is capable of correcting corrupted or noisy input patterns by recalling the closest stored pattern. The network is characterized by its symmetric weight matrix, and it converges to a stable state after a number of iterations.


PROJECT DESCRIPTION

Hopfield Networks: For this assignment, implement the Hopfield Network that is based on 8 exemplars, each 100 elements in length. These exemplars are to be displayed in a 10x10 rasterized array and should depict the first 8 numerals: 0 – 7.  The output of this program should display the sequence of iterates displayed in the 10x10 array in similar fashion as the examples shown in Lippman’s paper "Introduction to Neural Networks". Display at least two sequences each sequence starting with a noisy version of one of the exemplars. Feel free to experiment.


CODE OVERVIEW

In the provided code, a Hopfield Network is implemented in Python, with the following main components:


Hopfield Network Class

The `HopfieldNetwork` class represents the structure and behavior of a Hopfield Network. It contains:

* `__init__(self, size)`: Initializes a new Hopfield Network with a given size. The weight matrix is initialized with zeros.
* `train(self, patterns)`: Trains the network using Hebbian learning rule by adjusting the weights based on the outer product of the input patterns.
* `recall(self, pattern, max_iterations=100)`: Recalls a stored pattern from a noisy input pattern by updating the network state iteratively until convergence or the maximum number of iterations is reached.


Utility Functions

* `create_noisy_input(pattern, noise_level)`: Creates a noisy version of an input pattern by flipping a specified percentage of the pixels.
* `display_pattern(pattern)`: Displays a pattern as a 10x10 rasterized array using matplotlib.


Training and Testing the Network

The Hopfield Network is trained using a set of exemplar patterns provided by the data module. The network is then tested with noisy inputs generated from the exemplars, and the results are displayed using the display_pattern function.

1. Instantiate the Hopfield Network with a size of 100.
2. Train the network using the exemplar patterns.
3. Test the network using noisy inputs generated from the exemplars with varying noise levels.
4. Display the original, noisy, and recalled patterns.


SUMMARY 

In summary, the Hopfield Network implemented in the provided code is a simple and effective model for storing and recalling patterns. The network is capable of correcting corrupted or noisy input patterns by recalling the closest stored pattern. The code demonstrates the process of training the network, generating noisy inputs, and recalling stored patterns using the trained network.


**RESTRICTED BOLTZMANN MACHINE**

A Restricted Boltzmann Machine (RBM) is a type of generative stochastic artificial neural network, consisting of two layers: a visible layer and a hidden layer. The connections between the nodes are undirected, and there are no connections within the same layer. RBMs can learn a probability distribution over their input patterns and are widely used for feature learning, dimensionality reduction, and pattern recognition tasks.


CODE OVERVIEW

In the provided code, a Restricted Boltzmann Machine is implemented in Python, with the following main components:


Restricted Boltzmann Machine Class

The `RestrictedBoltzmannMachine` class represents the structure and behavior of an RBM. It contains:

__init__(self, n_visible, n_hidden): Initializes a new RBM with a given number of visible and hidden nodes, and initializes the weights, visible bias, and hidden bias.
sigmoid(self, x): The sigmoid activation function.
train(self, patterns, learning_rate=0.01, n_epochs=10, batch_size=1, k=1): Trains the RBM using Hinton's contrastive divergence method with a given learning rate, number of epochs, batch size, and number of Gibbs sampling cycles (k).
recall(self, pattern, n_gibbs=100): Recalls the original pattern from a noisy input using Gibbs sampling.


Utility Functions

* `create_noisy_input(pattern, noise_level=0.1)`: Creates a noisy version of an input pattern by flipping a specified percentage of the pixels.
* `jaccard_similarity(a, b)`: Computes the Jaccard similarity between two binary arrays.
* `display_pattern(pattern)`: Displays a pattern as a 10x10 rasterized array using matplotlib.


Training, Optimizing, and Testing the Network

The RBM is trained using a set of exemplar patterns provided by the data module. The optimal number of Gibbs sampling cycles (k) is determined by comparing the performance of the RBM with different k values. The network is then tested with noisy inputs generated from the exemplars, and the results are displayed using the display_pattern function.

1. Instantiate the RBM with a size of 100 visible and 100 hidden nodes.
2. Train the RBM using the exemplar patterns and Hinton's contrastive divergence method.
3. Determine the optimal k value by comparing the performance of the RBM with different k values.
4. Train the RBM with the best k value.
5. Test the network using noisy inputs generated from the exemplars.
6. Display the original, noisy, and recalled patterns.


SUMMARY

In summary, the Restricted Boltzmann Machine implemented in the provided code is a powerful and flexible generative model for learning patterns, features, and distributions. The RBM is trained using Hinton's contrastive divergence method, and the optimal number of Gibbs sampling cycles is determined by comparing the performance of the RBM with different k values. The code demonstrates the process of training the RBM, optimizing its performance, generating noisy inputs, and recalling stored patterns using the trained network.
