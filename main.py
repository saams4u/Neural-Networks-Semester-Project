
# Import helper functions
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Import exemplar data
from data import exemplars

# Import RBM class
from rbm import RBM

# Import to train and evaluate RBM
from execute import train_and_evaluate


# Define the network parameters
num_classes = 8 
iters = 10
num_samples = 100
test_size = 0.1
num_hidden = 30
noise_factor = 0.1
learning_rate = 0.001
epochs = 20

if __name__ == '__main__':

    print(f"Training with parameters: \n",
            f"- recon_iters: {iters}\n",
            f"- num_samples: {num_samples}\n"
            f" - test_size: {test_size}\n"
            f" - num_hidden: {num_hidden}\n"
            f" - noise_factor: {noise_factor}\n"
            f" - learning_rate: {learning_rate}\n"
            f" - epochs: {epochs}\n")

    # Instantiate the RBM network
    rbm = RBM(num_visible=100, num_hidden=num_hidden) 

    # Generate noisy samples to train on
    samples = rbm.generate_samples(exemplars, num_samples=num_samples, noise_factor=noise_factor)

    # Generate labels based on the index of the exemplars
    num_exemplars_per_class = len(samples) // num_classes
    labels = np.repeat(np.arange(num_classes), num_exemplars_per_class)

    # Combine exemplars with their labels
    exemplars_with_labels = list(zip(samples, labels))

    # Split data into training and testing sets using the custom function
    train_data, (test_exemplars, test_labels) = rbm.split_data(exemplars_with_labels, test_size=test_size, random_seed=42, original_exemplars=exemplars)

    # Separate digit arrays and labels in the testing set
    test_data = test_exemplars, test_labels

    # Test the performance of the RBM with different numbers of Gibbs cycles
    gibbs_cycle_list = range(1, 11, 1)
    correct_reconstructions = train_and_evaluate(rbm,
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
                                                output_file=None)

    print("\nFrequencies of correct reconstructions:", correct_reconstructions)