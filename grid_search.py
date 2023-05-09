
# Import helper functions
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Import exemplar data
from utils import exemplars

# Import RBM class
from rbm import RBM

# Import to train and evaluate RBM
from execute import train_and_evaluate


num_classes           = 8
num_samples           = 100
test_size_options     = [0.1]
num_hidden_options    = [10, 20, 30]
noise_factor_options  = [0.1, 0.3, 0.5, 0.7, 0.9]
learning_rate_options = [0.001]
epochs_options        = [20, 40, 60]
gibbs_cycle_list      = [1, 2, 3, 4, 5, 6,7 , 8, 9, 10]
num_reconstructions   = [10, 20, 40, 80]

best_params = None
best_accuracy = -1

start_time = time.time()  # Get the start time

with open('grid_search/trial_01.txt', 'w') as output_file:
    for test_size, num_hidden, noise_factor, learning_rate, epochs, iters in product(test_size_options, 
                                                                            num_hidden_options, 
                                                                            noise_factor_options, 
                                                                            learning_rate_options, 
                                                                            epochs_options,
                                                                            num_reconstructions):
        print(f"\nTraining with parameters: \n",
                f"- recon_iters: {iters}\n",
                f"- num_samples: {num_samples}\n"
                f" - test_size: {test_size}\n"
                f" - num_hidden: {num_hidden}\n"
                f" - noise_factor: {noise_factor}\n"
                f" - learning_rate: {learning_rate}\n"
                f" - epochs: {epochs}\n", file=output_file)

        # Instantiate the RBM network
        rbm = RBM(num_visible=100, num_hidden=num_hidden)

        # Generate noisy samples to train on
        samples = rbm.generate_samples(exemplars, num_samples=num_samples, noise_factor=noise_factor)

        # Generate labels based on the index of the exemplars
        num_exemplars_per_class = len(samples) // num_classes
        labels = np.repeat(np.arange(num_classes), num_exemplars_per_class)

        # Combine exemplars with their labels
        samples_with_labels = list(zip(samples, labels))

        # Split data into training and testing sets using the custom function
        train_data, (test_exemplars, test_labels) = rbm.split_data(samples_with_labels, 
                                                                test_size=test_size, 
                                                                random_seed=42, 
                                                                original_exemplars=exemplars)

        # Separate digit arrays and labels in the testing set
        test_data = test_exemplars, test_labels

        # Test the performance of the RBM with the current set of parameters
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
                                                    output_file)

        current_accuracy = max(correct_reconstructions)

        print(f"Current accuracy: {current_accuracy}\n", file=output_file)

        # Update the best parameters if the current accuracy is higher than the previous best
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_params = {
                'recon_iters': iters,
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
        f" - recon_iters: {best_params['recon_iters']}\n"
        f" - num_samples: {best_params['num_samples']}\n"
        f" - test_size: {best_params['test_size']}\n"
        f" - num_hidden: {best_params['num_hidden']}\n"
        f" - noise_factor: {best_params['noise_factor']}\n"
        f" - learning_rate: {best_params['learning_rate']}\n"
        f" - epochs: {best_params['epochs']}\n", file=output_file)

    print("Best accuracy:", best_accuracy, file=output_file)
    
    print(f"Grid search took {time_taken:.2f} seconds", file=output_file)