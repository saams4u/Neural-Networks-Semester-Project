
# Import helper functions
import time
import numpy as np

# Import plot functions
from plot import plot_exemplars, plot_performance

# Import data from exemplars
from utils import exemplars


def train_and_evaluate(rbm,
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
                    output_file):
    
    correct_reconstructions = []
    for gibbs_cycles in gibbs_cycle_list:
        print(f"Training with {gibbs_cycles} Gibbs cycles...", file=output_file)

        # Train the exemplars with their respective labels
        train_exemplars = [exemplar for exemplar, _ in train_data]
        rbm.train(train_exemplars, epochs=epochs, learning_rate=learning_rate, gibbs_steps=gibbs_cycles, validation_data=None)
        
        # Create noisy versions of the test exemplars
        noisy_test_exemplars = [rbm.add_noise(exemplar, noise_factor=noise_factor) for exemplar in test_exemplars]

        # Reconstruct the noisy test exemplars using the RBM
        reconstructed_test_exemplars = [rbm.reconstruct(noisy_exemplar, iters) for noisy_exemplar in noisy_test_exemplars]

        # Determine the correct reconstructions
        correct_reconstructions_count = 0
        for _, (reconstructed_exemplar, true_label) in enumerate(zip(reconstructed_test_exemplars, test_labels)):
            # Calculate the MSE between the reconstructed exemplar and all exemplars
            mse_values = [mean_squared_error((reconstructed_exemplar > 0.5).astype(int), (exemplar > 0.5).astype(int)) for exemplar, _ in train_data]

            # Find the index of the most similar exemplar (smallest MSE)
            most_similar_index = np.argmin(mse_values)

            # Check if the most similar exemplar has the same label as the true label
            if train_data[most_similar_index][1] == true_label:
                correct_reconstructions_count += 1

        # Compute the frequency of correct reconstructions
        frequency = correct_reconstructions_count / len(test_labels)
        print(f"Frequency of correct reconstructions with {gibbs_cycles} Gibbs cycles: {frequency}", file=output_file)

        correct_reconstructions.append(frequency)

        '''
        NOTE: COMMENT/UNCOMMENT one of the function calls below to either plot the exemplar pairs
              (in sequence) OR all the exemplars
        '''
        
        ## Call function to plot original, noisy, and reconstructed exemplars for all labels
        plot_exemplars(rbm,
                       iters,
                       num_samples,
                       test_size,
                       num_hidden,
                       noise_factor,
                       learning_rate,
                       epochs,
                       gibbs_cycles, 
                       labels, 
                       test_exemplars, 
                       test_labels,
                       noisy_test_exemplars, 
                       reconstructed_test_exemplars,
                       output_file)

    # Call function to plot RBM performance with respect to Gibbs sampling cycles
    plot_performance(gibbs_cycle_list, 
                     correct_reconstructions,
                     iters, 
                     epochs, 
                     num_hidden,
                     noise_factor, 
                     learning_rate, 
                     test_size, 
                     output_file)

    return correct_reconstructions