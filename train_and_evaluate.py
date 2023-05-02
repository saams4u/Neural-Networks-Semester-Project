
# Import helper functions
import time
import numpy as np

# Import plot functions
from plot_exemplars import plot_for_two_labels, plot_for_all_labels, plot_rbm_performance


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def train_and_evaluate_gibbs_cycles(rbm,
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
                                    output_file=None):
    
    correct_reconstructions = []
    for gibbs_cycles in gibbs_cycle_list:
        print(f"Training with {gibbs_cycles} Gibbs cycles...")
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
            # Calculate the Euclidean distance between the reconstructed exemplar and all exemplars
            mse_values = [mean_squared_error((reconstructed_exemplar > 0.5).astype(int), (exemplar > 0.5).astype(int)) for exemplar, _ in train_data]

            # Find the index of the most similar exemplar (smallest MSE)
            most_similar_index = np.argmin(mse_values)

            # Check if the most similar exemplar has the same label as the true label
            if train_data[most_similar_index][1] == true_label:
                correct_reconstructions_count += 1

        # Compute the frequency of correct reconstructions
        frequency = correct_reconstructions_count / len(test_labels)
        print(f"Frequency of correct reconstructions with {gibbs_cycles} Gibbs cycles: {frequency}")

        correct_reconstructions.append(frequency)

        '''
        NOTE: COMMENT/UNCOMMENT one of the function calls below to either plot the exemplar pairs
              (in sequence) OR all the exemplars
        '''
        
        ## Call function to plot original, noisy, and reconstructed exemplars for two labels at a time 
        # plot_for_two_labels(rbm,
        #                     num_samples,
        #                     test_size,
        #                     num_hidden,
        #                     noise_factor,
        #                     learning_rate,
        #                     epochs,
        #                     gibbs_cycles, 
        #                     labels,
        #                     test_exemplars, 
        #                     test_labels,
        #                     noisy_test_exemplars, 
        #                     reconstructed_test_exemplars,
        #                     output_file)

        ## Call function to plot original, noisy, and reconstructed exemplars for all labels
        plot_for_all_labels(rbm,
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
                            output_file=None)

    # Call function to plot RBM performance with respect to Gibbs sampling cycles
    plot_rbm_performance(gibbs_cycle_list, correct_reconstructions, index_1=0, index_2=1, output_file=output_file)

    return correct_reconstructions