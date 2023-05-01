
# Import to calculate Euclidean distance for similarity metric
from scipy.spatial.distance import euclidean


def train_and_evaluate_gibbs_cycles(gibbs_cycle_list, test_data, test_noise_factor, epochs, learning_rate):
    correct_reconstructions = []
    for gibbs_cycles in gibbs_cycle_list:
        print(f"Training with {gibbs_cycles} Gibbs cycles...")
        # Train the exemplars with their respective labels
        train_exemplars = [exemplar for exemplar, _ in train_data]
        rbm.train(train_exemplars, epochs=epochs, learning_rate=learning_rate, gibbs_steps=gibbs_cycles, validation_data=test_data)
        
        # Create noisy versions of the test exemplars
        noisy_test_exemplars = [rbm.add_noise(exemplar, noise_factor=test_noise_factor) for exemplar in test_exemplars]

        # Reconstruct the noisy test exemplars using the RBM
        reconstructed_test_exemplars = [rbm.reconstruct(noisy_exemplar) for noisy_exemplar in noisy_test_exemplars]

        # Determine the correct reconstructions
        correct_reconstructions_count = 0
        for _, (reconstructed_exemplar, true_label) in enumerate(zip(reconstructed_test_exemplars, test_labels)):
            # Calculate the Euclidean distance between the reconstructed exemplar and all exemplars
            distances = [euclidean((reconstructed_exemplar > 0.5).astype(int), (exemplar > 0.5).astype(int)) for exemplar, _ in train_data]

            # Find the index of the most similar exemplar (smallest Euclidean distance)
            most_similar_index = np.argmin(distances)

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
        # plot_exemplars_for_two_labels(gibbs_cycles, 
        #                               test_labels, 
        #                               test_exemplars, 
        #                               noisy_test_exemplars, 
        #                               reconstructed_test_exemplars)

        ## Call function to plot original, noisy, and reconstructed exemplars for all labels
        plot_exemplars_for_all_labels(gibbs_cycles, 
                                      labels, 
                                      test_exemplars, 
                                      noisy_test_exemplars, 
                                      reconstructed_test_exemplars)

    # Brief delay (for debugging purposes)
    time.sleep(3)

    # Call function to plot RBM performance with respect to Gibbs sampling cycles
    plot_rbm_performance(gibbs_cycle_list, correct_reconstructions)

    return correct_reconstructions