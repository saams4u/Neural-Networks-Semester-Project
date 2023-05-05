
# Import helper functions
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_exemplars(rbm, 
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
                   output_file): 
    
    unique_labels = np.unique(labels)  # Include all unique labels (0 to 7)
    num_labels = len(unique_labels)

    plt.figure(1, figsize=(10, 10))

    for i, label in enumerate(unique_labels):
        label_indices = [index for index, lbl in enumerate(test_labels) if lbl == label]
        index = label_indices[0]

        plt.subplot(num_labels, 3, i * 3 + 1)
        rbm.plot_digit(test_exemplars[index])
        plt.title(f"Original {label}")

        plt.subplot(num_labels, 3, i * 3 + 2)
        rbm.plot_digit(noisy_test_exemplars[index])
        plt.title(f"Noisy {label}")

        plt.subplot(num_labels, 3, i * 3 + 3)
        rbm.plot_digit(reconstructed_test_exemplars[index])
        plt.title(f"Reconstructed {label}")

    plt.tight_layout()

    # Adjust the spacing between the subplots
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)


    ## Save the plot as a PNG file with the accompanying metadata
    metadata = f"_recon_iters-{iters}_num_samples-{num_samples}_test_size-{test_size}_num_hidden-{num_hidden}_noise_factor-{noise_factor}_learning_rate-{learning_rate}_epochs-{epochs}"
    filename = f"exemplars=gibbs_cycles-{gibbs_cycles}{metadata}.png"
    save_directory = "plots"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if output_file:
        plt.savefig(os.path.join(save_directory, filename), dpi=300)
        plt.close()
    else:
        plt.show()
        
def plot_performance(gibbs_cycle_list,
                     correct_reconstructions,
                     iters, 
                     epochs, 
                     num_hidden,
                     noise_factor, 
                     learning_rate, 
                     test_size, 
                     output_file):

    # Plot relationship between number of Gibbs cycles and frequency of correct reconstructions
    plt.plot(gibbs_cycle_list, correct_reconstructions, marker='o')
    plt.xlabel('Number of Gibbs sampling cycles')
    plt.ylabel('Frequency of correct reconstructions')
    plt.title('Measuring the Performance of the RBM')
    plt.grid()

    ## Save the plot as a PNG file
    filename = f"rbm_performance=gibbs_cycles-{gibbs_cycle_list}_recon_iters-{iters}_epochs-{epochs}_num_hidden-{num_hidden}_noise_factor-{noise_factor}_learning_rate-{learning_rate}_test_size-{test_size}.png"
    save_directory = "plots"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if output_file:
        plt.savefig(os.path.join(save_directory, filename), dpi=300)
        plt.close()
    else:
        plt.show()