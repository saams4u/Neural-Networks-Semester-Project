
# Import helper functions
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_for_two_labels(rbm, 
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
                        output_file=None): 

    # Change indices to a discrete sequence pair (i.e., 0 and 1, 2 and 3, etc.)
    index_1 = 0
    index_2 = 1
    unique_labels = np.unique(labels)[index_1:index_2+1]
    for i, label in enumerate(unique_labels):
        label_indices = [index for index, lbl in enumerate(test_labels) if lbl == label]
        index = label_indices[0]

        plt.subplot(2, 3, i * 3 + 1)
        rbm.plot_digit(test_exemplars[index])
        plt.title(f"Original {label}")

        plt.subplot(2, 3, i * 3 + 2)
        rbm.plot_digit(noisy_test_exemplars[index])
        plt.title(f"Noisy {label}")

        plt.subplot(2, 3, i * 3 + 3)
        rbm.plot_digit(reconstructed_test_exemplars[index])
        plt.title(f"Reconstructed {label}")

    plt.tight_layout()

    ## Save the plot as a PNG file with the accompanying metadata
    metadata = f"_num_samples-{num_samples}_test_size-{test_size}_num_hidden-{num_hidden}_noise_factor-{noise_factor}_learning_rate-{learning_rate}_epochs-{epochs}"
    filename = f"exemplars_gibbs_cycles-{gibbs_cycles}{metadata}.png"
    save_directory = f"rbm_plots/{index_1}_and_{index_2}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if output_file:
        plt.savefig(os.path.join(save_directory, filename), dpi=300)
    else:
        plt.show()

def plot_for_all_labels(rbm, 
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

    # plt.tight_layout()

    # Adjust the spacing between the subplots
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

    ## Save the plot as a PNG file with the accompanying metadata
    metadata = f"_num_samples-{num_samples}_test_size-{test_size}_num_hidden-{num_hidden}_noise_factor-{noise_factor}_learning_rate-{learning_rate}_epochs-{epochs}"
    filename = f"exemplars_gibbs_cycles-{gibbs_cycles}{metadata}.png"
    save_directory = "rbm_plots/all_numerals"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(os.path.join(save_directory, filename), dpi=300)

    if output_file:
        plt.savefig(os.path.join(save_directory, filename), dpi=300)
    else:
        plt.show()
        
def plot_rbm_performance(gibbs_cycles, correct_reconstructions, index_1, index_2, output_file=None):
    # Plot relationship between number of Gibbs cycles and frequency of correct reconstructions
    plt.plot(gibbs_cycles, correct_reconstructions, marker='o')
    plt.xlabel('Number of Gibbs sampling cycles')
    plt.ylabel('Frequency of correct reconstructions')
    plt.title('Measuring the Performance of the RBM')
    plt.grid()

    ## Save the plot as a PNG file
    filename = "rbm_performance.png"
    save_directory = f"rbm_plots/{index_1}_and_{index_2}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if output_file:
        plt.savefig(os.path.join(save_directory, filename), dpi=300)
    else:
        plt.show()