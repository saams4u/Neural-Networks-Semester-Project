
# Import for plotting
import matplotlib.pyplot as plt


def plot_for_two_labels(gibbs_cycles, labels, test_exemplars, noisy_test_exemplars, reconstructed_test_exemplars):
    # Change indices below to a discrete sequence pair (i.e., 0 & 1, 2 & 3, etc.)
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
    # metadata = f"_num_samples-{num_samples}_test_size-{test_size}_num_hidden-{num_hidden}_learning_rate-{learning_rate}_epochs-{epochs}"
    # filename = f"exemplars_gibbs_cycles-{gibbs_cycles}{metadata}.png"
    # save_directory = "rbm_plots/noise_factor_0.5/6_and_7"
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)
    # plt.savefig(os.path.join(save_directory, filename), dpi=300)

    plt.show()

def plot_for_all_labels(gibbs_cycles, labels, test_exemplars, noisy_test_exemplars, reconstructed_test_exemplars):
    unique_labels = np.unique(labels)  # Include all unique labels (0 to 7)
    num_labels = len(unique_labels)

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

    ## Save the plot as a PNG file with the accompanying metadata
    # metadata = f"_num_samples-{num_samples}_test_size-{test_size}_num_hidden-{num_hidden}_learning_rate-{learning_rate}_epochs-{epochs}"
    # filename = f"exemplars_gibbs_cycles-{gibbs_cycles}{metadata}.png"
    # save_directory = "rbm_plots/noise_factor_0.5/all_numerals"
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)
    # plt.savefig(os.path.join(save_directory, filename), dpi=300)

    plt.show()

def plot_rbm_performance(gibbs_cycles, correct_reconstructions):
    # Plot relationship between number of Gibbs cycles and frequency of correct reconstructions
    plt.plot(gibbs_cycles, correct_reconstructions, marker='o')
    plt.xlabel('Number of Gibbs sampling cycles')
    plt.ylabel('Frequency of correct reconstructions')
    plt.title('Measuring the Performance of the RBM')
    plt.grid()

    ## Save the plot as a PNG file
    # filename = "rbm_performance.png"
    # save_directory = "rbm_plots/noise_factor_0.5/all_numerals"
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)
    # plt.savefig(os.path.join(save_directory, filename), dpi=300)

    plt.show()