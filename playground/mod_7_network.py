
# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


# Create our neural network class
class ModSevenNetwork:

    # Initialize the neural network with random weights, learning rate, and number of training cycles
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int):
        self.weights_input_hidden = np.random.randn(input_nodes, hidden_nodes)
        self.weights_hidden_output = np.random.randn(hidden_nodes, output_nodes)
        self.learning_rate = 0.1
        self.num_cycles = 30

    # Define the sigmoid activation function
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    # Define the derivative of the sigmoid function
    def sigmoid_derivative(x: float) -> float:
        return ModSevenNetwork.sigmoid(x) * (1 - ModSevenNetwork.sigmoid(x))

    # Train the neural network using the training data
    def train(self, training_data: List[Dict[str, float]]):
        for _ in range(self.num_cycles):
            for sample in training_data:
                LAC, SOW, TACA = sample["LAC"], sample["SOW"], sample["TACA"]
                inputs = np.array([[LAC, SOW]]).T
                targets = np.array([[TACA]]).T

                # Calculate the output of the hidden and output layers
                hidden_layer_output = ModSevenNetwork.sigmoid(np.dot(self.weights_input_hidden.T, inputs))
                output_layer_output = ModSevenNetwork.sigmoid(np.dot(self.weights_hidden_output.T, hidden_layer_output))

                # Calculate the error for the output and hidden layers
                output_error = targets - output_layer_output
                hidden_error = np.dot(self.weights_hidden_output, output_error)

                # Calculate the deltas for the output and hidden layers
                delta_output = output_error * ModSevenNetwork.sigmoid_derivative(np.dot(self.weights_hidden_output.T, hidden_layer_output))
                delta_hidden = hidden_error * ModSevenNetwork.sigmoid_derivative(np.dot(self.weights_input_hidden.T, inputs))

                # Update the weights of the output and hidden layers
                self.weights_hidden_output += self.learning_rate * np.dot(hidden_layer_output, delta_output.T)
                self.weights_input_hidden += self.learning_rate * np.dot(inputs, delta_hidden.T)

    # Predict the output given inputs using the trained neural network
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        hidden_layer_output = ModSevenNetwork.sigmoid(np.dot(self.weights_input_hidden.T, inputs))
        output_layer_output = ModSevenNetwork.sigmoid(np.dot(self.weights_hidden_output.T, hidden_layer_output))
        return output_layer_output

    # Find the best threshold to minimize false positives and false negatives
    def find_best_threshold(self, training_data: List[Dict[str, float]]) -> float:
        best_threshold = 0.0
        best_accuracy = 0.0
        thresholds = np.arange(0.0, 1.0, 0.01)

        for threshold in thresholds:
            correct_predictions = 0
            for sample in training_data:
                inputs = np.array([[sample["LAC"], sample["SOW"]]]).T
                predicted_TACA = int(self.predict(inputs) >= threshold)
                if predicted_TACA == sample["TACA"]:
                    correct_predictions += 1

            accuracy = correct_predictions / len(training_data)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        return best_threshold

    # Evaluate the accuracy of the neural network using the testing data and the threshold
    def evaluate(self, testing_data: List[Dict[str, float]], threshold: float) -> float:
        correct_predictions = 0
        for sample in testing_data:
            inputs = np.array([[sample["LAC"], sample["SOW"]]]).T
            predicted_TACA = int(self.predict(inputs) >= threshold)
            if predicted_TACA == sample["TACA"]:
                correct_predictions += 1

        accuracy = correct_predictions / len(testing_data)
        return accuracy

    # Calculate the total squared error, referred to as Big E
    def calculate_big_e(self, training_data: List[Dict[str, float]]) -> float:
        big_e = 0
        for sample in training_data:
            inputs = np.array([[sample["LAC"], sample["SOW"]]]).T
            targets = np.array([[sample["TACA"]]]).T
            output_layer_output = self.predict(inputs)
            error = targets - output_layer_output
            squared_error = np.sum(np.square(error))
            big_e += squared_error

        return big_e

    # Calculate the true positive rate (tpr) and false positive rate (fpr) for various thresholds
    def calculate_roc_values(self, training_data: List[Dict[str, float]]) -> Tuple[List[float], List[float]]:
        fpr, tpr = [], []
        thresholds = np.arange(0.0, 1.0, 0.01)

        for threshold in thresholds:
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            for sample in training_data:
                inputs = np.array([[sample["LAC"], sample["SOW"]]]).T
                predicted_TACA = int(self.predict(inputs) >= threshold)

                if predicted_TACA == sample["TACA"]:
                    if predicted_TACA == 1:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if predicted_TACA == 1:
                        false_positives += 1
                    else:
                        false_negatives += 1

            true_positive_rate = true_positives / (true_positives + false_negatives)
            false_positive_rate = false_positives / (false_positives + true_negatives)
            tpr.append(true_positive_rate)
            fpr.append(false_positive_rate)

        return fpr, tpr

    # Plot the ROC curve using the calculated tpr and fpr values
    def plot_roc_curve(self, fpr: List[float], tpr: List[float]):
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

# Instantiate the neural network
nn = ModSevenNetwork(input_nodes=2, hidden_nodes=2, output_nodes=1)

# Prepare and split data into training and testing sets
data = [
    {"LAC": 0.90, "SOW": 0.87, "TACA": 1},
    {"LAC": 1.81, "SOW": 1.02, "TACA": 0},
    {"LAC": 1.31, "SOW": 0.75, "TACA": 1},
    {"LAC": 2.36, "SOW": 1.60, "TACA": 0},
    {"LAC": 2.48, "SOW": 1.14, "TACA": 0},
    {"LAC": 2.17, "SOW": 2.08, "TACA": 1},
    {"LAC": 0.41, "SOW": 1.87, "TACA": 0},
    {"LAC": 2.85, "SOW": 2.91, "TACA": 1},
    {"LAC": 2.45, "SOW": 0.52, "TACA": 0},
    {"LAC": 1.05, "SOW": 1.93, "TACA": 0},
    {"LAC": 2.54, "SOW": 2.97, "TACA": 1},
    {"LAC": 2.32, "SOW": 1.73, "TACA": 0},
    {"LAC": 0.07, "SOW": 0.09, "TACA": 1},
    {"LAC": 1.86, "SOW": 1.31, "TACA": 0},
    {"LAC": 1.32, "SOW": 1.96, "TACA": 0},
    {"LAC": 1.45, "SOW": 2.19, "TACA": 0},
    {"LAC": 0.94, "SOW": 0.34, "TACA": 1},
    {"LAC": 0.28, "SOW": 0.71, "TACA": 1},
    {"LAC": 1.75, "SOW": 2.21, "TACA": 0},
    {"LAC": 2.49, "SOW": 1.52, "TACA": 0},
]

print("\n")

training_data = data[::2]
testing_data = data[1::2]

# Train the neural network
nn.train(training_data)

# Determine the best threshold for the output node
best_threshold = nn.find_best_threshold(training_data)

# Calculate the Big E total error
big_e = nn.calculate_big_e(training_data)
print(f"Big E total error: {big_e}")

# Evaluate the neural network using the testing data
accuracy = nn.evaluate(testing_data, best_threshold)
print(f"Accuracy on testing data: {accuracy * 100:.2f}%")
print("\n")

# Print the desired outputs and network outputs for each input tuple from the testing data
print("Desired Output vs. Network Output")
for sample in testing_data:
    inputs = np.array([[sample["LAC"], sample["SOW"]]]).T
    predicted_TACA = int(nn.predict(inputs) >= best_threshold)
    print(f"Input: {sample['LAC']:.2f}, {sample['SOW']:.2f} | Desired Output: {sample['TACA']} | Network Output: {predicted_TACA}")

# Calculate ROC values
fpr, tpr = nn.calculate_roc_values(training_data)

# Plot ROC curve
nn.plot_roc_curve(fpr, tpr)

print("\n")