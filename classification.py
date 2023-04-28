
import random
import numpy as np

import matplotlib.pyplot as plt


# Define the Sigmoid function and its derivative
def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    
    return sigmoid(x) * (1 - sigmoid(x))


# Functions for feedforward and backpropagation for single perceptron
def feedforward_single(inputs, weights, bias):

    output = sigmoid(np.dot(weights, inputs) + bias)
    return output


def backpropagation_single(input_pair, output, desired_output, weights, bias, learning_rate):

    output_error = desired_output - output
    output_delta = output_error * sigmoid_derivative(output)

    for i in range(len(weights)):
        weights[i] += learning_rate * output_delta * input_pair[i]
        bias += learning_rate * output_delta

    return weights, bias


# Train and evaluate the single perceptron
def train_single_perceptron(inputs, desired_outputs, epochs, learning_rate=0.001, seed=None):

    if seed is not None:
        random.seed(seed)

    weights = [random.uniform(-1, 1) for _ in range(len(inputs[0]))]
    bias = random.uniform(-1, 1)

    for epoch in range(epochs):
        for i in range(0, len(inputs), 2):  # Iterate through odd-numbered input/output pairs
            input_pair = inputs[i]
            desired_output = desired_outputs[i]

            output = feedforward_single(input_pair,
                                        weights,
                                        bias)

            weights, bias = backpropagation_single(input_pair,
                                                   output,
                                                   desired_output,
                                                   weights,
                                                   bias,
                                                   learning_rate)

    return weights, bias


# Train and evaluate the single perceptron with different initialized random weights
def train_and_evaluate_single_perceptron(n_trials, training_inputs, training_outputs, testing_inputs,
                                         testing_outputs,
                                         epochs,
                                         learning_rate=0.001):

    best_error = float('inf')
    best_weights = None
    best_bias = None

    for trial in range(n_trials):
        seed = random.randint(0, 2**32 - 1)
        weights, bias = train_single_perceptron(training_inputs,
                                                training_outputs,
                                                epochs,
                                                learning_rate,
                                                seed)

        error = total_error_single(testing_inputs,
                                   testing_outputs,
                                   weights,
                                   None,
                                   bias,
                                   None,
                                   0)  # Set n_hidden_nodes to 0 for single perceptron

        if error < best_error:
            best_error = error
            best_weights = weights
            best_bias = bias

    return best_weights, best_bias, best_error


# Calculate total error for single perceptron
def total_error_single(inputs, desired_outputs, weights, weights_ho, biases_h, biases_o, n_hidden_nodes):

    big_e = 0
    for i in range(len(inputs)):
        input_i = inputs[i]
        desired_output = desired_outputs[i]

        if n_hidden_nodes == 0:  # Single perceptron
            output = feedforward_single(input_i,
                                        weights,
                                        biases_h)
        else:  # Two-layer neural network
            _, output = feedforward(input_i,
                                    weights,
                                    weights_ho,
                                    biases_h,
                                    biases_o,
                                    n_hidden_nodes)

        big_e += calculate_error(output, desired_output)
    return big_e


# Functions for feedforward and backpropagation
def feedforward(inputs, weights_ih, weights_ho, biases_h, biases_o, n_hidden_nodes):

    hidden_layer = [sigmoid(sum(weights_ih[j][i] * inputs[j] + biases_h[i] for j in range(len(inputs))))
                    for i in range(n_hidden_nodes)]

    output = sigmoid(sum(weights_ho[i][0] * hidden_layer[i] + biases_o for i in range(n_hidden_nodes)))
    return hidden_layer, output


def backpropagation(input_pair, hidden_layer, output, desired_output, weights_ih, weights_ho, biases_h, biases_o,
                    learning_rate,
                    n_hidden_nodes):

    output_error = desired_output - output
    output_delta = output_error * sigmoid_derivative(output)
    hidden_errors = [weights_ho[i][0] * output_delta for i in range(n_hidden_nodes)]

    hidden_deltas = [hidden_errors[i] * sigmoid_derivative(hidden_layer[i]) for i in range(n_hidden_nodes)]

    for j in range(len(weights_ho)):
        weights_ho[j][0] += learning_rate * output_delta * hidden_layer[j]
        biases_o += learning_rate * output_delta

    for j in range(len(weights_ih)):
        for k in range(len(weights_ih[j])):
            weights_ih[j][k] += learning_rate * hidden_deltas[k] * input_pair[j]

    for k in range(len(biases_h)):
        biases_h[k] += learning_rate * hidden_deltas[k]

    return weights_ih, weights_ho, biases_h, biases_o


# Online training using backpropagation
def train_online(inputs, desired_outputs, epochs, n_hidden_nodes, learning_rate=0.001, seed=None):

    if seed is not None:
        random.seed(seed)

    weights_ih = [[random.uniform(-1, 1) for _ in range(n_hidden_nodes)] for _ in range(len(inputs[0]))]
    weights_ho = [[random.uniform(-1, 1)] for _ in range(n_hidden_nodes)]

    biases_h = [random.uniform(-1, 1) for _ in range(n_hidden_nodes)]
    biases_o = random.uniform(-1, 1)

    for epoch in range(epochs):
        for i in range(0, len(inputs), 2):  # Iterate through odd-numbered input/output pairs
            input_pair = inputs[i]
            desired_output = desired_outputs[i]

            hidden_layer, output = feedforward(input_pair,
                                               weights_ih,
                                               weights_ho,
                                               biases_h,
                                               biases_o,
                                               n_hidden_nodes)

            weights_ih, weights_ho, biases_h, biases_o = backpropagation(input_pair,
                                                                          hidden_layer,
                                                                          output,
                                                                          desired_output,
                                                                          weights_ih,
                                                                          weights_ho,
                                                                          biases_h,
                                                                          biases_o,
                                                                          learning_rate,
                                                                          n_hidden_nodes)

    return weights_ih, weights_ho, biases_h, biases_o


# Train and evaluate the two-layer neural network with different initialized random weights
def train_and_evaluate(n_trials, training_inputs, training_outputs, testing_inputs, testing_outputs, epochs,
                       n_hidden_nodes,
                       learning_rate=0.1):
                       
    best_error = float('inf')
    best_weights_ih = None
    best_weights_ho = None
    best_biases_h = None
    best_biases_o = None

    for trial in range(n_trials):
        seed = random.randint(0, 2**32 - 1)
        weights_ih, weights_ho, biases_h, biases_o = train_online(training_inputs,
                                                                   training_outputs,
                                                                   epochs,
                                                                   n_hidden_nodes,
                                                                   learning_rate,
                                                                   seed)

        error = total_error(testing_inputs,
                            testing_outputs,
                            weights_ih,
                            weights_ho,
                            biases_h,
                            biases_o,
                            n_hidden_nodes)

        if error < best_error:
            best_error = error
            best_weights_ih = weights_ih
            best_weights_ho = weights_ho
            best_biases_h = biases_h
            best_biases_o = biases_o

    return best_weights_ih, best_weights_ho, best_biases_h, best_biases_o, best_error


# Calculate error
def calculate_error(output, desired_output):

    return 0.5 * (desired_output - output)**2


# Calculate total error
def total_error(inputs, desired_outputs, weights_ih, weights_ho, biases_h, biases_o, n_hidden_nodes):

    big_e = 0
    for i in range(len(inputs)):
        input_i = inputs[i]
        desired_output = desired_outputs[i]

        _, output = feedforward(input_i,
                                weights_ih,
                                weights_ho,
                                biases_h,
                                biases_o,
                                n_hidden_nodes)

        big_e += calculate_error(output, desired_output)
    return big_e


# Determine the best threshold for ROCs
def find_best_threshold(inputs, desired_outputs, weights_ih, weights_ho, biases_h, biases_o, n_hidden_nodes):

    best_threshold = 0.5
    best_correct = 0

    for threshold in [i * 0.01 for i in range(101)]:
        correct = 0
        for i in range(len(inputs)):
            input_i = inputs[i]
            desired_output = desired_outputs[i]

            _, output = feedforward(input_i, 
                                    weights_ih, 
                                    weights_ho, 
                                    biases_h, 
                                    biases_o,
                                    n_hidden_nodes)

            prediction = int(output >= threshold)
            if prediction == desired_output:
                correct += 1

        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold

    return best_threshold
    

# Calculate ROCs for given thresholds
def calculate_ROCs(inputs, desired_outputs, weights_ih, weights_ho, biases_h, biases_o, n_hidden_nodes, thresholds):

    TPR_list = []
    FPR_list = []

    for threshold in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(len(inputs)):
            input_i = inputs[i]
            desired_output = desired_outputs[i]

            if n_hidden_nodes == 0:  # Single perceptron
                output = feedforward_single(input_i, weights_ih, biases_h)
            else:  # Two-layer neural network
                _, output = feedforward(input_i, weights_ih, weights_ho, biases_h, biases_o, n_hidden_nodes)

            predicted_output = 1 if output >= threshold else 0

            if predicted_output == 1 and desired_output == 1:
                TP += 1
            elif predicted_output == 1 and desired_output == 0:
                FP += 1
            elif predicted_output == 0 and desired_output == 0:
                TN += 1
            elif predicted_output == 0 and desired_output == 1:
                FN += 1

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TPR_list.append(TPR)
        FPR_list.append(FPR)

    return TPR_list, FPR_list

# Provided tabular data
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

# Prepare the input data
inputs = [[row["LAC"], row["SOW"]] for row in data]
desired_outputs = [row["TACA"] for row in data]

# Split the data into training and testing sets
training_inputs = inputs[::2]
training_outputs = desired_outputs[::2]

testing_inputs = inputs[1::2]
testing_outputs = desired_outputs[1::2]

# Initialize trials, epochs, and learning rate
n_trials = 10
epochs = 30  # Set the number of epochs to 30
learning_rate = 0.001


###########################################
###      SINGLE PERCEPTRON NETWORK      ###
###########################################

print("\n")

# Set n_hidden_nodes to 0 for single perceptron
n_hidden_nodes_single = 0

# Train and evaluate the single perceptron
best_weights_single, best_bias_single, best_error_single = train_and_evaluate_single_perceptron(
    n_trials, training_inputs, training_outputs, testing_inputs, testing_outputs, epochs, learning_rate
)

# Find the best threshold for ROCs using the training data and the total error for the single perceptron
best_threshold_single = find_best_threshold(training_inputs,
                                            training_outputs,
                                            best_weights_single,
                                            None,
                                            best_bias_single,
                                            None,
                                            n_hidden_nodes_single)

print("Best threshold for single perceptron:", best_threshold_single)
print("Total (Big E) error for single perceptron:", str(best_error_single))

# Evaluate the single perceptron using even-numbered rows (testing data)
correct_single = 0
total_predictions_single = len(testing_inputs)

for i in range(len(testing_inputs)):
    input_i_single = testing_inputs[i]
    desired_output_single = testing_outputs[i]

    output_single = feedforward_single(input_i_single,
                                       best_weights_single,
                                       best_bias_single)

    prediction_single = int(output_single >= best_threshold_single)
    if prediction_single == desired_output_single:
        correct_single += 1

    print(f"Input: {input_i_single}, Desired output: {desired_output_single}, Single Perceptron output: {round(output_single)}")

accuracy_single = correct_single / total_predictions_single

print(f"Accuracy for single perceptron network: {accuracy_single}")

print("\n")


###########################################
###   NETWORK FROM MODULE 7 ASSIGNMENT  ###
###########################################

# Set n_hidden_nodes to 2 for our two-layer network
n_hidden_nodes = 2

best_weights_ih, best_weights_ho, best_biases_h, best_biases_o, best_error = train_and_evaluate(
    n_trials, training_inputs, training_outputs, testing_inputs, testing_outputs, 
    epochs, n_hidden_nodes, learning_rate)

# Find the best threshold for ROCs using the training data and the total error for the two-layer neural network
best_threshold = find_best_threshold(training_inputs, 
                                     training_outputs, 
                                     best_weights_ih, 
                                     best_weights_ho, 
                                     best_biases_h, 
                                     best_biases_o,
                                     n_hidden_nodes)

print("Best threshold:", best_threshold)
print("Total (Big E) error for two-layer network:", str(best_error))

# Evaluate the network using even-numbered rows (testing data)
correct = 0
total_predictions = len(testing_inputs)

for i in range(len(testing_inputs)):
    input_i = testing_inputs[i]
    desired_output = testing_outputs[i]

    _, output = feedforward(input_i, 
                            best_weights_ih, 
                            best_weights_ho, 
                            best_biases_h, 
                            best_biases_o, 
                            n_hidden_nodes)

    prediction = int(output >= best_threshold)
    if prediction == desired_output:
        correct += 1

    print(f"Input: {input_i}, Desired output: {desired_output}, Network output: {round(output)}") 
    
accuracy = correct / total_predictions
print(f"Accuracy for two-layer network: {accuracy}")

print("\n")


################################################################################
###   Determine Receiver Operating Characteristics (ROCs) of both networks   ###
################################################################################

# Calculate ROCs for both networks using testing data
thresholds = [i * 0.01 for i in range(101)]

TPR_single, FPR_single = calculate_ROCs(testing_inputs, 
                                        testing_outputs, 
                                        best_weights_single, 
                                        None, 
                                        best_bias_single, 
                                        None, 
                                        n_hidden_nodes_single, 
                                        thresholds)

TPR_network, FPR_network = calculate_ROCs(testing_inputs, 
                                          testing_outputs, 
                                          best_weights_ih, 
                                          best_weights_ho, 
                                          best_biases_h, 
                                          best_biases_o, 
                                          n_hidden_nodes, 
                                          thresholds)

# Plot ROC curve
plt.figure()
plt.plot(FPR_single, TPR_single, label='Single Perceptron')
plt.plot(FPR_network, TPR_network, label='Two-layer Neural Network')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()