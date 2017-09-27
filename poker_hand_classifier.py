# coding: utf-8

import csv
import numpy as np
import matplotlib.pyplot as plt
from deep_neural_network import *

def run():
    with open('data/poker-lsn.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
        parsed = np.int_(data)

    X = parsed[:, :10].T
    # poker hand classification has 10 ranks (0..9), therefore we need to normalize it
    Y = parsed[:, 10:11].T / 10
    test_set = np.array([[1, 1, 1, 2, 1, 3, 1, 5, 4, 11], [1, 1, 1, 2, 1, 3, 1, 5, 4, 11], [1, 1, 1, 12, 2, 12, 4, 12, 2, 13], [1, 1, 1, 2, 2, 3, 2, 4, 2, 5], [1, 3, 1, 4, 1, 5, 1, 6, 1, 7], [2, 1, 2, 10, 2, 11, 2, 12, 2, 13] ]).T

    parameters = L_layer_model(X, Y, [X.shape[0], 25, 12, 6, 1])

    predictions = classifyPokerHands(test_set, parameters)
    print(f"0 rank categorised as: {predictions[0][1]}")
    print(f"0 rank categorised as: {predictions[0][1]}")
    print(f"3 rank categorised as: {predictions[0][2]}")
    print(f"4 rank categorised as: {predictions[0][3]}")
    print(f"8 rank categorised as: {predictions[0][4]}")
    print(f"9 rank categorised as: {predictions[0][5]}")

def L_layer_model(X, Y, layers_dims, learning_rate = 0.25, num_iterations = 1500):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, number_of_features)
    Y -- true "label" vector; size:(1, number_of_features)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    
    Returns:
    parameters -- parameters learnt by the model
    """
    costs = []
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations + 1):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if i % 100 == 0:
            print(f"Cost after {i} iteration: {cost}")
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tensrun())')
    plt.title(f"Cost function for learning rate = {learning_rate}")
    plt.show(block=False)
    
    return parameters

def classifyPokerHands(X, parameters):
    """
    Classifies each test case in X using the learned parameters
    
    Arguments:
    parameters -- dictionary containing learned parameters 
    X -- input data - size:(number_of_features, number_of_examples)
    
    Returns
    ranks -- vector of poker hand ranks
    """
    # Computes probabilities using forward propagation
    A2, cache = L_model_forward(X, parameters)
    # A2 values are in range (0..1) and poker hand classification has 10 class
    ranks = A2 * 10
    
    return ranks
