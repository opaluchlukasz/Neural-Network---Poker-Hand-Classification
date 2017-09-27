# coding: utf-8

import numpy as np

def sigmoid(Z):
    """
    Computes sigmoid activation function

    Arguments:
    x -- a scalar or numpy array of any size

    Returns:
    A -- activation value A
    cache -- cache that contains Z
    """
    
    return 1/(1+ np.exp(-Z)), Z

def relu(Z):
    """
    Computes relu activation function

    Arguments:
    x -- a scalar or numpy array of any size

    Returns:
    A -- activation value A
    cache -- cache that contains Z
    """
    return np.maximum(0,Z), Z

def sigmoid_backward(dA, cache):
    """
    Computes the backward pass for layer of sigmoid units

    Arguments:
    dA -- numpy array of shape (N,D) representing gradients of output layer
    cache -- numpy array of shape (N,D) used for backpropagation

    Returns:
    dZ -- gradient of sigmoid activation function
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Arguments:
  dA -- numpy array of shape (N,D) representing gradients of output layer
  cache -- numpy array of shape (N,D) used for backpropagation

  Returns:
  dZ -- gradient of relu activation function
  """
  Z = cache
  dZ = np.array(dA, copy=True)
  dZ[Z <= 0] = 0
  return dZ

def initialize_parameters_deep(layer_dims):
    """
    Initializes parameters for deep L-layer neural network and stores them in dictionary 

    Arguments:
    layer_dims -- python array containing the dimensions of each layer in the network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[layer_index], layer_dims[layer_index-1])
                    bl -- bias vector of shape (layer_dims[layer_index], 1)
    """
    
    parameters = {}
    L = len(layer_dims)

    # He initialization is used for layers with ReLU activation function
    for layer_index in range(1, L - 1):
        parameters[f"W{layer_index}"] = np.random.randn(layer_dims[layer_index], layer_dims[layer_index - 1]) * np.sqrt(2./layer_dims[layer_index - 1])
        parameters[f"b{layer_index}"] = np.zeros((layer_dims[layer_index], 1)) 

    parameters[f"W{L - 1}"] = np.random.randn(layer_dims[L - 1], layer_dims[L - 2]) * 0.01
    parameters[f"b{L - 1}"] = np.zeros((layer_dims[L - 1], 1)) 
 
    return parameters

def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer/input data: (size of previous layer/number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- dictionary containing "A", "W" and "b" - stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A) + b
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation_function):
    """
    Implements the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer/input data: (size of previous layer/number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation_function -- the activation function - can be either relu or sigmoid

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- dictionary containing "linear_cache" and "activation_cache" - stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation_function(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implements forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation (LINEAR->RELU combination to be used on all but last layer, followed by LINEAR->SIGMOID on the last layer)
    
    Arguments:
    X -- data, size: (input size, number of examples)
    parameters -- parameters initialized via initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them)
                the cache of linear_sigmoid_forward() (last one)
    """

    caches = []
    A = X
    L = len(parameters) // 2
    
    # Implement [LINEAR -> RELU]*(L-1)
    for layer_index in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters[f"W{layer_index}"], parameters[f"b{layer_index}"], relu)
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID (output layer)
    AL, cache = linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], sigmoid)
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to label predictions, size:(1, number of examples)
    Y -- true "label" vector, size:(1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = - (1 / m) * np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))) 
    cost = np.squeeze(cost)
    
    return cost

def linear_backward(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * dZ.dot(cache[0].T)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = cache[1].T.dot(dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation_backward_function):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation_backward_function -- the activation_backward function to be used in this layer, can be either sigmoid_backward or relu_backward
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    dZ = activation_backward_function(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implements the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (0...L-2 caches)
                the cache of linear_activation_forward() with "sigmoid" (L - 1 cache)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA{layer_index}"] = ... 
             grads["dW{layer_index}"] = ...
             grads["db{layer_index}"] = ... 
    """
    grads = {}
    L = len(caches) 
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L - 1]
    grads[f"dA{L}"], grads[f"dW{L}"], grads[f"db{L}"] = linear_activation_backward(dAL, current_cache, sigmoid_backward)

    # Previous layers (RELU -> LINEAR) gradients
    for layer_index in reversed(range(L-1)):
        current_cache = caches[layer_index]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f"dA{layer_index + 2}"], current_cache, relu_backward)
        grads[f"dA{layer_index + 1}"] = dA_prev_temp
        grads[f"dW{layer_index + 1}"] = dW_temp
        grads[f"db{layer_index + 1}"] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients (output of L_model_backward)
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W{layer_index}"] = ... 
                  parameters["b{layer_index}"] = ...
    """
    
    L = len(parameters) // 2

    for layer_index in range(1, L + 1):
        parameters[f"W{layer_index}"] -= learning_rate * grads[f"dW{layer_index}"]
        parameters[f"b{layer_index}"] -= learning_rate * grads[f"db{layer_index}"]
    return parameters
