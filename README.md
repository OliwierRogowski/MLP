# Multi-Layer Perceptron (MLP) in C++
A lightweight, efficient implementation of a Multi-Layer Perceptron (MLP) using C++ and the Eigen linear algebra library. 
This project demonstrates how a neural network with hidden layers can learn non-linear functions, specifically solving the classic XOR (Exclusive OR) problem.
## Project Architecture
Unlike a basic single-layer perceptron, this MLP consists of multiple layers of neurons, allowing it to model complex relationships between inputs and outputs.
### Neural Network Structure
     The project is built around two core classes:
       Layer: Represents a fully connected (dense) layer. It manages a weight matrix and a bias vector.
       MLP: A container for multiple Layer objects, managing the flow of data through the network.
### Mathematical Foundation
        #### Activation Function
       Uses the Sigmoid function to introduce non-linearity:
       $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
       #### Forward Propagation
       Calculates the dot product of inputs and weights, adds the bias, and applies the activation function:
       $$output = \sigma(X \cdot W + b)$$
       #### Backpropagation
       Implements the Gradient Descent algorithm. The network calculates the error at the output and propagates it backward to update weights using the derivative of the sigmoid function:
       $$\Delta w = \eta \cdot (error \cdot \sigma'(z))$$
## The XOR Problem
A single-layer perceptron cannot solve the XOR problem because the data is not linearly separable. This implementation solves it by using:
  Input Layer: 2 neurons
  Hidden Layer: 4 neurons (allowing for non-linear feature mapping)
  Output Layer: 1 neuron.
After 10,000 epochs of training with a learning rate ($\eta$) of 0.1, the network successfully predicts the XOR logic 
table:
0, 0 $\rightarrow$ 0
0, 1 $\rightarrow$ 1
1, 0 $\rightarrow$ 1
1, 1 $\rightarrow$ 0 

## Key Technical Features
### Eigen Library 
Used for optimized matrix and vector operations, significantly improving performance over manual loops.

### Dynamic Topology 
The add_layer method allows for easy creation of networks with any number of hidden layers and neurons.
### State Memory
Each layer stores its last input and output, which is essential for calculating gradients during the training phase.
