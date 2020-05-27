#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <stdio.h>

/**
Stores a sample containing an input activation vector and an desired output
vector, .first and .second, respectively.
*/
typedef std::pair<Eigen::VectorXd, Eigen::VectorXd> sample;

/**
A std::vector of Eigen::VectorXd, making a vector of bias vectors.
*/
typedef std::vector<Eigen::VectorXd> bias_list;

/**
A std::vector of Eigen::MatrixXd, making a vector of weight vectors.
*/
typedef std::vector<Eigen::MatrixXd> weight_list;

/**
@brief Stores a eural network (nn) and implements methods for performing
stochastic gradient descent.

This class is a cpp adaptation of the first two chapters in MichealNielsen's
book on neural networks and deep learning, found at
http://neuralnetworksanddeeplearning.com/. It stores a neural network as a
vector of vectors for biases and a vector of matrices for weights.

@author Nathanael Frisch
@date May 2020
*/
class Network {
private:

    int num_layers;
    std::vector<int> sizes;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::MatrixXd> weights;


public:
    /**
    Initializes the neural network. Accepts a vector of
    sizes, the first index corresponding to the number of neurons in the input
    layer, the last index corresponding to the number of neurons in the output
    layer, and the middle values corresponding to hidden layers.

    @param sizes vector of layer sizes.
    */
    Network(std::vector<int> &sizes);

    /**
    Feeds forward a vector of activations from the first input layer to the
    output layer, returning a the vector of activations at the output layer.
    Input vector size must be equal to the number of neurons in the input layer.

    @param input vector of activations at input layer.
    @return output activations.
    */
    Eigen::VectorXd feedForward(Eigen::VectorXd &input);

    /**
    Performs stochastic gradient descent on the nn.

    @param training_data a std::vector of type sample i.e. a list of pairs of
    input activations and desired output activations. Used for training.
    @param epochs number of times to train on the entire dataset.
    @param eta learning rate.
    @param test_data a std::vector of type sample i.e. a list of pairs of
    input activations and desired output activations. Used for evaluate.
    */
    void SGD(std::vector<sample> &training_data, int epochs, int mini_batch_size, int eta, std::vector<sample> &test_data);

    /**
    Performs backprop on each sample from a mini batch of the data (a smaller
    subset of the full training data), then updates the weights and biases.

    @param mini_batch a std::vector of type sample i.e. a list of pairs of
    input activations and desired output activations.

    @param eta learning rate.
    */
    void update_mini_batch(std::vector<sample> &mini_batch, int eta);

    /**
    The heart of the nn's ability to learn. It performs gradient descent,
    starting with the output layer and propogates backwards.

    @param in input layer activation.
    @param desired_out desired output.
    @return a pair of a list of biases and a list of weights, the
    desired nudges to them from this one training example.
    */
    std::pair<bias_list, weight_list> backprop(Eigen::VectorXd &in, Eigen::VectorXd &desired_out);

    /**
    Feed forward a vector of samples. Return the percentage of samples that it
    classifies correctly.

    @param test_data vector of samples.
    @return percentage of samples correctly classified.
    */
    double evaluate(std::vector<sample> test_data);

    /**
    The derivative of the cost function. In this case simply simply the actual
    output activations - the desired output (elementwise).

    @param output_activations actual output activations.
    @param desired_out desired output activations.
    @return derivative of the cost function.
    */
    Eigen::VectorXd cost_derivative(Eigen::VectorXd &output_activations, Eigen::VectorXd &desired_out);

    /**
    Element-wise squishification btw. 0 and 1.

    @param vector to be squished.
    @return squished vector.
    */
    Eigen::VectorXd sigmoid(Eigen::VectorXd);

    /**
    Derivative of squishification function.

    @param vector to be derived.
    @return derivative of the squish function.
    */
    Eigen::VectorXd sigmoid_prime(Eigen::VectorXd &);

    /**
    This is a weird one, but it's used in backprop. Takes a column vector and a
    row vector, multiplies each element in the column by each element in the row
    and, instead of summing, returns a matrix for each individual product.

    @param col column vector.
    @param row row vector.
    @return a matrix as described above.
    */
    Eigen::MatrixXd colvec_dot_rowvec(Eigen::VectorXd col, Eigen::VectorXd row);

}; //class Network
