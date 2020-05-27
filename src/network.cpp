//=======================================================================
// Copyright (c) 2020 Nathanael Frisch
//=======================================================================
// ^ in case my work becomes famous

#include "network.h"

Network::Network(std::vector<int> &sizes) {

    //assign member variables
    this->num_layers = sizes.size();
    this->sizes = sizes;

    //this guy somehow makes ::Random() actually random >u<. Goes btw. 0 and -1
    std::srand((unsigned int) time(0));

    for (int i = 1; i < num_layers; i++) {
        //random bias vector for each layer after the input layer
        //size of vector corresponds to number of neurons, because each neuron
        //has 1 bias.
        Eigen::VectorXd layer_i_biases = Eigen::VectorXd::Random(sizes[i]);
        biases.push_back(layer_i_biases);

        //random weight matrix for each layer. size of matrix is (rows, cols) ->
        //(layer i-1.size, layer i size), because each neuron has a weight for
        //each of the next layers nodes.
        Eigen::MatrixXd layer_prev_weights = Eigen::MatrixXd::Random(sizes[i], sizes[i-1]);
        weights.push_back(layer_prev_weights);
    }
}

Eigen::VectorXd Network::feedForward(Eigen::VectorXd &a) {
    //for each layer before the last one, dot product of weights to the next
    //layer with the activation at the current layer, + the biases to the next
    //layer
    for (int i = 0; i < num_layers-1; i++) {
        a = sigmoid((weights[i] * a) + biases[i]);
    }
    return a;
}

//element-wise squishification (forces to be btw -1 and 1)
Eigen::VectorXd Network::sigmoid(Eigen::VectorXd z) {
    for(int i = 0; i < z.size(); i++) {
        z[i] = 1.0 / (1 + std::exp(z[i]*-1));
    }
    return z;
}

//takes vector, and performs squishy derivative.
Eigen::VectorXd Network::sigmoid_prime(Eigen::VectorXd &z) {
    return (Eigen::VectorXd::Ones(z.size())-sigmoid(z)).cwiseProduct(sigmoid(z));
}

//derivative of cost function, output activation - desired out
Eigen::VectorXd Network::cost_derivative(Eigen::VectorXd &output_activations, Eigen::VectorXd &desired_out) {
    return output_activations - desired_out;
}

std::pair<bias_list, weight_list> Network::backprop(Eigen::VectorXd &in, Eigen::VectorXd &desired_out) {


    //create empty lists for biases and weights to store the gradient of the cost function
    bias_list nabla_b; weight_list nabla_w;
    for(Eigen::VectorXd v : biases) {
        nabla_b.push_back(Eigen::VectorXd::Zero(v.size()));
    }
    for(Eigen::MatrixXd m : weights) {
        nabla_w.push_back(Eigen::MatrixXd::Zero(m.rows(), m.cols()));
    }

    // vector to store all the activations after sigmoid
    std::vector<Eigen::VectorXd> activation_list;
    activation_list.push_back(in);

    // vector to store all the activations before sigmoid
    std::vector<Eigen::VectorXd> z_list;

    // current activation
    Eigen::VectorXd activation = in;

    //feed forward
    for (int i = 0; i < num_layers-1; i++) {
        //activation before sigmoid
        Eigen::VectorXd z = (weights[i] * activation) + biases[i];
        z_list.push_back(z);
        //activation after sigmoid
        activation = sigmoid(z);
        activation_list.push_back(activation);
    }

    //backprop

    //start with last layer
    //compute error and multiply by the derivative of the activation in the last
    //layer.
    Eigen::VectorXd delta = cost_derivative(activation_list.back(),
        desired_out).cwiseProduct(sigmoid_prime(z_list.back()));

    //simply shift the biases by delta
    nabla_b.back() = delta;
    
    nabla_w.back() = colvec_dot_rowvec(delta, activation_list[activation_list.size()-2].transpose());

    for (int i = 2; i < num_layers; i++) {
        Eigen::VectorXd z = z_list[z_list.size()-i];
        Eigen::VectorXd sp = sigmoid_prime(z);
        Eigen::VectorXd delta_new;
        delta_new = (weights[weights.size()-i+1].transpose() * delta).cwiseProduct(sp);
        nabla_b[nabla_b.size()-i] = delta_new;
        nabla_w[nabla_w.size() - i] = colvec_dot_rowvec(delta_new, activation_list[activation_list.size()-i-1].transpose());
        delta.resize(delta_new.size());
        delta = delta_new;
    }



    std::pair<bias_list, weight_list> p;
    p.first = nabla_b;
    p.second = nabla_w;
    return p;
}


Eigen::MatrixXd Network::colvec_dot_rowvec(Eigen::VectorXd col, Eigen::VectorXd row) {
    //std::cout << col << std::endl << std::endl << row << std::endl;
    Eigen::MatrixXd ret(col.size(), row.size());
    for(int i = 0; i < col.size(); i++) {
      ret.row(i) = row * col[i];
    }
    //std::cout << ret << std::endl;
    return ret;
}

void Network::update_mini_batch(std::vector<sample> &mini_batch, int eta) {

    //create empty lists for biases and weights to store the gradient of the cost function
    bias_list nabla_b; weight_list nabla_w;
    for(Eigen::VectorXd v : biases) {
        nabla_b.push_back(Eigen::VectorXd::Zero(v.size()));
    }
    for(Eigen::MatrixXd m : weights) {
        nabla_w.push_back(Eigen::MatrixXd::Zero(m.rows(), m.cols()));
        //std::cout << m.size() << std::endl;
    }

    for (sample s : mini_batch) {
        std::pair<bias_list, weight_list> nablas_new = backprop(s.first, s.second);
        for (int i = 0; i < num_layers - 1; i++) {
            nabla_b[i] += nablas_new.first[i];
            nabla_w[i] += nablas_new.second[i];
        }
    }

    for (int i = 0; i < num_layers - 1; i++) {
        weights[i] = weights[i] - (eta/mini_batch.size()) * nabla_w[i];
        biases[i] = biases[i] - (eta/biases.size()) * nabla_b[i];
    }
}

void Network::SGD(std::vector<sample> &training_data, int epochs, int mini_batch_size, int eta, std::vector<sample> &test_data) {

    for (int i = 0; i < epochs; i++) {
        std::cout << "Epoch: " << i << std::endl;
        std::random_shuffle(training_data.begin(), training_data.end());
        int mini_start = 0, mini_end = mini_batch_size;
        std::printf("    %06d/%06lu", mini_start, training_data.size());
        while (mini_end < training_data.size()) {
            std::vector<sample> mini(training_data.begin() + mini_start,
            training_data.begin() + mini_end);
            update_mini_batch(mini, mini_batch_size);
            mini_start = mini_end;
            mini_end += mini_batch_size;
            std::printf("\r    %06d/%06lu", mini_start, training_data.size());
            if (mini_end % 10000 == 0) {
                std::cout << " Current accuracy: " << evaluate(test_data);
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


double Network::evaluate(std::vector<sample> test_data) {
    int sum = 0;
    for (int i = 0; i < test_data.size(); i++) {
        Eigen::VectorXd fed = feedForward(test_data[i].first);
        assert(fed.size() == test_data[i].second.size());
        Eigen::VectorXd::Index ind1, ind2;
        fed.maxCoeff(&ind1); test_data[i].second.maxCoeff(&ind2);
        if (ind1 == ind2)
            sum += 1;
    }
    return ((double) sum) / test_data.size();
}
