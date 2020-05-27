#include <iostream>
#include <Eigen/Core>
#include "mnist/mnist_reader.hpp"

typedef Eigen::VectorXd input;
typedef Eigen::VectorXd output;

struct BetterDataset {
    std::vector<input> training_images;
    std::vector<output> training_labels;
    std::vector<input> test_images;
    std::vector<output> test_labels;
    std::vector<std::pair<input, output>> training_samples;
    std::vector<std::pair<input, output>> test_samples;

};

std::vector<input> process_images(std::vector<std::vector<double>> &images);
std::vector<output> process_labels(std::vector<uint8_t> &labels);
BetterDataset process_dataset(mnist::MNIST_dataset<std::vector, std::vector<double>, uint8_t> &dataset);
void print_data(input);
