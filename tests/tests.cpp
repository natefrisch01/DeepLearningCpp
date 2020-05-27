#include "network.h"
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include <string>
#include "utils.h"

int main(int argc, char* argv[]) {

    std::cout << "Loading Data: ";

    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<double>, uint8_t> dataset_original =
        mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION);

    BetterDataset dataset = process_dataset(dataset_original);

    std::cout << "    Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "    Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "    Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "    Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    std::vector<int> sizes = {784, 20, 20, 10};
    Network net = Network(sizes);
    /*
    std::vector<int> sizes = {2, 3, 4};

    Network net = Network(sizes);

    Eigen::VectorXd x(2);
    x << 0, 1;
    Eigen::VectorXd y(4);
    y << 0, 1, 2, 3;
    for (int i = 0; i < 10; i++)
        std::cout << Eigen::MatrixXd::Random(i, i) << std::endl;

    sample s;
    std::cout << "here1" << std::endl;
    s.first = x;
    s.second = y;
    std::vector<sample> batch;
    std::cout << "here" << std::endl;
    batch.push_back(s);
    batch.push_back(s);
    //std::cout << dataset.training_images[0];
    net.update_mini_batch(batch, 10);
    */



    net.SGD(dataset.training_samples, 5, 10, 3.0, dataset.test_samples);

    return 0;
}
