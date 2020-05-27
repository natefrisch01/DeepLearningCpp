//=======================================================================
// Copyright (c) 2017 Adrian Schneider
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "utils.h"

std::vector<output> process_labels(std::vector<uint8_t> &labels) {

    // convert uint8_t entries to onehot Eigen vectors

    std::vector<Eigen::VectorXd> ret(labels.size());

    for (int i = 0; i < labels.size(); i++) {
        ret[i] = Eigen::VectorXd::Zero(10);
        int ind = labels[i];
        ret[i][ind] = 1;
    }
    return ret;

}

std::vector<input> process_images(std::vector<std::vector<double>> &images) {
    std::vector<Eigen::VectorXd> ret(images.size());
    for (int i = 0; i < images.size(); i++) {
        ret[i] = Eigen::Map<Eigen::Matrix<double, 784, 1> >(images[i].data());
        ret[i] *= (1.0/255.0);
    }

    return ret;
}


std::vector<std::pair<input, output>> make_samples(std::vector<input> &in, std::vector<output> &out) {
    assert(in.size() == out.size());
    std::vector<std::pair<input, output>> samples(in.size());
    for (int i = 0; i < in.size(); i++) {
        std::pair<input, output> s;
        s.first = in[i];
        s.second = out[i];
        samples[i] = s;
    }
    return samples;
}

BetterDataset process_dataset(mnist::MNIST_dataset<std::vector, std::vector<double>, uint8_t> &dataset) {
    BetterDataset ret = BetterDataset();
    ret.training_images = process_images(dataset.training_images);
    ret.training_labels = process_labels(dataset.training_labels);
    ret.test_images = process_images(dataset.test_images);
    ret.test_labels = process_labels(dataset.test_labels);
    ret.training_samples = make_samples(ret.training_images, ret.training_labels);
    ret.test_samples = make_samples(ret.test_images, ret.test_labels);
    return ret;
}
