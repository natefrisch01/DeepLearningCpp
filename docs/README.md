# README #

git clone --recurse-submodules https://github.com/natefrisch01/DeepLearningCpp.git

# Overview #
Deep Learning C++ (DLCPP) is a small software package that implements a neural
network (nn) with methods for stochastic gradient descent. It is intended to be
used for educational purposes, not performance, and the code reflects this. This
code implements the nn described in the first two chapters of Michael Nielsen's
book, *Neural Networks and Deep Learning*.

# Dependencies #
Eigen3

# Installation #

The mnist/ folder is meant to hold the git repository that can be found here:
https://github.com/wichtounet/mnist

If you didn't use the --recurse-submodules tag when cloning, you can use the
command: `git submodule update --init --recursive`

To build:
`mkdir build`
`cd build`
`cmake ..`

To run executable (from build directory):
`./tests`

To build with doxygen:
`cmake -DCMAKE_BUILD_TYPE=Release ..`

# Contributions #
Please feel free to fork the code and fix any bugs, add updates, or implement
the more complicated nn's described later in the book.

# Access #
GitHub: https://github.com/natefrisch01/DeepLearningCpp
Documentation: https://natesnote.com/DeepLearningCpp

Copyright (c) 2020 Nathanael Frisch
