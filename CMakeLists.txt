cmake_minimum_required(VERSION 3.0)

PROJECT(perceptron_nn)
# .. -> hint, that the mnist package is one directory level above.
# When using just "find_package(MNIST REQUIRED)", "MNIST_DIR"
#    cmake variable has to be set correctly.
#
#####################
# Find the dependencies
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MNIST PATHS mnist/)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)
include_directories(${MNIST_INCLUDE_DIR})

#Doxygen
# first we can indicate the documentation build as an option and set it to ON by default
option(BUILD_DOC "Build documentation" ON)

# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)


#####################
# Set up the DLCPP executable
# Make sure the DLCPP headers can be found by the compiler
include_directories(${PROJECT_SOURCE_DIR}/include)
# Define which cpp files need to be compiled
set(DLCPP_SOURCES
    src/network.cpp
    src/utils.cpp
)
set(TEST_SOURCES
    tests/tests.cpp
)
# Define the DLCPP library
add_library(DLCPP ${DLCPP_SOURCES})
target_link_libraries(DLCPP Eigen3::Eigen ${MNIST_LIBRARIES})
# Define the tests executable
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests DLCPP Eigen3::Eigen ${MNIST_LIBRARIES})
# Pass MNIST data directory to tests.cpp
target_compile_definitions(tests PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
