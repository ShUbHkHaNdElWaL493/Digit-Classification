/*
    Shubh Khandelwal
*/

#ifndef LAYERS_H
#define LAYERS_H

#include <neuron.h>

class convolution_layer_2D
{

    private:

    std::vector<int> dimensions;
    std::vector<std::vector<double>> input;
    std::vector<neuron *> filters;

    public:

    convolution_layer_2D(int, std::vector<int>);

    std::vector<std::vector<std::vector<double>>> feed_forward(std::vector<std::vector<double>>);

    void back_propagate(std::vector<std::vector<std::vector<double>>>, double);

};

class pooling_layer_2D
{

    private:

    std::vector<int> dimensions;
    std::vector<std::vector<std::vector<double>>> input;

    public:

    pooling_layer_2D(std::vector<int>);

    std::vector<std::vector<std::vector<double>>> feed_forward(std::vector<std::vector<std::vector<double>>>);

    std::vector<std::vector<std::vector<double>>> back_propagate(std::vector<std::vector<std::vector<double>>>);

};

class softmax_layer
{

    private:

    std::vector<double> flattened_input;
    std::vector<double> x;
    std::vector<int> dimensions;
    std::vector<neuron *> nodes;

    public:

    softmax_layer(int, std::vector<int>);

    std::vector<double> feed_forward(std::vector<std::vector<std::vector<double>>>);

    std::vector<std::vector<std::vector<double>>> back_propagate(std::vector<double>, double);

};

#endif