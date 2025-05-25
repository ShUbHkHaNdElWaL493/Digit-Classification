/*
    Shubh Khandelwal
*/

#include "neuron.h"
#include <random>

neuron::neuron(int input_size)
{

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < input_size; i++)
    {
        this->weights.push_back(dist(gen) / input_size);
    }

}

double neuron::feed_forward(std::vector<double> input)
{

    double output = 0;
    for (int i = 0; i < input.size(); i++)
    {
        output += input[i] * this->weights[i];
    }
    return output;

}

void neuron::back_propagate(std::vector<double> dL_dw, double learn_rate)
{

    for (int i = 0; i < this->weights.size(); i++)
    {
        this->weights[i] -= learn_rate * dL_dw[i];
    }

}