/*
    Shubh Khandelwal
*/

#ifndef NEURON_H
#define NEURON_H

#include <vector>

class neuron
{

    public:

    std::vector<double> weights;
    double bias;

    neuron(int);

    double feed_forward(std::vector<double>);

    void back_propagate(double, std::vector<double>, double);

};

#endif