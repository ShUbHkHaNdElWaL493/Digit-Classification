/*
    Shubh Khandelwal
*/

#include <algorithm>
#include <cmath>
#include "layers.h"

convolution_layer_2D::convolution_layer_2D(int number_of_filters, std::vector<int> filter_dimensions)
{

    int filter_size = 1;
    for (int i = 0; i < filter_dimensions.size(); i++)
    {
        this->dimensions.push_back(filter_dimensions[i]);
        filter_size *= filter_dimensions[i];
    }
    
    for (int i = 0; i < number_of_filters; i++)
    {
        neuron *temp = new neuron(filter_size);
        this->filters.push_back(temp);
    }

}

std::vector<std::vector<std::vector<double>>> convolution_layer_2D::feed_forward(std::vector<std::vector<double>> input)
{

    this->input = input;

    std::vector<std::vector<std::vector<double>>> output(this->filters.size());
    
    for (int i = 0; i < this->filters.size(); i++)
    {
        std::vector<std::vector<double>> filter_output;
        for (int j = (int) this->dimensions[0] / 2; j < (int) input.size() - this->dimensions[0] / 2; j++)
        {
            std::vector<double> row;
            for (int k = (int) this->dimensions[1] / 2; k < (int) input.size() - this->dimensions[1] / 2; k++)
            {
                std::vector<double> cell;
                for (int l = (int) j - this->dimensions[0] / 2; l <= (int) j + this->dimensions[0] / 2; l++)
                {
                    for (int m = (int) k - this->dimensions[1] / 2; m <= (int) k + this->dimensions[1] / 2; m++)
                    {
                        cell.push_back(input[l][m]);
                    }
                }
                row.push_back(this->filters[i]->feed_forward(cell));
            }
            filter_output.push_back(row);
        }
        output[i] = filter_output;
    }

    return output;

}

void convolution_layer_2D::back_propagate(std::vector<std::vector<std::vector<double>>> dL_dout, double learn_rate)
{

    for (int i = 0; i < this->filters.size(); i++)
    {
        std::vector<double> dL_dfilter((this->dimensions[0] * this->dimensions[1]), 0);
        for (int j = 0; j < this->dimensions[0]; j++)
        {
            for (int k = 0; k < this->dimensions[1]; k++)
            {
                for (int l = 0; l < dL_dout[i].size(); l++)
                {
                    for (int m = 0; m < dL_dout[i][l].size(); m++)
                    {
                        dL_dfilter[j * this->dimensions[1] + k] += dL_dout[i][l][m] * this->input[j + l][k + m];
                    }
                }
            }
        }
        this->filters[i]->back_propagate(dL_dfilter, learn_rate);
    }

}

pooling_layer_2D::pooling_layer_2D(std::vector<int> filter_dimensions)
{

    for (int i = 0; i < filter_dimensions.size(); i++)
    {
        this->dimensions.push_back(filter_dimensions[i]);
    }

}

std::vector<std::vector<std::vector<double>>> pooling_layer_2D::feed_forward(std::vector<std::vector<std::vector<double>>> input)
{

    this->input = input;

    std::vector<std::vector<std::vector<double>>> output(input.size());
    for (int i = 0; i < input.size(); i++)
    {
        std::vector<std::vector<double>> filter_output;
        for (int j = 0; j < input[i].size(); j += this->dimensions[0])
        {
            std::vector<double> row;
            for (int k = 0; k < input[i][j].size(); k += this->dimensions[1])
            {
                std::vector<double> cell;
                for (int l = j; l < j + this->dimensions[0]; l++)
                {
                    for (int m = k; m < k + this->dimensions[1]; m++)
                    {
                        cell.push_back(input[i][l][m]);
                    }
                }
                row.push_back(*std::max_element(cell.begin(), cell.end()));
            }
            filter_output.push_back(row);
        }
        output[i] = filter_output;
    }

    return output;

}

std::vector<std::vector<std::vector<double>>> pooling_layer_2D::back_propagate(std::vector<std::vector<std::vector<double>>> dL_dout)
{

    std::vector<std::vector<std::vector<double>>> dL_din;
    for (int i = 0; i < this->input.size(); i++)
    {
        std::vector<std::vector<double>> grid;
        for (int j = 0; j < this->input[i].size(); j++)
        {
            std::vector<double> row(this->input[i][j].size(), 0);
            grid.push_back(row);
        }
        dL_din.push_back(grid);
    }

    for (int i = 0; i < this->input.size(); i++)
    {
        for (int j = 0; j < this->input[i].size(); j += this->dimensions[0])
        {
            for (int k = 0; k < this->input[i][j].size(); k += this->dimensions[1])
            {
                int max_j = j, max_k = k;
                for (int l = j; l < j + this->dimensions[0]; l++)
                {
                    for (int m = k; m < k + this->dimensions[1]; m++)
                    {
                        if (this->input[i][l][m] > this->input[i][max_j][max_k])
                        {
                            max_j = l;
                            max_k = m;
                        }
                    }
                }
                dL_din[i][max_j][max_k] = dL_dout[i][j / this->dimensions[0]][k / this->dimensions[1]];
            }
        }
    }

    return dL_din;

}

softmax_layer::softmax_layer(int nodes, std::vector<int> dimensions)
{

    this->dimensions = dimensions;

    int input_size = 1;
    for (int i : dimensions)
    {
        input_size *= i;
    }

    for (int i = 0; i < nodes; i++)
    {
        neuron *temp = new neuron(input_size + 1);
        this->nodes.push_back(temp);
    }
    
}

std::vector<double> softmax_layer::feed_forward(std::vector<std::vector<std::vector<double>>> input)
{

    std::vector<double> output;

    std::vector<double> flattened_input;
    for (int i = 0; i < input.size(); i++)
    {
        for (int j = 0; j < input[i].size(); j++)
        {
            for (int k = 0; k < input[i][j].size(); k++)
            {
                flattened_input.push_back(input[i][j][k]);
            }
        }
    }
    flattened_input.push_back(1);
    this->flattened_input = flattened_input;

    for (int i = 0; i < this->nodes.size(); i++)
    {
        double x = this->nodes[i]->feed_forward(flattened_input);
        output.push_back(x);
    }

    this->x = output;

    double sum = 0;
    for (int i = 0; i < output.size(); i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    for (int i = 0; i < output.size(); i++)
    {
        output[i] /= sum;
    }

    return output;

}

std::vector<std::vector<std::vector<double>>> softmax_layer::back_propagate(std::vector<double> dL_dout, double learn_rate)
{

    int index;

    for (index = 0; index < dL_dout.size(); index++)
    {
        if (dL_dout[index] != 0)
        {
            break;
        }
    }

    std::vector<std::vector<std::vector<double>>> dL_din;

    std::vector<double> factor;
    double sum = 0;
    for (int i = 0; i < this->x.size(); i++)
    {
        factor.push_back(exp(this->x[i]));
        sum += factor[i];
    }

    std::vector<double> dout_dx;
    for (int i = 0; i < factor.size(); i++)
    {
        dout_dx.push_back((-1) * factor[i] * factor[i] / sum / sum);
    }
    dout_dx[index] = factor[index] * (sum - factor[index]) / sum / sum;

    std::vector<double> dL_dx;
    for (int i = 0; i < dout_dx.size(); i++)
    {
        dL_dx.push_back(dL_dout[index] * dout_dx[i]);
    }

    for (int i = 0; i < this->dimensions[2]; i++)
    {
        std::vector<std::vector<double>> grid;
        for (int j = 0; j < this->dimensions[1]; j++)
        {
            std::vector<double> row;
            for (int k = 0; k < this->dimensions[0]; k++)
            {
                double cell = 0;
                for (int l = 0; l < this->nodes.size(); l++)
                {
                    cell += this->nodes[l]->weights[i * this->dimensions[1] * this->dimensions[0] + j * this->dimensions[0] + k] * dL_dx[l];
                }
                row.push_back(cell);
            }
            grid.push_back(row);
        }
        dL_din.push_back(grid);
    }

    for (int i = 0; i < this->nodes.size(); i++)
    {
        for (int j = 0; j < flattened_input.size(); j++)
        {
            flattened_input[j] = flattened_input[j] * dL_dx[i];
        }
        flattened_input.push_back(dL_dx[i]);
        this->nodes[i]->back_propagate(flattened_input, learn_rate);
        flattened_input.pop_back();
        for (int j = 0; j < flattened_input.size(); j++)
        {
            flattened_input[j] = flattened_input[j] / dL_dx[i];
        }
    }

    return dL_din;

}