/*
    Shubh Khandelwal
*/

#include <datapack.h>
#include <iostream>
#include <layers.h>
#include <unistd.h>

int main()
{

    std::cout << "\nBASIC CONVOLUTION NEURAL NETWORK IMPLEMENTATION" << std::endl;

    datapack<int> sample("dataset/emnist-mnist-train.csv");

    convolution_layer_2D c(8, {3, 3});
    pooling_layer_2D p({2, 2});
    softmax_layer s(10, {13, 13, 8});

    std::vector<std::vector<std::vector<double>>> images;
    for (int i = 0; i < sample.size(); i++)
    {
        std::vector<std::vector<double>> image;
        for (int j = 0; j < 28; j++)
        {
            std::vector<double> row;
            for (int k = 0; k < 28; k++)
            {
                row.push_back(sample[i][28 * j + k] / 255 - 0.5);
            }
            image.push_back(row);
        }
        images.push_back(image);
    }

    std::cout << "\nTraining start!" << std::endl;
    int positive = 0;
    for (int i = 0; i < images.size(); i++)
    {

        std::vector<std::vector<std::vector<double>>> output_c = c.feed_forward(images[i]);
        std::vector<std::vector<std::vector<double>>> output_p = p.feed_forward(output_c);
        std::vector<double> output_s = s.feed_forward(output_p);

        int index = 0;
        for (int j = 1; j < output_s.size(); j++)
        {
            if (output_s[j] > output_s[index])
            {
                index = j;
            }
        }

        std::vector<double> dL_dout(output_s.size(), 0);
        dL_dout[sample[i][0]] = -1 / output_s[sample[i][0]];

        double learn_rate = 0.01;
        std::vector<std::vector<std::vector<double>>> dL_ds = s.back_propagate(dL_dout, learn_rate);
        std::vector<std::vector<std::vector<double>>> dL_dp = p.back_propagate(dL_ds);
        c.back_propagate(dL_dp, learn_rate);

        if (index == sample[i][0])
        {
            positive++;
        }

        if ((i + 1) % 100 == 0)
        {
            std::cout << "[" << i + 1 << " steps]: Accuracy:- " << positive << "%" << std::endl;
            positive = 0;
        }

    }
    std::cout << "\nTraining complete!\n" << std::endl;

    return 0;

}