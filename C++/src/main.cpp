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

    datapack<double> sample;

    convolution_layer_2D c(8, {3, 3});
    pooling_layer_2D p({2, 2});
    softmax_layer s(10, {13, 13, 8});

    std::cout << "\nTraining start!" << std::endl;
    int count = 0;
    int positive = 0;

    for (int epoch = 0; epoch < 1000; epoch++)
    {
        
        sample.open("dataset/emnist-mnist-train.csv");

        while (sample.next())
        {

            std::vector<double> row = sample.get_row();

            std::vector<std::vector<double>> image;
            for (int i = 0; i < 28; i++)
            {
                std::vector<double> line;
                for (int j = 0; j < 28; j++)
                {
                    line.push_back(row[28 * i + j + 1] / 255 - 0.5);
                }
                image.push_back(line);
            }

            // std::cout << "Image:" << std::endl;
            // for (int i = 0; i < image.size(); i++)
            // {
            //     for (int j = 0; j < image[i].size(); j++)
            //     {
            //         std::cout << image[i][j] << " ";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;
            // sleep(1);

            std::vector<std::vector<std::vector<double>>> output_c = c.feed_forward(image);
            std::vector<std::vector<std::vector<double>>> output_p = p.feed_forward(output_c);
            std::vector<double> output_s = s.feed_forward(output_p);

            int index = 0;
            for (int i = 1; i < output_s.size(); i++)
            {
                if (output_s[i] > output_s[index])
                {
                    index = i;
                }
            }

            std::vector<double> dL_dout(output_s.size(), 0);
            dL_dout[row[0]] = -1 / output_s[row[0]];

            double learn_rate = 0.001;
            std::vector<std::vector<std::vector<double>>> dL_ds = s.back_propagate(dL_dout, learn_rate);
            std::vector<std::vector<std::vector<double>>> dL_dp = p.back_propagate(dL_ds);
            c.back_propagate(dL_dp, learn_rate);

            if (index == row[0])
            {
                positive++;
            }

            if ((count + 1) % 100 == 0)
            {
                std::cout << "[" << count + 1 << " steps]: Accuracy:- " << positive << "%" << std::endl;
                positive = 0;
            }

            count++;

        }

        sample.close();

    }
    std::cout << "\nTraining complete!\n" << std::endl;

    return 0;

}