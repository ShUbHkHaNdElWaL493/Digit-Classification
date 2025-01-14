#   Shubh Khandelwal

import numpy as np

class convolution_layer_3x3:

    def __init__(self, number_of_filters):
        
        self.number_of_filters = number_of_filters
        self.filters = np.random.randn(self.number_of_filters, 3, 3) / 9

    def iterate_regions(self, image):

        h, w = image.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                image_region = image[(i - 1) : (i + 2), (j - 1) : (j + 2)]
                yield image_region, i, j
    
    def feed_forward(self, input):

        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.number_of_filters))

        for image_region, i, j in self.iterate_regions(input):
            output[i - 1, j - 1] = np.sum(image_region * self.filters, axis = (1, 2))

        return output
    
    def back_propagate(self, dL_dout, learn_rate):

        dL_dfilters = np.zeros(self.filters.shape)

        for image_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.number_of_filters):
                dL_dfilters[f] += dL_dout[i - 1, j - 1, f] * image_region

        self.filters -= learn_rate * dL_dfilters

        return None
    
class pooling_layer_2x2:

    def iterate_regions(self, image):

        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                image_region = image[(2 * i) : (2 * i + 2), (2 * j) : (2 * j + 2)]
                yield image_region, i, j

    def feed_forward(self, input):

        self.last_input = input

        h, w, number_of_filters = input.shape
        output = np.zeros((h // 2, w // 2, number_of_filters))

        for image_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(image_region, axis = (0, 1))
        
        return output
    
    def back_propagate(self, dL_dout):

        dL_din = np.zeros(self.last_input.shape)

        for image_region, i, j in self.iterate_regions(self.last_input):

            h, w, f = image_region.shape
            max_value = np.amax(image_region, axis = (0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if image_region[i2, j2, f2] == max_value[f2]:
                            dL_din[i * 2 + i2, j * 2 + j2, f2] = dL_dout[i, j, f2]
        
        return dL_din
    
class softmax_layer:

    def __init__(self, input_length, nodes):

        self.weights = np.random.randn(input_length, nodes) / input_length
        self.biases = np.zeros(nodes)

    def feed_forward(self, input):

        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input
        input_length, nodes = self.weights.shape

        x = np.dot(input, self.weights) + self.biases
        self.last_x = x
        factor = np.exp(x)
        return factor / np.sum(factor, axis = 0)
    
    def back_propagate(self, dL_dout, learn_rate):

        for i, gradient in enumerate(dL_dout):

            if gradient == 0:
                continue

            factor = np.exp(self.last_x)
            S = np.sum(factor)

            dout_dx = -factor[i] * factor / (S ** 2)
            dout_dx[i] = factor[i] * (S - factor[i]) / (S ** 2)

            dx_dw = self.last_input
            dx_db = 1
            dx_din = self.weights

            dL_dx = gradient * dout_dx

            dL_dw = dx_dw[np.newaxis].T @ dL_dx[np.newaxis]
            dL_db = dL_dx * dx_db
            dL_din = dx_din @ dL_dx

            self.weights -= learn_rate * dL_dw
            self.biases -= learn_rate * dL_db

            return dL_din.reshape(self.last_input_shape)