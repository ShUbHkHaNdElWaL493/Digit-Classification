#   Shubh Khandelwal

import layers
from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

convolution = layers.convolution_layer_3x3(8)
pooling = layers.pooling_layer_2x2()
softmax = layers.softmax_layer((13 * 13 * 8), 10)

def feed_forward(image, label):

    output_c = convolution.feed_forward(image / 255 - 0.5)
    output_p = pooling.feed_forward(output_c)
    output_s = softmax.feed_forward(output_p)

    loss = -np.log(output_s[label])
    accuracy = 1 if np.argmax(output_s) == label else 0

    return output_s, loss, accuracy

def train(image, label, learn_rate = 0.005):

    output_s, loss, accuracy = feed_forward(image, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / output_s[label]

    gradient = softmax.back_propagate(gradient, learn_rate)
    gradient = pooling.back_propagate(gradient)
    gradient = convolution.back_propagate(gradient, learn_rate)

    return loss, accuracy

print("\nMNIST CONVOLUTION NEURAL NETWORK\n")
print("Training start!")

for epoch in range(3):
    correct = 0
    loss = 0
    for i, (image, label) in enumerate(zip(train_images[:1000], train_labels[:1000])):

        l, c = train(image, label)
        correct += c
        loss += l

        if i % 100 == 99:
            print("[Step %d] Average Loss %.3f | Accuracy: %d%%" % (i + 1, loss / 100, correct))
            loss = 0
            correct = 0

print("Training end!")

print("\nTesting start!")

correct = 0
loss = 0
for i, (image, label) in enumerate(zip(test_images[:1000], test_labels[:1000])):

    _, l, c = feed_forward(image, label)
    correct += c
    loss += l

    if i % 100 == 99:
        print("[Step %d] Average Loss %.3f | Accuracy: %d%%" % (i + 1, loss / 100, correct))
        loss = 0
        correct = 0

print("Testing end!")