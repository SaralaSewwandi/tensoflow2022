import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(tf.__version__)
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()