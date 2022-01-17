#Setup

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib

#Train a TensorFlow model

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(
  train_images,
  train_labels,
  epochs=1,
  validation_data=(test_images, test_labels)
)

'''
Convert to a TensorFlow Lite model
Using the Python TFLiteConverter, you can now convert the trained model into a TensorFlow Lite model.

Now load the model using the TFLiteConverter:
'''

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


#Write it out to a tflite file:


tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/")
print(tflite_models_dir)
#D:\tmp\mnist_tflite_models
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"mnist_model.tflite"
print(tflite_model_file.write_bytes(tflite_model))
#84540

#Run the TFLite models
'''
Run the TensorFlow Lite model using the Python TensorFlow Lite Interpreter.

Load the model into an interpreter
'''

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

