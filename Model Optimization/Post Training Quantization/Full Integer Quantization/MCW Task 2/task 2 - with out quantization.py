#Setup
#In order to quantize both the input and output tensors, we need to use APIs added in TensorFlow r2.3:


import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
assert float(tf.__version__[:3]) >= 2.3

'''
Generate a TensorFlow Model
We'll build a simple model to classify numbers from the MNIST dataset.

This training won't take long because you're training the model for just a 5 epochs, which trains to about ~98% accuracy.
'''

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
  hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4")
])

#First, here's a converted model with no quantization:
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

#It's now a TensorFlow Lite model, but it's still using 32-bit float values for all parameter data.

'''
Save the models as files
You'll need a .tflite file to deploy your model on other devices.
So let's save the converted models to files and then load them when we run inferences below.
'''
import pathlib

tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/full_integer_quantization")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)


'''
Run the TensorFlow Lite models
Now we'll run inferences using the TensorFlow Lite Interpreter to compare the model accuracies.

First, we need a function that runs inference with a given model and images, and then returns the predictions:
'''

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
  global test_images

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions


'''
Test the models on one image
Now we'll compare the performance of the float model and quantized model:

tflite_model_file is the original TensorFlow Lite model with floating-point data.
tflite_model_quant_file is the last model we converted using integer-only quantization (it uses uint8 data for input and output).
'''

#Let's create another function to print our predictions:
import matplotlib.pylab as plt

# Change this to test a different image
test_image_index = 1

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type):
  global test_labels

  predictions = run_tflite_model(tflite_file, [test_image_index])

  plt.imshow(test_images[test_image_index])
  template = model_type + " Model \n True:{true}, Predicted:{predict}"
  _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
  plt.grid(False)
  plt.show()

test_model(tflite_model_file, test_image_index, model_type="Float")


'''
Evaluate the models on all images
Now let's run both models using all the test images we loaded at the beginning of this tutorial:
'''

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global test_images
  global test_labels

  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)

  accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))

evaluate_model(tflite_model_file, model_type="Float")
