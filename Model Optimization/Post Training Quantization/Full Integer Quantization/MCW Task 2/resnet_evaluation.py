#Setup
#In order to quantize both the input and output tensors, we need to use APIs added in TensorFlow r2.3:


import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tfds
assert float(tf.__version__[:3]) >= 2.3

'''
# Define the model architecture
model = tf.keras.Sequential([
  # tf.keras.layers.InputLayer(input_shape=(None, None)),
  # tf.keras.layers.Reshape(target_shape=(224, 224, 3)),
  tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
  hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4")
])


def evaluate_resnet(test_images, test_labels):
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)

def inference_resnet(test_image):
  predictions = model.predict(test_image)
  return np.argmax(predictions[0])

peacock = 'E:\\image_net\\peacock.jpeg'
cock = 'E:\\image_net\\cock.jpeg'

test_images = [peacock,cock]
test_labels = [84,7]

evaluate_resnet(test_images, test_labels)

'''
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpeg'
#actual 386	African elephant, Loxodonta africana

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
