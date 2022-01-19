'''
https://www.tensorflow.org/tutorials/load_data/tfrecord#walkthrough_reading_and_writing_image_data
'''


import pathlib

import tensorflow as tf
#Let's create another function to print our predictions:
import matplotlib.pylab as plt
from PIL import Image
import io


# cat_in_snow  = tf.keras.utils.get_file(
#     '320px-Felis_catus-cat_on_snow.jpg',
#     'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
#
# data_dir="E:\\image_net"
# data_dir = pathlib.Path(data_dir)

peacock = 'E:\\image_net\\peacock.jpeg'
cock = 'E:\\image_net\\cock.jpeg'
# file_names_dataset = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=False)

# williamsburg_bridge = tf.keras.utils.get_file(
#     '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
#     'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')


'''
Write the TFRecord file
As before, encode the features as types compatible with tf.train.Example. 
This stores the raw image string feature, as well as the height, width, depth, and arbitrary label feature. T
he latter is used when you write the file to distinguish between the cat image and the bridge image. 
Use 0 for the cat image, and 1 for the bridge image:
'''
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


image_labels = {
    cock : 7,
    peacock : 84,
}

# This is an example, just using the cat image.
image_string = open(cock, 'rb').read()

label = image_labels[cock]

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
  image_shape = tf.io.decode_jpeg(image_string).shape

  feature = {
      #'height': _int64_feature(image_shape[0]),
      #'width': _int64_feature(image_shape[1]),
      #'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')

# Write the raw image files to `images.tfrecords`.
# First, process the two images into `tf.train.Example` messages.
# Then, write to a `.tfrecords` file.
record_file = 'E:\\image_net\\images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
    image_string = open(filename, 'rb').read()
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())

'''
Read the TFRecord file
You now have the file—images.tfrecords—and can now iterate over the records in it to read back what you wrote. 
Given that in this example you will only reproduce the image, the only feature you will need is the raw image string. 
Extract it using the getters described above, namely example.features.feature['image_raw'].bytes_list.value[0]. 
You can also use the labels to determine which record is the cat and which one is the bridge:
'''

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    #'height': tf.io.FixedLenFeature([], tf.int64),
    #'width': tf.io.FixedLenFeature([], tf.int64),
    #'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset

#Recover the images from the TFRecord file:

for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  image = Image.open(io.BytesIO(image_raw))
  plt.imshow(image)
  plt.grid(False)
  plt.show()
