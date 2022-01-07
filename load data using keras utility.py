"""
Author  : sarala kumarage
Copyright : Jan 2022

Reference
https://www.tensorflow.org/tutorials/load_data/images#load_data_using_a_keras_utility
"""

import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np

# data_dir="E:\\flower_photos"
# data_dir = pathlib.Path(data_dir)

#Download the image data set from the cloud -  vedio/blog content 1

#1.define the image data set url
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

#2.
# get the image data set url
# get the folder name
# unzip the image folder
# use tf.keras.utils.get_file() method to download the image data set
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)

data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=5, #If using `validation_split` and shuffling the data, you must provide a `seed` argument, to make sure that there is no overlap between the training and validation subset.
  image_size=(img_height, img_width),
  batch_size=batch_size)

#label :  tf.Tensor([1 1 4 0 4], shape=(5,), dtype=int32)


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#You can find the class names in the class_names attribute on these datasets
class_names = train_ds.class_names
print(class_names)

'''
#visualize the data
plt.figure(figsize=(10, 10))
for image_batch, label_batch in train_ds.take(1):
  #train_ds is a batched data set - images has an image batch, labels have respective labels batch , these batches are tensors -can eb converted
  #into numpy arrays
  for i in range(10):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    plt.title(class_names[label_batch[i]])
    plt.axis("off")
  #plt.show()

'''
#Standardize the data
#The RGB channel values are in the [0, 255] range. This is not ideal for a neural network; in general you should seek to make your input values small.

#Here, you will standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling

'''
For instance:
To rescale an input in the [0, 255] range to be in the [0, 1] range, you would pass scale=1./255.

To rescale an input in the [0, 255] range to be in the [-1, 1] range, you would pass scale=1./127.5, offset=-1.

The rescaling is applied both during training and inference
'''
normalization_layer = tf.keras.layers.Rescaling(1./255) # can do like this or include this layer in the model

#vectorize mapping - apply batching before map , here since train_ds is a batched dataset its ok otherwise call train_ds.batch() before map()

#parallelize data transformation - mapping
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y),num_parallel_calls=tf.data.AUTOTUNE)

'''
#train_ds -> BatchDataset
#normalized_ds - > <class 'tensorflow.python.data.ops.dataset_ops.MapDataset'>
image_batch, labels_batch = next(iter(normalized_ds))
#image_batch - > <class 'tensorflow.python.framework.ops.EagerTensor'>

#take the first image of the batch
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
'''

#Configure the dataset for performance

'''
Let's make sure to use buffered prefetching so you can yield data from disk without having I/O become blocking. These are two important methods you should use when loading data:

Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
Dataset.prefetch overlaps data preprocessing and model execution while training.
'''

AUTOTUNE = tf.data.AUTOTUNE
#check if the train_ds set can fit in to memory
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
##<BatchDataset shapes: ((None, 180, 180, 3), (None,)), types: (tf.float32, tf.int32)>
#<CacheDataset shapes: ((None, 180, 180, 3), (None,)), types: (tf.float32, tf.int32)>
#<PrefetchDataset shapes: ((None, 180, 180, 3), (None,)), types: (tf.float32, tf.int32)>

#check if the val_ds set can fit in to memory
# cache a dataset, either in memory or on local storage
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#print(train_ds)
#define the model
num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

#compile the model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

#train the model
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)