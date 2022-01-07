"""
Author  : sarala kumarage
Copyright : Jan 2022

Reference
https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control

"""
#For finer grain control, you can write your own input pipeline using tf.data.
import os

import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import glob

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

image_count = len(list(data_dir.glob('*/*.jpg')))


'''

tf.keras.utils.image_dataset_from_directory or tf.data.Dataset.list_files can be used to list files but globbing the file path first and then create a data set from tensorslices
is the most effiecient way
compare the performance of these three
#list all the files in the specified glob patterns (for files)
#create a tf dataset with the image file names
#this is reglobing every file name with list_files result in poor performance

The file_pattern argument should be a small number of glob patterns. 
If your filenames have already been globbed, use Dataset.from_tensor_slices(filenames) instead, 
as re-globbing every filename with list_files may result in poor performance with remote storage systems.

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)

# list_ds , <TensorSliceDataset shapes: (), types: tf.string>


list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
  print(f.numpy()) # convert to numpy array
'''

file_names_dataset = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)


#During training, it's important to shuffle the data well - poorly shuffled data can result in lower training accuracy.
shuffled_file_names_dataset = file_names_dataset.shuffle(image_count, reshuffle_each_iteration=False)

#print(list(shuffled_file_names_dataset.as_numpy_iterator()))

#prepare folder list with the class  names , can include a LICENSE.txt file but do not include any other additional file or folder in the root
#directory
#The tree structure of the files can be used to compile a class_names list.

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
#print(class_names)


#data set split - validation and training data sets
#image count =50  val_size=10
val_size = int(image_count * 0.2)
#create a data set skipping the first 10 elements, take last 40 elements
#<class 'tensorflow.python.data.ops.dataset_ops.SkipDataset'>
train_ds = shuffled_file_names_dataset.skip(val_size)
#take first 10 elements
#<class 'tensorflow.python.data.ops.dataset_ops.TakeDataset'>
val_ds = shuffled_file_names_dataset.take(val_size)

# print(type(train_ds))
# print(type(val_ds))
#print(tf.data.experimental.cardinality(train_ds).numpy())
#print(tf.data.experimental.cardinality(val_ds).numpy())

def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def process_path(file_path):
  # #<class 'tensorflow.python.framework.ops.EagerTensor'>
  # label = get_label(file_path)
  # # Load the raw data from the file as a string
  # img = tf.io.read_file(file_path)
  # # <class 'tensorflow.python.framework.ops.EagerTensor'>,
  # img = decode_img(img)
  # return img, label

  label = get_label(file_path)
  image = tf.io.read_file(file_path)
  image = tf.io.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [180, 180])
  return image, label


#Use Dataset.map to create a dataset of image, label pairs
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
#<class 'tensorflow.python.data.ops.dataset_ops.ParallelMapDataset'>
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
#<class 'tensorflow.python.data.ops.dataset_ops.ParallelMapDataset'>
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

'''

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print(type(image))
  print("Label: ", label.numpy())
  print(type(label))

'''

#Configure dataset for performance
def configure_for_performance(dataset):
    # 1. vectorized or batched before mapping
    dataset = dataset.batch(32)
    # 2. interleaving & time consuming mapping
    #map function in interleave - 	A function that takes a dataset element and returns a tf.data.Dataset.
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    dataset = dataset.interleave(lambda x,y: dataset.from_tensors((x,y)),
         cycle_length=4, block_length=16,num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: (normalization_layer(x), y), num_parallel_calls= tf.data.AUTOTUNE)

    # 3. caching - >>to memory  if you want to cache to a file specify a file, apply time consuming mapping before cache
    #dataset = dataset.cache()
    #Large datasets are sharded (split in multiple files) and typically do not fit in memory, so they should not be cached.



    # 4.interleaving & memory consuming mapping
    #map function in interleave - 	A function that takes a dataset element and returns a tf.data.Dataset. each element becomes a dataset
    # normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    # dataset = dataset.interleave(lambda x,y: dataset.from_tensors((x,y)),
    #      cycle_length=4, block_length=16,num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: (normalization_layer(x), y), num_parallel_calls= tf.data.AUTOTUNE)

    # 5. prefetching -  last step before training
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

    '''
    tf.data.Dataset.range(2)
    .interleave(  # Parallelize data reading
        dataset_generator_fun,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(  # Vectorize your mapped function
        _batch_map_num_items,
        drop_remainder=True)
    .map(  # Parallelize map transformation
        time_consuming_map,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .cache()  # Cache data
    .map(  # Reduce memory usage
        memory_consuming_map,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .prefetch(  # Overlap producer and consumer works
        tf.data.AUTOTUNE
    )
    '''

train_ds_batches = configure_for_performance(train_ds)


'''
for image_batch, label_batch in train_ds_batches.take(1):
  print("Image batch shape: ", image_batch.numpy().shape)
  #print(image)
  print(type(image_batch))
  print("Label batch: ", label_batch.numpy())
  print(type(label_batch))
  
'''

val_ds_batches = configure_for_performance(val_ds)

'''
for image_batch, label_batch in val_ds_batches.take(1):
  print("Image batch shape: ", image_batch.numpy().shape)
  #print(image)
  print(type(image_batch))
  print("Label batch: ", label_batch.numpy())
  print(type(label_batch))

#should be
#<PrefetchDataset shapes: ((None, 180, 180, 3), (None,)), types: (tf.float32, tf.int32)>

print(train_ds_batches)
#<PrefetchDataset shapes: ((None, 180, 180, 3), (None,)), types: (tf.float32, tf.int64)>
'''

#define the model
num_classes = 5

model = tf.keras.Sequential([
  #tf.keras.layers.Rescaling(1./255),
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
  train_ds_batches,
  validation_data=val_ds_batches,
  epochs=3
)




