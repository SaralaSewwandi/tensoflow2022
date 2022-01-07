import tensorflow as tf
import pathlib
import PIL.Image

print(tf.__version__)

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


# data gets downloaded to C:\Users\<username\.keras\datasets\ folder
#
# C:\Users\Sarala Kumarage\.keras\datasets\flower_photos
#
# data_dir is a str which defines the image data set folder path
#
# <class 'str'>

# convert the str path to a system path depending on the operating system  -  vedio/blog content 2
# so that we can make system calls based on the OS(windows  linux)
data_dir = pathlib.Path(data_dir)

#<class 'str'> ----> <class 'pathlib.WindowsPath'>


#Iterate over all the jpg files in all subdirectories & return generator object Path.glob & then convert it to a windows file path list - -  vedio/blog content 3
# define the pattern to decide the files and folders to iterate through
# * - all subdirectories
# *.jpg - all jpg files


image_files = list(data_dir.glob('*/*.jpg'))
image_count = len(image_files)
print(image_count)

#open an image inside the roses folder in flowers_photos folder(data_dir) & display in the image viewer -  vedio/blog content 3
roses = list(data_dir.glob('roses/*'))

#open an image
image = PIL.Image.open(str(roses[0]))

#display in the image viewer
#image.show()

# load the images off the disk - vedio/blog content 4
# using tf.keras.utils.image_dataset_from_directory utility create a image data set

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.8,
  subset="training",
  seed=123, #random number to start shuffling / transformations in a sequence
  image_size=(img_height, img_width),
  batch_size=batch_size)

#print(train_ds)
#<BatchDataset shapes: ((None, 180, 180, 3), (None,)), types: (tf.float32, tf.int32)>

#print(type(train_ds))
#<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>

print(type(train_ds)) #<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>
print(train_ds)
for i in train_ds:
    print(type(i)) #<class 'tuple'> ((None, 180, 180, 3), (None,))
    break
