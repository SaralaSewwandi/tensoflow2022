"""
Author : Sarala Kumarage
"""

import pathlib
import os
import logging

logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

def validate_resnet101(data_dir):
    #data_dir path to validation data set
    data_dir = pathlib.Path(data_dir)

    # The tree structure of the files can be used to compile a class_names list.
    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

    # convert a file path to an (img, label) pair:
    def get_label(file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)


    # process file path to get the label and image to create a pair
    def process_path(file_path):
        label = get_label(file_path)
        img = file_path
        return img, label

    #iterate through all the folders and files inside the folders and create a image file list
    file_names_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)

    #call process path on each image file path and prepare a validation data set with imagse and their corresponding labels
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    validation_dataset = file_names_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    #load the ResNet101 model with trained imagenet weights
    model = ResNet101(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000

    )

    correct = 0
    count = 0
    acc = 0
    #call inference on the trained resnet model on each image on the validation dataset
    for img_path, label in validation_dataset:
        #convert tensor image file path to string
        img_path = str(img_path.numpy(), 'UTF-8')
        #load the image with specified height and width for resnet
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        #convert the image to array
        x = image.img_to_array(img)
        #expand the image dimension by one since the model was trained to get batches as inputs with higher dimension
        x = np.expand_dims(x, axis=0)
        #perform preprocess input for the ResNet model - this is a must before passing an input to the resnet model, there are resnet specific preprocessing
        x = preprocess_input(x)
        #call predict on the image
        preds = model.predict(x)
        #get the top prediction with the highest prediction probability for the class
        top1_outputs = decode_predictions(preds, top=1)[0]
        #get the predicted class name from the inference , based on the class name from the decoded results (class, description, probability)
        predcited_class = top1_outputs[0][0]
        #get the class name for the image label
        actual = class_names[label]
        #check whether the predicted class is equal to the actual class
        if (predcited_class == actual):
            correct = correct + 1
        count = count + 1

    #calculate the accuracy
    acc = (correct / count) * 100
    print('Correctly Predicted:', correct)
    print('Images Count:', count)
    print('Validation Accuracy:', acc)


def main():
    '''
    download the imagenet data set with class name folders
    so this script validated the model based on the decoded class name
    https://drive.google.com/drive/u/1/folders/10pJ28cmO2KfdDdfX9uC0iNZnMPSp9ZM8

    '''
    print("validation started")

    data_dir = 'D:\\smaller_imagenet_validation'
    validate_resnet101(data_dir)

    print("validation completed")


if __name__ == "__main__":
    main()