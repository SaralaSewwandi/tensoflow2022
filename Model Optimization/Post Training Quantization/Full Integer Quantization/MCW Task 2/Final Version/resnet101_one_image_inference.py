import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

def validate_resnet(img_path):
    model = ResNet101(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000
    )

    #img = image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    top1_output = decode_predictions(preds, top=1)[0]
    #based on the class from decoded results (class, description, probability)
    predcited_class=top1_output[0][0]
    print('Predicted:',predcited_class)
    print('Predicted Probability:', top1_output[0][2])

def main():
    img_path = 'D:\\smaller_imagenet_validation\\n01440764\\ILSVRC2012_val_00040358.jpeg'
    validate_resnet(img_path)

if __name__ == "__main__":
    main()


