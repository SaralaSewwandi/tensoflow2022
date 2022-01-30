import tempfile
import os

import tensorflow as tf

from tensorflow import keras

import tempfile
import os

import tensorflow as tf

from tensorflow import keras
import tensorflow_model_optimization as tfmot

#Train a model for MNIST with quantization aware training

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])



quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 quantize_layer (QuantizeLay  (None, 28, 28)           3         
 er)                                                             
                                                                 
 quant_reshape (QuantizeWrap  (None, 28, 28, 1)        1         
 perV2)                                                          
                                                                 
 quant_conv2d (QuantizeWrapp  (None, 26, 26, 12)       147       
 erV2)                                                           
                                                                 
 quant_max_pooling2d (Quanti  (None, 13, 13, 12)       1         
 zeWrapperV2)                                                    
                                                                 
 quant_flatten (QuantizeWrap  (None, 2028)             1         
 perV2)                                                          
                                                                 
 quant_dense (QuantizeWrappe  (None, 10)               20295     
 rV2)                                                            
                                                                 
=================================================================
Total params: 20,448
Trainable params: 20,410
Non-trainable params: 38
_________________________________________________________________
'''

q_aware_model.fit(  train_images,
  train_labels,
  epochs=1,
  validation_split=0.1)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   test_images, test_labels, verbose=0)

print("q_aware_model_accuracy",q_aware_model_accuracy)
#q_aware_model_accuracy 0.9595999717712402