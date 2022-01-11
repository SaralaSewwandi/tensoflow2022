import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras
'''
1.Train a model for MNIST without pruning
'''
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
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

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  train_images,
  train_labels,
  epochs=4,
  validation_split=0.1,
)

#Evaluate baseline test accuracy and save the model for later usage.
_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)


'''
OUTPUT
======================================================================
Epoch 1/4 
1688/1688 [==============================] - 12s 7ms/step - loss: 0.2888 - accuracy: 0.9186 - val_loss: 0.1099 - val_accuracy: 0.9698 

Epoch 2/4 
1688/1688 [==============================] - 12s 7ms/step - loss: 0.1098 - accuracy: 0.9686 - val_loss: 0.0821 - val_accuracy: 0.9780 

Epoch 3/4 
1688/1688 [==============================] - 12s 7ms/step - loss: 0.0814 - accuracy: 0.9763 - val_loss: 0.0744 - val_accuracy: 0.9788 

Epoch 4/4 
1688/1688 [==============================] - 12s 7ms/step - loss: 0.0680 - accuracy: 0.9791 - val_loss: 0.0754 - val_accuracy: 0.9793 

Baseline test accuracy: 0.9757999777793884 
#Saved baseline model to: C:\\Users\\SARALA~1\\AppData\\Local\\Temp\\tmpyj_exab_.h5 

'''

'''
2.
Fine-tune pre-trained model with pruning
Define the model
You will apply pruning to the whole model and see this in the model summary.

In this example, you start the model with 50% sparsity (50% zeros in weights) and end with 80% sparsity.
'''

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set.

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model_for_pruning.summary())

'''
OUTPUT
=======================================================================
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 prune_low_magnitude_reshape  (None, 28, 28, 1)        1         
  (PruneLowMagnitude)                                            
                                                                 
 prune_low_magnitude_conv2d   (None, 26, 26, 12)       230       
 (PruneLowMagnitude)                                             
                                                                 
 prune_low_magnitude_max_poo  (None, 13, 13, 12)       1         
 ling2d (PruneLowMagnitude)                                      
                                                                 
 prune_low_magnitude_flatten  (None, 2028)             1         
  (PruneLowMagnitude)                                            
                                                                 
 prune_low_magnitude_dense (  (None, 10)               40572     
 PruneLowMagnitude)                                              
                                                                 
=================================================================
Total params: 40,805
Trainable params: 20,410
Non-trainable params: 20,395
_________________________________________________________________
None
=========================================================================

'''

#Train and evaluate the model against baseline
'''
Fine tune with pruning for two epochs.

tfmot.sparsity.keras.UpdatePruningStep is required during training, and tfmot.sparsity.keras.PruningSummaries provides logs for tracking progress and debugging.
'''

logdir = tempfile.mkdtemp()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_images, train_labels,
                      batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                      callbacks=callbacks)

#For this example, there is minimal loss in test accuracy after pruning, compared to the baseline.
_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Pruned test accuracy:', model_for_pruning_accuracy)

'''
OUTPUT
Epoch 1/2
  4/422 [..............................] - ETA: 8s - loss: 0.0833 - accuracy: 0.9707   WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0156s vs `on_train_batch_end` time: 0.0206s). Check your callbacks.
422/422 [==============================] - 16s 28ms/step - loss: 0.0854 - accuracy: 0.9767 - val_loss: 0.1041 - val_accuracy: 0.9793
Epoch 2/2
422/422 [==============================] - 10s 23ms/step - loss: 0.1026 - accuracy: 0.9729 - val_loss: 0.0859 - val_accuracy: 0.9785
Baseline test accuracy: 0.9793999791145325
Pruned test accuracy: 0.9743000268936157

there is minimal loss in test accuracy after pruning, compared to the baseline
'''

'''
3.
Create 3x smaller models from pruning
'''

'''
3.1
Both tfmot.sparsity.keras.strip_pruning and applying a standard compression algorithm (e.g. via gzip) are necessary to see the compression benefits of pruning.

strip_pruning is necessary since it removes every tf.Variable that pruning only needs during training, which would otherwise add to model size during inference
Applying a standard compression algorithm is necessary since the serialized weight matrices are the same size as they were before pruning. 
However, pruning makes most of the weights zeros, which is added redundancy that algorithms can utilize to further compress the model.
'''

#First, create a compressible model for TensorFlow.
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

#Then, create a compressible model for TFLite.
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

#Define a helper function to actually compress the models via gzip and measure the zipped size.

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

#Compare and see that the models are 3x smaller from pruning.
print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))


'''
OUTPUT
Saved pruned Keras model to: C:\\Users\\SARALA~1\\AppData\\Local\\Temp\\tmp8fjyd39m.h5
Saved pruned TFLite model to: C:\\Users\\SARALA~1\\AppData\\Local\\Temp\\tmp1a20eh_x.tflite

Size of gzipped baseline Keras model: 78269.00 bytes
Size of gzipped pruned Keras model: 25742.00 bytes
Size of gzipped pruned TFlite model: 25145.00 bytes
'''

'''
4.
Create a 10x smaller model from combining pruning and quantization
You can apply post-training quantization to the pruned model for additional benefits
'''

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))

'''
OUTPUT
Saved quantized and pruned TFLite model to: C:\Users\SARALA~1\AppData\Local\Temp\tmp6w6zjw60.tflite
Size of gzipped baseline Keras model: 78342.00 bytes
Size of gzipped pruned and quantized TFlite model: 8219.00 bytes
'''
