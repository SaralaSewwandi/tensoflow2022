'''
Resnets with pre-activation layers (Resnet-v2) are widely used for vision applications. Pre-trained frozen graph for resnet-v2-101 is available on Tensorflow Hub.

You can convert the frozen graph to a TensorFLow Lite flatbuffer with quantization by:

'''
import tensorflow_hub as hub
import tensorflow as tf
import pathlib

resnet_v2_101 = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
  hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4")
])

converter = tf.lite.TFLiteConverter.from_keras_model(resnet_v2_101)


# Convert to TF Lite with quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/")
resnet_quantized_tflite_file = tflite_models_dir/"resnet_v2_101_quantized.tflite"
print(resnet_quantized_tflite_file.write_bytes(converter.convert()))