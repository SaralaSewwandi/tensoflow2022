import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                testing_wav_file_name=tf.keras.utils.get_file('miaow_16k.wav',
                                                                                              'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                                                              cache_dir='./',
                                                                                              cache_subdir='test_data')

print(testing_wav_file_name)
cache_dir='./',
                                                cache_subdir='test_data')

print(testing_wav_file_name)