import tensorflow_datasets as tfds
import tensorflow as tf

import os

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']