"""
Author  : sarala kumarage
Copyright : Jan 2022

Reference
https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance
"""
import tensorflow as tf

dataset = tf.data.Dataset.range(10)

# 1. vectorized or batched before mapping
dataset = dataset.batch(2)

@tf.autograph.experimental.do_not_convert
def multipy(x):
    return 2*x

# 2. interleaving & time consuming mapping
#map function in interleave - 	A function that takes a dataset element and returns a tf.data.Dataset.
dataset = dataset.interleave(lambda x: dataset.from_tensors(x).map(multipy, num_parallel_calls= tf.data.AUTOTUNE),
    cycle_length=4, block_length=16,num_parallel_calls=tf.data.AUTOTUNE)

# 3. caching - >>to memory  if you want to cache to a file specify a file
dataset = dataset.cache()

'''# 4.interleaving & memory consuming mapping
#map function in interleave - 	A function that takes a dataset element and returns a tf.data.Dataset.
dataset = dataset.interleave(lambda x: dataset.from_tensors(x**2),
    cycle_length=4, block_length=16)
'''

# 5. prefetching -  last step before training
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print(list(dataset.as_numpy_iterator()))