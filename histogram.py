
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

'''
class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (1,), dtype = tf.int64),
            args=(num_samples,)
        )

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)

print(tf.data.Dataset.range(2))
benchmark(ArtificialDataset())


x = np.array([[1,1,1, 2,3,5,5, 4]])

x=tf.convert_to_tensor(x)
plt.hist(x)
plt.show()
'''

def multiply(x):
    return 2*x

#print(tf.data.Dataset.range(2))
dataset = tf.data.Dataset.range(10)  # ==> [ 1, 2, 3, 4, 5 ]
dataset = dataset.batch(2).map(multiply, num_parallel_calls=tf.data.AUTOTUNE)
img = tf.io.read_file('E:\\flower_photos\\roses\\22679076_bdb4c24401_m.jpg')
print(list(dataset.as_numpy_iterator()))