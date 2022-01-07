import tensorflow as tf

# Preprocess 4 files concurrently, and interleave blocks of 16 records
# from each file.
filenames = ["D:\\Tensorflow\\MLbasics\\var\\data\\file1.txt", "D:\\Tensorflow\\MLbasics\\var\\data\\file2.txt",
             "D:\\Tensorflow\\MLbasics\\var\\data\\file3.txt", "D:\\Tensorflow\\MLbasics\\var\\data\\file4.txt"]
dataset = tf.data.Dataset.from_tensor_slices(filenames)

print(dataset)

dataset = dataset.interleave(lambda x:tf.data.TextLineDataset(x),
    cycle_length=4, block_length=16)

print(list(dataset.as_numpy_iterator()))


dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
# NOTE: New lines indicate "block" boundaries.
dataset = dataset.interleave(
    lambda x: dataset.from_tensors(x).repeat(6),
    cycle_length=2, block_length=4)
#print(list(dataset.as_numpy_iterator()))