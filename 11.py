import pandas as pd
import tensorflow as tf


#Read data using pandas
SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

#Download the CSV file containing the heart disease dataset
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

#Read the CSV file using pandas
df = pd.read_csv(csv_file)

print(df.head())
print(df.dtypes)

#You will build models to predict the label contained in the target column
target = df.pop('target')

#Take the numeric features from the dataset (skip the categorical features for now)
numeric_feature_names = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]
print(numeric_features.head())

#To convert it to a tensor, use tf.convert_to_tensor
#array of lists
#data frame is a single tensor
tf.convert_to_tensor(numeric_features)

#The first step is to normalize the input ranges
#Use a tf.keras.layers.Normalization layer for that.
#To set the layer's mean and standard-deviation before running it be sure to call the Normalization.adapt method

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)

#Call the layer on the first three rows of the DataFrame to visualize an example of the output from this layer
normalizer(numeric_features.iloc[:3])

#Use the normalization layer as the first layer of a simple model
def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)


'''
#to apply transformations to each data item by iterating one by one
#data rame has to ave uniform data , data with same data type
#after preparing a each data item as feature values, label

numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

for row in numeric_dataset.take(3):
  print(row)

#shuffle and make batches of 2
numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)

model = get_basic_model()
model.fit(numeric_batches, epochs=15)

'''

#When you start dealing with heterogenous data, it is no longer possible to treat the DataFrame as if it were a single array since TensorFlow tensors require that all elements have the same dtype.

#So, in this case, you need to start treating it as a dictionary of columns, where each column has a uniform dtype. A DataFrame is a lot like a dictionary of arrays, so typically all you need to do is cast the DataFrame to a Python dict.
# Many important TensorFlow APIs support (nested-)dictionaries of arrays as inputs.
# So, to make a dataset of dictionary-examples from a DataFrame, just cast it to a dict before slicing it with Dataset.from_tensor_slices
numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))

#Here are the first three examples from that dataset
for row in numeric_dict_ds.take(3):
  print(row)

#Typically, Keras models and layers expect a single input tensor, but these classes can accept and return nested structures of dictionaries, tuples and tensors.
# These structures are known as "nests" (refer to the tf.nest module for details).

#There are two equivalent ways you can write a keras model that accepts a dictionary as input
'''
1. The Model-subclass style
You write a subclass of tf.keras.Model (or tf.keras.Layer). You directly handle the inputs, and create the outputs:
'''

def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)

class MyModel(tf.keras.Model):
  def __init__(self):
    # Create all the internal layers in init.
    super().__init__(self)

    self.normalizer = tf.keras.layers.Normalization(axis=-1)

    self.seq = tf.keras.Sequential([
      self.normalizer,
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(1)
    ])

  def adapt(self, inputs):
    # Stach the inputs and `adapt` the normalization layer.
    inputs = stack_dict(inputs)
    self.normalizer.adapt(inputs)

  def call(self, inputs):
    # Stack the inputs
    inputs = stack_dict(inputs)
    # Run them through all the layers.
    result = self.seq(inputs)

    return result


model = MyModel()

model.adapt(dict(numeric_features))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

#This model can accept either a dictionary of columns or a dataset of dictionary-elements for training
model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)
model.predict(dict(numeric_features.iloc[:3]))

'''
2. The Keras functional style
'''

inputs = {}
for name, column in numeric_features.items():
  inputs[name] = tf.keras.Input(
      shape=(1,), name=name, dtype=tf.float32)

inputs

x = stack_dict(inputs, fun=tf.concat)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

x = normalizer(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, x)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)

model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)
numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)

