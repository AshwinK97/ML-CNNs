from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
import csv, copy, random
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# convert dataframe to tf dataset
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('class')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

##################################### Main #####################################
dataframe = pd.read_csv('./irisdata.csv')
dataframe.head()

# split data into training, validation and testing sets
train, test = train_test_split(dataframe, test_size=0.3)
print(len(train), 'training examples')
print(len(test), 'testing examples')

# Choose columns for feature layer
feature_columns = []
for header in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    feature_columns.append(feature_column.numeric_column(header))
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# convert datasets to TF dataset objects
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
test_ds = df_to_dataset(test, batch_size=batch_size)

# Create keras model
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile keras Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
model.fit(train_ds, epochs=15)

# Test the model
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

# get confusion matrix
prediction = np.argmax(model.predict(df_to_dataset(dataframe, batch_size=batch_size)), axis=1)
matrix = confusion_matrix(dataframe['class'], prediction)
print(matrix)