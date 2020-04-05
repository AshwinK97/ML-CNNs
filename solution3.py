from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
import csv, copy, random
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

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
dataframe = pd.read_csv('./bank-full.csv')
dataframe.head()

# convert to numerical only data
dataframe['job'].replace(["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],range(12),inplace=True)
dataframe['marital'].replace(["married","divorced","single"],range(3),inplace=True)
dataframe['education'].replace(["unknown","secondary","primary","tertiary"],range(4),inplace=True)
dataframe['default'].replace(["yes","no"],range(2),inplace=True)
dataframe['housing'].replace(["yes","no"],range(2),inplace=True)
dataframe['loan'].replace(["yes","no"],range(2),inplace=True)
dataframe['contact'].replace(["unknown","telephone","cellular"],range(3),inplace=True)
dataframe['month'].replace(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],range(12),inplace=True)
dataframe['poutcome'].replace(["unknown","other","failure","success"],range(4),inplace=True)
dataframe['class'].replace(["yes","no"],range(2),inplace=True)

# split data into training, validation and testing sets
train, test = train_test_split(dataframe, test_size=13563)
print(len(train), 'training examples')
print(len(test), 'testing examples')

# Choose columns for feature layer
feature_columns = []
for header in ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]:
    feature_columns.append(feature_column.numeric_column(header))
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# convert datasets to TF dataset objects
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
test_ds = df_to_dataset(test, batch_size=batch_size)

# Create keras model
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile keras Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
model.fit(train_ds, epochs=10)

# Test the model
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)