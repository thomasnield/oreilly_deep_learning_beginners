"""
In the Python code below, replace the question marks "?" 
with the proper code to perform a neural network prediction 
on a maintenance dataset on whether a part needs replacement (1) or not (0).

Use 3 nodes in the hidden layer, and ReLU as the activation function.

Experiment with learning rate and iterations to optimize training. 
"""

import pandas as pd
import tensorflow as tf

df = pd.read_csv('https://bit.ly/3wlFsb4')

# Extract input variables (all rows, all columns but last column)
# Note we should do some linear scaling here
X = df.values[:, :-1] / 1000.0

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Separate training and testing data

# declare the model 
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(?, activation=?),
  tf.keras.layers.Dense(?, activation=?)
])

loss_fn = tf.keras.losses.MeanSquaredError()

# compile the model 
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# fit the model
model.fit(?, ?, epochs=?, batch_size=?)


# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print(f"Dataset Score: {scores[1]}")
