"""
COMPLETE THE CODE BELOW BY REPLACING THE QUESTION MARKS ?'s
SO FORWARD PROPAGATION IS COMPLETE 
"""

import numpy as np
import pandas as pd

all_data = pd.read_csv("https://bit.ly/3wlFsb4")

# Extract the input columns, scale down by 255
X = (all_data.iloc[:, 0:3].values / 1000.0)
Y = all_data.iloc[:, -1].values

# Build neural network with weights and biases
# with random initialization
w_hidden = np.random.rand(3, 3)
w_output = np.random.rand(1, 3)

b_hidden = np.random.rand(3, 1)
b_output = np.random.rand(1, 1)

# Activation functions
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))

# Runs inputs through the neural network to get predicted outputs
def forward_prop(X):
    Z1 = ? @ X + ?
    A1 = relu(?)
    Z2 = ? @ ? + b_output
    A2 = logistic(?)
    return Z1, A1, Z2, A2

# Calculate accuracy
test_predictions = forward_prop(X.transpose())[3]  # grab only A2
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y)
accuracy = sum(test_comparisons.astype(int) / X.shape[0])
print("ACCURACY: ", accuracy)

