"""
Linear Regression using TensorFlow 2.x
=======================================
Fits a line y = Wx + b to randomly generated data
using gradient descent, and visualizes each iteration.

Usage:
  python linear_regession.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the number of points to generate
num_points = 1200

# Generate the data based on equation y = mx + c
data = []
m = 0.2
c = 0.5
for i in range(num_points):
    # Generate 'x' 
    x = np.random.normal(0.0, 0.8)

    # Generate some noise
    noise = np.random.normal(0.0, 0.04)

    # Compute 'y' 
    y = m*x + c + noise 

    data.append([x, y])

# Separate x and y
x_data = np.array([d[0] for d in data], dtype=np.float32)
y_data = np.array([d[1] for d in data], dtype=np.float32)

# Plot the generated data
plt.plot(x_data, y_data, 'ro')
plt.title('Input data')
plt.show()

# Generate weights and biases
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# Define the gradient descent optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.5)

# Start iterating
num_iterations = 10
for step in range(num_iterations):
    # Use GradientTape to compute gradients
    with tf.GradientTape() as tape:
        # Define equation for 'y'
        y_pred = W * x_data + b

        # Compute the loss (mean squared error)
        loss = tf.reduce_mean(tf.square(y_pred - y_data))

    # Compute and apply gradients
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

    # Print the progress
    print(f'\nITERATION {step + 1}')
    print(f'W = {W.numpy()[0]:.4f}')
    print(f'b = {b.numpy()[0]:.4f}')
    print(f'loss = {loss.numpy():.6f}')

    # Plot the input data 
    plt.plot(x_data, y_data, 'ro')

    # Plot the predicted output line
    plt.plot(x_data, W.numpy() * x_data + b.numpy())

    # Set plotting parameters
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title(f'Iteration {step + 1} of {num_iterations}')
    plt.show()
