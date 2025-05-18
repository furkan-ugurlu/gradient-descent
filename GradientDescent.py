# ml/gradient_descent_numpy.py
import numpy as np

# Data dimensions
n_samples = 5
n_features = 4

# Random input, weights, and biases
x_list = np.random.uniform(0, 2, (n_samples, n_features))
w_list = np.random.uniform(0, 2, (n_samples, n_features))
b_list = np.random.uniform(0, 2, (n_samples, 1))

# Target: y = x·w + b (per sample)
target = np.sum(w_list * x_list, axis=1, keepdims=True) + b_list
target = np.round(target, 1)

# Training parameters
learning_rate = 0.01
epochs = 100
loss_list = []

for epoch in range(epochs):
    # Prediction
    y_pred = np.sum(w_list * x_list, axis=1, keepdims=True) + b_list

    # Loss calculation
    loss = np.mean((y_pred - target) ** 2)
    loss_list.append(loss)

    # Gradients
    d_loss = (y_pred - target) / n_samples
    d_w = d_loss * x_list
    d_b = d_loss

    # Parameter update
    w_list -= learning_rate * d_w
    b_list -= learning_rate * d_b

# Results
print("Target:\n", target)
print("Result:\n", np.sum(w_list * x_list, axis=1, keepdims=True) + b_list)
print("Distance:\n", np.abs(target - (np.sum(w_list * x_list, axis=1, keepdims=True) + b_list)))

# Loss output
loss_array = np.array(loss_list)
print("Loss:\n", loss_array.reshape(5, epochs // 5))
