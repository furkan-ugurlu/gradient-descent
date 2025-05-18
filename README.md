# Gradient Descent Demo 

This project demonstrates the basic logic of gradient descent using Numpy. The code randomly generates input data, weights, and biases, then tries to fit the weights and biases to match a target output using gradient descent.

## How It Works

- **Random Data Generation:**  
  Inputs (`x_list`), weights (`w_list`), and biases (`b_list`) are randomly initialized.
- **Target Calculation:**  
  The target output is calculated as:  
  **y = x · w + b**  
  where `x` and `w` are vectors, and `b` is a bias term.
- **Gradient Descent:**  
  The code iteratively updates the weights and biases to minimize the mean squared error (MSE) between the predicted output and the target.

## Mathematical Formulation

**Model Prediction:**
\[
\hat{y} = x \cdot w + b
\]

**Loss Function (Mean Squared Error):**
\[
L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
\]

**Gradients:**
- For weights:
\[
\frac{\partial L}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i
\]
- For bias:
\[
\frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
\]

> **Note:** In the code, the factor 2 is omitted for simplicity and absorbed into the learning rate.

## How to Run

1. Make sure you have Python and Numpy installed.
2. Save the script as `GradientDescent.py`.
3. Run the script:
    ```
    python GradientDescent.py
    ```

## Output

- The script prints the target values, the final predicted values, and the absolute difference between them.
- It also prints the loss values over epochs.

## Notes

- Each sample has its own weights and bias in this demo, which is different from classical linear regression (where all samples share the same weights and bias). This is for demonstration purposes.
- The code is intended for educational use to illustrate the steps of gradient descent.

---
