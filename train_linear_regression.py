import numpy as np
import torch
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error

class SGDLinearRegression:
   """Linear regression implementation using stochastic gradient descent optimization.
   
   Attributes:
       weights (np.ndarray): Model weights 
       bias (np.ndarray): Model bias
       lr (float): Learning rate for gradient descent
       
   Methods:
       fit(X, y): Train model using mini-batch SGD
       predict(X): Make predictions on new data
       
   Example:
       >>> model = SGDLinearRegression(learning_rate=0.01)
       >>> model.fit(X_train, y_train, batch_size=32, epochs=100)
       >>> y_pred = model.predict(X_test)
   """
   def __init__(self, learning_rate=0.01):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
   
   def _initialize_parameters(self, input_dim, output_dim):
       """Initialize model weights and bias.
       
       Args:
           input_dim (int): Number of input features
           output_dim (int): Number of output dimensions
       """
       self.weights = np.random.randn(input_dim, output_dim) * 0.01
       self.bias = np.zeros((1, output_dim))
       
   def _compute_loss(self, y_pred, y_true):
       """Compute MSE loss between predictions and targets.
       
       Args:
           y_pred (np.ndarray): Model predictions
           y_true (np.ndarray): Ground truth values
           
       Returns:
           float: MSE loss value
       """
       return compute_mse(y_pred, y_true)
       
   def _compute_gradients(self, X, y_true, y_pred):
       """Compute gradients for weights and bias.
       
       Args:
           X (np.ndarray): Input features
           y_true (np.ndarray): Ground truth values  
           y_pred (np.ndarray): Model predictions
           
       Returns:
           tuple: Weight gradients and bias gradients
       """
       n_samples = X.shape[0]

       dw = (2/n_samples) * X.T.dot(y_pred - y_true)
       db = (2/n_samples) * np.sum(y_pred - y_true, axis=0, keepdims=True)

       return dw, db
       
   def fit(self, X, y, batch_size=32, epochs=100):
       """Train model using mini-batch SGD.
       
       Args:
           X (np.ndarray): Training features of shape (n_samples, n_features)
           y (np.ndarray): Target values of shape (n_samples, n_outputs)
           batch_size (int): Mini-batch size for SGD
           epochs (int): Number of training epochs
       """
       n_samples, n_features = X.shape
       n_outputs = y.shape[1]

       self._initialize_parameters(n_features, n_outputs)
       for epoch in range(epochs):
           indices = np.random.permutation(n_samples)
           X_shuffled, y_shuffled = X[indices], y[indices]
           for i in range(0, n_samples, batch_size):
               X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
               y_pred = self.predict(X_batch)

               dw, db = self._compute_gradients(X_batch, y_batch, y_pred)

               self.weights -= self.lr * dw
               self.bias -= self.lr * db
        
           train_loss = self._compute_loss(self.predict(X), y)
           print(f"Train Loss: {train_loss:.4f}")

       torch.save({"weights": model.weights, "bias": model.bias}, "linear_regression.pth")
       
   def predict(self, X):
       """Make predictions for given input features.
       
       Args:
           X (np.ndarray): Input features of shape (n_samples, n_features)
           
       Returns:
           np.ndarray: Predicted values of shape (n_samples, n_outputs)
       """
       return X.dot(self.weights) + self.bias


if __name__ == "__main__":
    from datasets import prepare_dataset

    # Load data
    X_train, X_test, y_train, y_test = prepare_dataset(
        "ur10dataset.csv"
    )

    # Convert to numpy
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # Train model
    model = SGDLinearRegression(learning_rate=0.01)
    model.fit(X_train, y_train, batch_size=32, epochs=100)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = compute_mse(y_pred, y_test)
    pos_error = compute_position_error(y_pred, y_test)
    rot_error = compute_rotation_error(y_pred, y_test)

    print(f"Test MSE: {mse:.4f}")
    print(f"Position Error: {pos_error:.4f}")
    print(f"Rotation Error: {rot_error:.4f}")
