import numpy as np
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error


class AnalyticalLinearRegression:
   """Linear regression using closed-form analytical solution.
   
   Methods:
       fit(X, y): Compute weights using normal equation
       predict(X): Make predictions using learned weights
       
   Example:
       >>> model = AnalyticalLinearRegression()
       >>> success = model.fit(X_train, y_train)
       >>> if success:
       >>>     y_pred = model.predict(X_test)
   """

   def __init__(self):
       self.weights = None
   
   def fit(self, X, y):
       """Compute weights using normal equation.
       
       Args:
           X (np.ndarray): Training features of shape (n_samples, n_features)
           y (np.ndarray): Target values of shape (n_samples, n_outputs)
           
       Returns:
           bool: True if successful, False if matrix is singular
       """
       pass
       
   def predict(self, X):
       """Make predictions for given input features.
       
       Args:
           X (np.ndarray): Input features of shape (n_samples, n_features)
           
       Returns:
           np.ndarray: Predicted values of shape (n_samples, n_outputs) 
       """
       pass


if __name__ == "__main__":
    from datasets import prepare_dataset

    # Load data
    X_train, X_test, y_train, y_test = prepare_dataset(
        "ur10dataset.csv"
    )

    # Convert to numpy and take subset due to memory constraints
    subset_size = 10000
    X_train = X_train.values[:subset_size]
    y_train = y_train.values[:subset_size]
    X_test = X_test.values
    y_test = y_test.values

    # Train model
    model = AnalyticalLinearRegression()
    success = model.fit(X_train, y_train)

    if success:
        # Evaluate
        y_pred = model.predict(X_test)
        mse = compute_mse(y_pred, y_test)
        pos_error = compute_position_error(y_pred, y_test)
        rot_error = compute_rotation_error(y_pred, y_test)

        print(f"Test MSE: {mse:.4f}")
        print(f"Position Error: {pos_error:.4f}")
        print(f"Rotation Error: {rot_error:.4f}")
