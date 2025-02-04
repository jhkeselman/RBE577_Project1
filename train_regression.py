import numpy as np
import torch
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
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            torch.save({"weights": self.weights}, "linear regression analytical.pth")
            return True
        except np.linalg.LinAlgError:
            print("Matrix is singular")
            return False
       
   def predict(self, X):
        """Make predictions for given input features.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted values of shape (n_samples, n_outputs) 
        """
        if self.weights is None:
            raise ValueError("Model has not been trained")
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)


if __name__ == "__main__":
    from datasets import prepare_dataset
    import matplotlib.pyplot as plt

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


    # PLOTTING ERROR FOR MULTIPLE SUBSET SIZES
    # X_test = X_test.values
    # y_test = y_test.values

    # subset_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]  # Different training set sizes
    # train_errors = []
    # test_errors = []

    # for size in subset_sizes:
    #     X_train_subset = X_train[:size]
    #     y_train_subset = y_train[:size]

    #     model = AnalyticalLinearRegression()
    #     success = model.fit(X_train, y_train)

    #     model.fit(X_train_subset, y_train_subset)
    #     y_pred_train = model.predict(X_train_subset)
    #     y_pred_test = model.predict(X_test)

    #     train_errors.append(compute_mse(y_pred_train, y_train_subset))
    #     test_errors.append(compute_mse(y_pred_test, y_test))

    # # Plot errors
    # plt.plot(subset_sizes, train_errors, label="Train Error", marker='o')
    # plt.plot(subset_sizes, test_errors, label="Test Error", marker='s')
    # plt.xlabel("Training Set Size")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.title("Train vs Test Error")
    # plt.show()
