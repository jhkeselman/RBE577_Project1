import numpy as np

def dh_transform(theta, d, a, alpha):
   return np.array([
      [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
      [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
      [0, np.sin(alpha), np.cos(alpha), d],
      [0, 0, 0, 1]
    ])

def forward_kinematics(angles):
   # UR10 DH parameters from https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
   dh_params = [
        [0.1273, 0, np.pi/2],
        [0, -0.612, 0],
        [0, -0.5723, 0], 
        [0.163941, 0, np.pi/2],
        [0.1157, 0, -np.pi/2], 
        [0.0922, 0, 0]
    ]
   
   T = np.eye(4)
   for i in range(len(angles)):
        theta, d, a, alpha = angles[i], *dh_params[i]
        T = T @ dh_transform(theta, d, a, alpha)
   return T
   
def engineer_features(angles):
   """Engineer features for robot kinematics based on forward kinematics equations.
   
   Creates trigonometric features from joint angles that better capture the nonlinear 
   relationships in robot forward kinematics.
   
   
   Args:
       angles (np.ndarray): Input joint angles array of shape (n_samples, 6)
       
   Returns:
       np.ndarray: Engineered features array of shape (n_samples, 42)
       
   Example:
       >>> angles = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]) 
       >>> features = engineer_features(angles)
       >>> print(features.shape)
       (1, 42)
   """
   features = []
   for joints in angles:
       T = forward_kinematics(joints)
       pos = T[:3, 3]
       rot = T[:3, :3]
      
       rx = np.arctan2(rot[2, 1], rot[2, 2])
       ry = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1]**2 + rot[2, 2]**2))
       rz = np.arctan2(rot[1, 0], rot[0, 0])

       features.append(np.hstack([pos, [rx, ry, rz]]))
   return np.array(features)

if __name__ == "__main__":
   """
   Script to compare performance between raw angles vs engineered features:
   1. Load and preprocess data
   2. Train linear regression with raw joint angles
   3. Train linear regression with engineered trigonometric features  
   4. Compare MSE, position error and rotation error metrics
   """
   from datasets import prepare_dataset
   from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
   from train_regression import AnalyticalLinearRegression

   X_train_raw, X_test_raw, y_train, y_test = prepare_dataset("ur10dataset.csv")
   X_train_raw = X_train_raw.values
   X_test_raw = X_test_raw.values
   Y_train = y_train.values
   Y_test = y_test.values

   X_train_eng = engineer_features(X_train_raw)
   X_test_eng = engineer_features(X_test_raw)

   model_raw = AnalyticalLinearRegression()
   model_raw.fit(X_train_raw, Y_train)
   Y_pred_raw = model_raw.predict(X_test_raw)

   model_eng = AnalyticalLinearRegression()
   model_eng.fit(X_train_eng, Y_train)
   Y_pred_eng = model_eng.predict(X_test_eng)

   print("Performance with Raw Angles:")
   print(f"Test MSE: {compute_mse(Y_pred_raw, y_test):.4f}")
   print(f"Position Error: {compute_position_error(Y_pred_raw, Y_test):.4f}")
   print(f"Rotation Error: {compute_rotation_error(Y_pred_raw, Y_test):.4f}")

   print("\nPerformance with Engineered Features:")
   print(f"Test MSE: {compute_mse(Y_pred_eng, y_test):.4f}")
   print(f"Position Error: {compute_position_error(Y_pred_eng, Y_test):.4f}")
   print(f"Rotation Error: {compute_rotation_error(Y_pred_eng, Y_test):.4f}")
    