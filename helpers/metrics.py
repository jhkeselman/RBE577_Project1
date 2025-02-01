import numpy as np

def compute_mse(predictions, targets):
   """Compute Mean Squared Error between predictions and targets.
   
   Args:
       predictions (np.ndarray): Model predictions
       targets (np.ndarray): Target values
       
   Returns:
       float: MSE value
   """
   return np.mean((predictions - targets) ** 2)

def compute_rmse(predictions, targets): 
   """Compute Root Mean Squared Error between predictions and targets.
   
   Args:
       predictions (np.ndarray): Model predictions
       targets (np.ndarray): Target values
       
   Returns:
       float: RMSE value
   """
   return np.sqrt(compute_mse(predictions, targets))

def compute_mae(predictions, targets):
   """Compute Mean Absolute Error between predictions and targets.
   
   Args:
       predictions (np.ndarray): Model predictions  
       targets (np.ndarray): Target values
       
   Returns:
       float: MAE value
   """
   return np.mean(np.abs(predictions - targets))

def compute_position_error(predictions, targets):
   """Compute mean Euclidean error for position predictions (x,y,z).
   
   Args:
       predictions (np.ndarray): Model predictions with position in first 3 columns
       targets (np.ndarray): Target values with position in first 3 columns
       
   Returns:
       float: Mean position error
   """
   pred_pos = predictions[:, :3]
   target_pos = targets[:, :3]
   return np.mean(np.linalg.norm(pred_pos - target_pos, axis=1))

def compute_rotation_error(predictions, targets):
   """Compute mean Euclidean error for rotation predictions (rx,ry,rz).
   
   Args:
       predictions (np.ndarray): Model predictions with rotation in last 3 columns
       targets (np.ndarray): Target values with rotation in last 3 columns
       
   Returns:
       float: Mean rotation error
   """
   pred_rot = predictions[:, 3:]
   target_rot = targets[:, 3:]
   return np.mean(np.linalg.norm(pred_rot - target_rot, axis=1))