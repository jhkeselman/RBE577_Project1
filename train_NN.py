import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from helpers.data_transforms import StandardScaler, convert_to_tensor
from helpers.loss import CustomLoss
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error


class MLP(nn.Module):
   """Multi-Layer Perceptron for robot kinematics prediction.
   
   Args:
       input_size (int): Number of input features (joint angles)
       hidden_sizes (list): List of hidden layer sizes
       output_size (int): Number of outputs (position + rotation)
       
   Example:
       >>> model = MLP(input_size=6, hidden_sizes=[128, 64], output_size=6)
       >>> output = model(input_tensor)
   """

   def __init__(self, input_size, hidden_sizes, output_size):
       super(MLP, self).__init__()
       layers = []
       previous_size = input_size
       for size in hidden_sizes:
           layers.append(nn.Linear(previous_size, size))
           layers.append(nn.ReLU())
           previous_size = size
       layers.append(nn.Linear(previous_size, output_size))
       self.network = nn.Sequential(*layers)

   
   def forward(self, x):
       """Forward pass through network.
       
       Args:
           x (torch.Tensor): Input tensor of shape (batch_size, input_size)
           
       Returns:
           torch.Tensor: Output predictions of shape (batch_size, output_size)
       """
       return self.network(x)

def train_nn(X_train, X_test, y_train, y_test, hidden_sizes=[128, 64],
            lr=0.001, batch_size=32, epochs=100, device="cpu"):
   """Train neural network model for robot kinematics.
   
   Args:
       X_train, X_test (np.ndarray): Training and test features
       y_train, y_test (np.ndarray): Training and test targets 
       hidden_sizes (list): Hidden layer sizes
       lr (float): Learning rate
       batch_size (int): Mini-batch size
       epochs (int): Number of training epochs
       device (str): Device to train on ('cpu' or 'cuda')
       
   Returns:
       tuple: Trained model, input scaler, output scaler
       
   Example:
       >>> model, in_scaler, out_scaler = train_nn(X_train, X_test, y_train, y_test)
       >>> y_pred = model(X_test_tensor)
   """
   input_scaler = StandardScaler()
   output_scaler = StandardScaler()
   
   input_scaler.fit(X_train)  # Fit the scaler
   X_train = input_scaler.transform(X_train)  # Then transform the data

   output_scaler.fit(y_train)
   y_train = output_scaler.transform(y_train)

   X_test = input_scaler.transform(X_test)
   y_test = output_scaler.transform(y_test)

   X_train_tensor = convert_to_tensor(X_train, device)
   X_test_tensor = convert_to_tensor(X_test, device)

   y_train_tensor = convert_to_tensor(y_train, device)
   y_test_tensor = convert_to_tensor(y_test, device)

   train_data = TensorDataset(X_train_tensor, y_train_tensor)
   train_load = DataLoader(train_data, batch_size=batch_size, shuffle=True)

   model = MLP(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=y_train.shape[1]).to(device)
   criterion = CustomLoss(position_weight=1.0, rotation_weight=0.5)
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   
   for epoch in range(epochs):
       model.train()
       epoch_loss = 0

       for X_batch, y_batch in train_load:
           optimizer.zero_grad()
           y_pred = model(X_batch)
           loss = criterion(y_pred, y_batch)
           loss.backward()
           optimizer.step()
           epoch_loss += loss.item()

       model.eval()
       with torch.no_grad():
           test_y_pred = model(X_test_tensor)
           test_loss = criterion(test_y_pred, y_test_tensor).item()
       print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(train_load):.4f}, Test Loss: {test_loss:.4f}")
   

   torch.save(model.state_dict(), "NN.pth")
   return model, input_scaler, output_scaler

if __name__ == "__main__":
    from datasets import prepare_dataset

    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_dataset(
        "ur10dataset.csv"
    )

    # Train model
    model, input_scaler, output_scaler = train_nn(
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values,
        hidden_sizes=[128, 64],
        lr=0.001,
        epochs=100,
    )
