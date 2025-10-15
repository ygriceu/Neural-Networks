import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Constants ---
# NOTE: Do not change these.
BATCH_SIZE = 64
INPUT_SIZE = 4
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3

# --- Helper Functions ---
class CustomDataset(Dataset):
    """A custom dataset class for loading the training data."""
    def __init__(self, file_path):
        # We use numpy to load the data
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        
        # The last column is the target, the rest are features
        X = data[:, :-1]
        y = data[:, -1]
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Neural Network Definition ---
class TwoLayerNN(nn.Module):
    """A simple two-layer neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the TwoLayerNN model. This should be a two-layer network,
        meaning it has one hidden layer.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output classes.
        """
        super(TwoLayerNN, self).__init__()
        # --- STUDENT CODE: START ---
        # TODO: INSERT YOUR CODE HERE.
        self.decompose = nn.Flatten()
        self.initalization = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        



        
        # --- STUDENT CODE: END ---

        # referred to the website: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html,
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html,
        # https://ljvmiranda921.github.io/notebook/2017/02/17/artificial-neural-networks/,
        # https://medium.com/@hoangngbot/code-a-2-layer-neural-network-from-scratch-33d7db0f0e5f
    
    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the network.
        """        
        # --- STUDENT CODE: START ---
        # TODO: INSERT YOUR CODE HERE.
        x = self.decompose(x)
        result = self.initalization(x)
        return result
        # --- STUDENT CODE: END ---

# --- Training Function ---
def train_model(model, train_loader):
    """
    This function trains the model on the provided data.
    
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): The data loader for the training set.
    """
    # --- STUDENT CODE: START ---
    # TODO: INSERT YOUR CODE HERE.
    loss_function = nn.CrossEntropyLoss()
    num_turns = 192
    opt = torch.optim.Adam(model.parameters(),lr = 0.01)
    for i in range(num_turns):
        for x,y in train_loader:
            prediction = model(x)
            loss = loss_function(prediction,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # reffered to the website: https://www.appsilon.com/post/pytorch-neural-network-tutorial
    # https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html

    
    # --- STUDENT CODE: END ---

    
# --- Prediction Function ---
def predict(model, file_path):
    """
    This function makes predictions on a new dataset.

    This function is not used in this file but is provided in case you
    would like to validate the implementation and trained model. A similar
    function will be used by autograder to evaluate your model.
    
    Args:
        model (nn.Module): The trained model.
        file_path (str): Path to the CSV file for which to make predictions.
        
    Returns:
        np.ndarray: A numpy array of predicted labels.
    """
    # Load the data for prediction
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_true = data[:, -1].astype(int)
    # Set the model to evaluation mode
    model.eval()
    
    # Make and return predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        # Get the index of the max log-probability
        _, predicted_labels = torch.max(outputs.data, 1)
    y_pred = predicted_labels.numpy()
    print((y_pred == y_true).mean())
    return predicted_labels.numpy()
    
# --- Main Execution Block ---
if __name__ == '__main__':
    # Instantiate the dataset and dataloader
    train_dataset = CustomDataset('train.csv')
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataset loaded.")
    
    # Instantiate the model
    model = TwoLayerNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    print("Model initialized.")
    
    # Train the model
    train_model(model, train_loader)
    print("Model training completed.")
    predict(model,'train.csv')
    
    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")


## References
#Adam — PyTorch 2.8 Documentation. https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html, /generated/torch.optim.Adam.html. Accessed 14 Sept. 2025.
#Build the Neural Network — PyTorch Tutorials 2.8.0+cu128 Documentation. https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html, /beginner/basics/buildmodel_tutorial.html. Accessed 14 Sept. 2025.
#Linear — PyTorch 2.8 Documentation. https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html, /generated/torch.nn.Linear.html. Accessed 14 Sept. 2025.
#MIRANDA, LJ. “Implementing a Two-Layer Neural Network from Scratch.” Lj Miranda, 17 Feb. 2017, https://ljvmiranda921.github.io/notebook/2017/02/17/artificial-neural-networks/.
#Nguyen, Hoang. “Code a 2-Layer Neural Network from Scratch.” Medium, 9 Apr. 2024, https://medium.com/@hoangngbot/code-a-2-layer-neural-network-from-scratch-33d7db0f0e5f.
#PyTorch: How to Train and Optimize A Neural Network in 10 Minutes. https://www.appsilon.com/post/pytorch-neural-network-tutorial. Accessed 14 Sept. 2025.