import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TAP30Dataset(Dataset):
    """
    A PyTorch Dataset for TAP30 demand data.

    This dataset loads demand data from a CSV file containing hourly demand
    records with spatial (row, col) and temporal (hour, day) features.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the TAP30 data.
    transform : callable, optional
        A function/transform that takes a feature tensor and returns a transformed
        version. Default: None.

    Attributes
    ----------
    data : pandas.DataFrame
        The raw data loaded from the CSV file.
    features : torch.Tensor
        Tensor of shape (n_samples, 4) containing the features
        [hour_of_day, day, row, col].
    targets : torch.Tensor
        Tensor of shape (n_samples,) containing the demand values.
    transform : callable
        The transform to be applied to the features.
    """

    def __init__(self, csv_path, transform=None):
        # Read the CSV file
        self.data = pd.read_csv(csv_path)

        # Convert to tensors
        self.features = torch.tensor(
            self.data[["hour_of_day", "day", "row", "col"]].values, dtype=torch.float32
        )
        self.targets = torch.tensor(self.data["demand"].values, dtype=torch.float32)

        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to fetch.

        Returns
        -------
        tuple
            A tuple containing:
                - features (torch.Tensor): Tensor of shape (4,) containing
                  [hour_of_day, day, row, col]. If transform is set,
                  returns the transformed features.
                - target (torch.Tensor): Scalar tensor containing the demand value
        """
        features = self.features[idx]
        target = self.targets[idx]

        if self.transform:
            features = self.transform(features)

        return features, target


def normalize_features(x):
    x[0] /= 23.0  # hour_of_day
    x[1] /= 365.0  # day
    x[2] /= 7.0  # row
    x[3] /= 7.0  # col

    return x


class Tap30NN(nn.Module):
    """
    Neural network model for TAP30 demand prediction.

    A simple feedforward neural network with multiple fully connected layers
    and ReLU activations for predicting demand based on spatial and temporal features.

    Parameters
    ----------
    in_features : int, default=4
        Number of input features (hour_of_day, day, row, col).
    out_features : int, default=1
        Number of output features (demand prediction).
    """

    def __init__(self, in_features=4, out_features=1):
        super().__init__()
        self.ffblock = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, out_features),
        )

    def forward(self, features):
        """
        Forward pass through the neural network.

        Parameters
        ----------
        features : torch.Tensor
            Input tensor of shape (batch_size, in_features) containing
            the normalized feature values.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size,) containing
            the predicted demand values (squeezed).
        """
        predictions = self.ffblock(features)
        predictions = predictions.squeeze(-1)
        return predictions


def train_epoch(dataloader, model, loss_fn, optimizer):
    """
    Train the model for one epoch.

    This function performs one epoch of training by iterating through the dataloader,
    computing predictions, calculating loss, and updating model parameters.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the training dataset.
    model : torch.nn.Module
        The neural network model to train.
    loss_fn : callable
        Loss function to compute the error between predictions and targets.
    optimizer : torch.optim.Optimizer
        Optimizer used to update the model parameters.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    train_loss = 0

    model.train()
    for _, (features, target) in enumerate(dataloader):
        pred = model(features)
        loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss /= len(dataloader)
    rmse = train_loss**0.5
    print(f"Train avg loss: {rmse:>8f}")

    return train_loss


def evaluate(dataloader, model, loss_fn):
    """
    Evaluate the model on the provided dataloader.

    This function evaluates the model performance by iterating through the dataloader,
    computing predictions, and calculating the loss without updating model parameters.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the evaluation dataset.
    model : torch.nn.Module
        The neural network model to evaluate.
    loss_fn : callable
        Loss function to compute the error between predictions and targets.

    Returns
    -------
    float
        Average evaluation loss.
    """
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for _, (features, target) in enumerate(dataloader):
            pred = model(features)
            test_loss += loss_fn(pred, target).item()

    test_loss /= len(dataloader)
    rmse = test_loss**0.5
    print(f"Test avg loss: {rmse:>8f}")
    return test_loss


if __name__ == "__main__":
    train_dataset = TAP30Dataset(
        "artifacts/data/tap30/train.csv",
        transform=normalize_features,
    )

    validation_dataset = TAP30Dataset(
        "artifacts/data/tap30/validation.csv",
        transform=normalize_features,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
    )

    model = Tap30NN()
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)

    epochs = 5

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n--------------------------------------------------")
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        evaluate(validation_dataloader, model, loss_fn)

    hour_of_day = 18
    day = 110
    row = 3
    col = 2
    features = normalize_features(
        torch.tensor([hour_of_day, day, row, col], dtype=torch.float32)
    )
    y_pred = model(features)
    print(y_pred)
