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
        targets = self.targets[idx]

        if self.transform:
            features = self.transform(features)

        return features, targets


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
        self.fc1 = nn.Linear(in_features, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 5)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(5, out_features)

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
            Output tensor of shape (batch_size, out_features) containing
            the predicted demand values.
        """
        hidden = self.fc1(features)
        hidden = self.relu1(hidden)
        hidden = self.fc2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.fc3(hidden)
        hidden = self.relu3(hidden)
        predictions = self.output(hidden)
        return predictions


if __name__ == "__main__":
    # Create dataset instance
    dataset = TAP30Dataset(
        "artifacts/data/tap30/train.csv", transform=normalize_features
    )

    # Create DataLoader instance
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    model = Tap30NN()
    print(model)

    for name, param in model.named_parameters():
        print(name, param.shape)

    y_pred = model(torch.Tensor([[0.1, 0.5, 0.33, 0.47]]))
    print(y_pred)
