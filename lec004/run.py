import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TAP30Dataset(Dataset):
    """
    A PyTorch Dataset for TAP30 demand data.

    This dataset loads demand data from a CSV file containing hourly demand
    records with spatial (row, col) and temporal (hour, day) features.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the TAP30 data.

    Attributes
    ----------
    data : pandas.DataFrame
        The raw data loaded from the CSV file.
    features : torch.Tensor
        Tensor of shape (n_samples, 4) containing the features
        [hour_of_day, day, row, col].
    targets : torch.Tensor
        Tensor of shape (n_samples,) containing the demand values.
    """

    def __init__(self, csv_path):
        # Read the CSV file
        self.data = pd.read_csv(csv_path)

        # Convert to tensors
        self.features = torch.tensor(
            self.data[["hour_of_day", "day", "row", "col"]].values, dtype=torch.float32
        )
        self.targets = torch.tensor(self.data["demand"].values, dtype=torch.float32)

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
                  [hour_of_day, day, row, col]
                - target (torch.Tensor): Scalar tensor containing the demand value
        """
        return self.features[idx], self.targets[idx]


if __name__ == "__main__":
    # Create dataset instance
    dataset = TAP30Dataset("artifacts/data/tap30/train.csv")

    # Print dataset size
    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    features, target = dataset[10]
    print(f"Sample features: {features}")
    print(f"Sample target: {target}")
