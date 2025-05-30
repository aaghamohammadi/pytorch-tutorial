import numpy as np
import pandas as pd
import torch
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


if __name__ == "__main__":
    # Create dataset instance
    dataset = TAP30Dataset(
        "artifacts/data/tap30/train.csv", transform=normalize_features
    )

    # DataLoader parameters
    batch_size = 32  # Number of samples per batch
    shuffle = True  # Whether to shuffle data at each epoch

    # Create DataLoader instance
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Print information about the dataset and batches
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {batch_size}")

    # Demonstrate batch iteration
    print("\nIterating through first batch:")
    for batch_idx, (features, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Features shape: {features.shape}")  # [batch_size, 4]
        print(f"Targets shape: {targets.shape}")  # [batch_size]

        if batch_idx == 0:
            print("\nFirst batch features:")
            print(features)
            print("\nFirst batch targets:")
            print(targets)
            break
