from torch.utils.data import Dataset
import pandas as pd
import torch

class num_ds(Dataset):
    def __init__(self, csv_file):
        # Read the CSV file
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        print(row)
        # Extract the features and label from the row
        features = torch.tensor(row[:-1].values, dtype=torch.float32) 
        label = row[-1]

        return features, label
