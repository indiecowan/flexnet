from torch.utils.data import Dataset
import pandas as pd
import torch

class pd1_dataset(Dataset):
    def __init__(self, csv_file):
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Drop the 'Patient Id' column
        self.data = data.drop(columns=['Patient Id'])
        
        # Define mapping for the label
        self.label_to_idx = {"Low": 0, "Medium": 1, "High": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        print(row)
        # Extract the features and label from the row
        features = torch.tensor(row[:-1].values, dtype=torch.float32) 
        label = row[-1]
        label = self.label_to_idx[label]

        return features, label
