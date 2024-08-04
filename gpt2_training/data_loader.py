import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the dataset class
class GPT2Dataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+1+self.block_size], dtype=torch.long)
        return x, y

# Define the data loader function
def get_data_loader(config):
    # Load the dataset
    data = np.memmap(config.data_path, dtype=np.uint16, mode='r')
    
    # Create the dataset and dataloader
    dataset = GPT2Dataset(data, config.block_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=True
    )
    
    return dataloader
