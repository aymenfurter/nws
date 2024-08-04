import torch
from torch.utils.data import Dataset
import numpy as np
import os

class WeightDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = config.gpt2_checkpoint_dir
        self.checkpoints = self._load_checkpoints()

    def _load_checkpoints(self):
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pt'):
                checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                flattened_weights = self._flatten_state_dict(checkpoint['model_state_dict'])
                checkpoints.append((flattened_weights, int(filename.split('_')[1].split('.')[0])))
        return sorted(checkpoints, key=lambda x: x[1])  # Sort by step number

    def _flatten_state_dict(self, state_dict):
        flattened = []
        for param in state_dict.values():
            flattened.append(param.view(-1))
        return torch.cat(flattened).numpy()

    def __len__(self):
        return len(self.checkpoints) - 1  # We need pairs of checkpoints

    def __getitem__(self, idx):
        weights_t0, step_t0 = self.checkpoints[idx]
        weights_t1, step_t1 = self.checkpoints[idx + 1]
        
        # Compute the difference between checkpoints
        weight_diff = weights_t1 - weights_t0
        
        # Normalize the difference
        weight_diff = (weight_diff - np.mean(weight_diff)) / np.std(weight_diff)
        
        return torch.FloatTensor(weight_diff), torch.FloatTensor([step_t1 - step_t0])