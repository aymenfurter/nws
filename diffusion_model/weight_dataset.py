import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re

class WeightDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoints = self._load_checkpoints()
        self.max_seq_length = config.max_seq_length if hasattr(config, 'max_seq_length') else 10000
        print(f"Loaded {len(self.checkpoints)} checkpoints.")

    def _load_checkpoints(self):
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith('checkpoint_') and filename.endswith('.pt'):
                checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                flattened_weights = self._flatten_state_dict(checkpoint['model_state_dict'])
                
                step = self._extract_step_from_filename(filename)
                if step is not None:
                    checkpoints.append((flattened_weights, step))
                    print(f"Loaded checkpoint {filename} with shape {flattened_weights.shape}")
                else:
                    print(f"Skipping file {filename} as step number couldn't be extracted.")

        return sorted(checkpoints, key=lambda x: x[1])  # Sort by step number

    def _extract_step_from_filename(self, filename):
        match = re.search(r'checkpoint_(\d+)\.pt', filename)
        if match:
            return int(match.group(1))
        return None

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
        weight_diff = (weight_diff - np.mean(weight_diff)) / (np.std(weight_diff) + 1e-8)
        
        # Ensure the sequence length is within the limit
        if len(weight_diff) > self.max_seq_length:
            start_idx = np.random.randint(0, len(weight_diff) - self.max_seq_length)
            weight_diff = weight_diff[start_idx:start_idx + self.max_seq_length]
        
        # Reshape to 2D (add channel dimension)
        weight_diff = weight_diff.reshape(1, -1)
        
        return torch.FloatTensor(weight_diff), torch.FloatTensor([step_t1 - step_t0])

    def get_input_shape(self):
        # Return the shape of a single input item
        sample, _ = self[0]
        return sample.shape