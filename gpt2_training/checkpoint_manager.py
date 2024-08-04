import os
import torch
import json

class CheckpointManager:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, step):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'config': self.config.__dict__
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata for diffusion model training
        metadata = {
            'step': step,
            'checkpoint_path': checkpoint_path,
            # Add any other relevant metadata
        }
        metadata_path = os.path.join(self.checkpoint_dir, f'metadata_{step}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def load_checkpoint(self, model, optimizer):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        if not checkpoints:
            print("No checkpoints found. Starting from scratch.")
            return 0  # Starting step

        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']

        print(f"Loaded checkpoint from step {step}")
        return step

    def get_all_checkpoints(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        return sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))

    def checkpoint_to_training_data(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_state = checkpoint['model_state_dict']
        
        # Flatten the state dict into a single vector
        flattened_state = []
        for param in model_state.values():
            flattened_state.append(param.view(-1))
        flattened_state = torch.cat(flattened_state)
        
        return flattened_state.cpu().numpy()