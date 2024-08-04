import torch
import numpy as np
from tqdm import tqdm
from diffusion_model.diffusion_model import DiffusionModel

class WeightGenerator:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = DiffusionModel(config).to(device)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded diffusion model from {checkpoint_path}")

    def generate_weights(self, shape):
        self.model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(shape, device=self.device)
            
            for t in tqdm(reversed(range(self.config.num_timesteps)), desc="Generating weights"):
                t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
                predicted_noise = self.model(x, t_batch)
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = self.reverse_step(x, t, predicted_noise, noise)
        
        return x

    def reverse_step(self, x, t, predicted_noise, noise):
        alpha = self.config.alphas[t]
        alpha_prev = self.config.alphas[t-1] if t > 0 else torch.tensor(1.0)
        sigma = self.config.sigmas[t]
        
        pred_x0 = (x - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
        dir_xt = (1 - alpha_prev).sqrt() * predicted_noise
        
        return alpha_prev.sqrt() * pred_x0 + dir_xt + sigma * noise

    def generate_weight_update(self, current_weights, num_samples=10):
        flattened_weights = self.flatten_weights(current_weights)
        shape = (num_samples,) + flattened_weights.shape
        
        generated_updates = self.generate_weights(shape)
        
        # Average the generated updates
        avg_update = generated_updates.mean(dim=0)
        
        return self.unflatten_weights(avg_update, current_weights)

    def flatten_weights(self, weights):
        return torch.cat([p.view(-1) for p in weights.values()])

    def unflatten_weights(self, flattened, reference_weights):
        unflattened = {}
        idx = 0
        for name, param in reference_weights.items():
            unflattened[name] = flattened[idx:idx+param.numel()].view(param.shape)
            idx += param.numel()
        return unflattened