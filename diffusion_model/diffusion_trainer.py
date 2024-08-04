import torch
from tqdm import tqdm
from diffusion_model.diffusion_model import DiffusionModel
from diffusion_model.weight_dataset import WeightDataset

class DiffusionTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = DiffusionModel(config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.dataset = WeightDataset(config)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

    def train(self):
        for epoch in range(self.config.num_epochs):
            pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            for it, (x0, _) in pbar:
                x0 = x0.to(self.device)
                
                # Sample t uniformly
                t = torch.randint(0, self.config.num_timesteps, (x0.shape[0],), device=self.device)
                
                # Compute the noisy version of x0 at time t
                noise = torch.randn_like(x0)
                alphas_cumprod = self.get_alphas_cumprod(t)
                xt = self.q_sample(x0, t, noise, alphas_cumprod)
                
                # Predict the noise
                predicted_noise = self.model(xt, t)
                
                # Compute loss
                loss = F.mse_loss(predicted_noise, noise)
                
                # Backprop and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1)

    def get_alphas_cumprod(self, t):
        """Get the cumulative product of alphas for the given timesteps."""
        return torch.tensor([self.config.alphas_cumprod[t] for t in t], device=self.device)

    def q_sample(self, x0, t, noise, alphas_cumprod):
        """Sample from q(x_t | x_0)."""
        return (alphas_cumprod.sqrt()[:, None, None] * x0 +
                (1 - alphas_cumprod).sqrt()[:, None, None] * noise)

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'{self.config.checkpoint_dir}/diffusion_model_epoch_{epoch}.pt')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def generate(self, shape):
        self.model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(shape, device=self.device)
            
            for t in reversed(range(self.config.num_timesteps)):
                t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
                predicted_noise = self.model(x, t_batch)
                alpha = self.config.alphas[t]
                alpha_cumprod = self.config.alphas_cumprod[t]
                beta = 1 - alpha
                
                # Compute the mean
                x0_predicted = (x - beta.sqrt() * predicted_noise) / alpha.sqrt()
                model_mean = (
                    (alpha_cumprod.sqrt() * beta / (1 - alpha_cumprod)) * x0_predicted +
                    ((1 - alpha_cumprod) * alpha.sqrt() / (1 - alpha_cumprod)) * x
                )
                
                # Add noise, more noise if t > 0, else no noise
                noise = torch.randn_like(x) if t > 0 else 0
                x = model_mean + (0 if t == 0 else ((1 - alpha_cumprod) / (1 - alpha)).sqrt() * noise)
        
        return x