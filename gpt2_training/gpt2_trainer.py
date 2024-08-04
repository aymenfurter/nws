import torch
from tqdm import tqdm
from gpt2_training.gpt2_model import GPT2Model
from gpt2_training.data_loader import get_data_loader, GPT2Dataset
from gpt2_training.checkpoint_manager import CheckpointManager

class GPT2Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = GPT2Model(config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        # Initialize data loader with a valid dataset
        self.data_loader = get_data_loader(config)  # Using the get_data_loader function
        
        self.checkpoint_manager = CheckpointManager(config)

    def train(self):
        self.model.train()

        # Access the length of the dataset, not the DataLoader
        dataset_length = len(self.data_loader.dataset)
        
        for epoch in range(self.config.num_epochs):
            # Use dataset length for progress bar
            pbar = tqdm(enumerate(self.data_loader), total=dataset_length)
            for it, (x, y) in pbar:
                # Place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward the model
                logits, loss = self.model(x, y)
                
                # Backprop and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                # Update progress bar
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")

                # Checkpoint if needed
                if it % self.config.checkpoint_interval == 0:
                    self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, epoch * dataset_length + it)

    def generate(self, idx, max_new_tokens):
        self.model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
