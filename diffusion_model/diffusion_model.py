import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Define the U-Net architecture
        self.down_blocks = nn.ModuleList([
            DownBlock(config.channels[i], config.channels[i+1])
            for i in range(len(config.channels) - 1)
        ])
        
        self.up_blocks = nn.ModuleList([
            UpBlock(config.channels[i+1], config.channels[i])
            for i in reversed(range(len(config.channels) - 1))
        ])
        
        self.final_conv = nn.Conv1d(config.channels[0], 1, 1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t = self.time_embed(t)
        
        # Down sampling
        residuals = []
        for block in self.down_blocks:
            x = block(x, t)
            residuals.append(x)
        
        # Up sampling
        for block in self.up_blocks:
            residual = residuals.pop()
            x = block(torch.cat([x, residual], dim=1), t)
        
        return self.final_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(64, out_channels)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        h = h + self.time_mlp(t)[:, :, None]
        h = F.relu(self.conv2(h))
        return self.pool(h)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(64, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')

    def forward(self, x, t):
        h = self.upsample(x)
        h = F.relu(self.conv1(h))
        h = h + self.time_mlp(t)[:, :, None]
        h = F.relu(self.conv2(h))
        return h