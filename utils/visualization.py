import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def plot_loss_curve(self, losses, title='Training Loss'):
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'))
        plt.close()

    def plot_weight_distribution(self, weights, title='Weight Distribution'):
        plt.figure(figsize=(10, 6))
        sns.histplot(weights.flatten(), kde=True)
        plt.title(title)
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.save_dir, 'weight_distribution.png'))
        plt.close()

    def plot_attention_heatmap(self, attention_matrix, title='Attention Heatmap'):
        plt.figure(figsize=(12, 10))
        sns.heatmap(attention_matrix, cmap='viridis')
        plt.title(title)
        plt.xlabel('Token Position')
        plt.ylabel('Token Position')
        plt.savefig(os.path.join(self.save_dir, 'attention_heatmap.png'))
        plt.close()

    def plot_gradient_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
        plt.figure(figsize=(10, 8))
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("Average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'gradient_flow.png'), bbox_inches="tight")
        plt.close()