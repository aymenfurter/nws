from evaluation import evaluator
import torch
from gpt2_training.gpt2_model import GPT2Model
from gpt2_training.gpt2_trainer import GPT2Trainer
from diffusion_model.diffusion_model import DiffusionModel
from diffusion_model.diffusion_trainer import DiffusionTrainer
from evaluation.evaluator import Evaluator
from generator.weight_generator import WeightGenerator
from generator.apply_weight import WeightApplier
from utils.config_manager import ConfigManager
from utils.logger import Logger
from utils.visualization import Visualizer
from types import SimpleNamespace
import numpy as np

def main():
    # Load configuration
    config = SimpleNamespace(
        # GPT-2 Configuration
        vocab_size=50257,         # Vocabulary size
        n_embd=768,               # Embedding dimension
        block_size=1024,          # Block size (sequence length)
        n_layer=12,               # Number of transformer layers
        n_head=12,                # Number of attention heads
        dropout=0.1,              # Dropout rate
        
        # Diffusion Model Configuration
        diffusion_channels=[32, 64, 128, 256], # Channels for diffusion model
        diffusion_time_embed_dim=256,          # Time embedding dimension for diffusion
        num_timesteps=1000,          # Number of timesteps in diffusion

        # Training Configuration
        batch_size=4,       # Batch size for GPT-2
        learning_rate=3e-5,  # Learning rate for GPT-2
        num_workers=4,       # Number of workers
        num_epochs=10,       # Number of epochs for GPT-2 training
        beta_start=0.0001,
        beta_end=0.02,

        channels=[32, 64, 128, 256], 
        diffusion_batch_size=64,  # Batch size for diffusion model
        diffusion_learning_rate=1e-4, # Learning rate for diffusion model
        diffusion_num_epochs=50,  # Number of epochs for diffusion training
        diffusion_num_timesteps=1000, # Number of timesteps in diffusion
        num_weight_updates=5,     # Number of weight updates

        # Data Configuration
        data_path="./data/input.txt",      # Path to data

        # Logging and Visualization
        log_dir="logs/",          # Directory for logs
        visualization_dir="visualizations/", # Directory for visualizations

        # Evaluation Configuration
        eval_batch_size=16,       # Batch size for evaluation
        num_eval_samples=1000,    # Number of evaluation samples

        # Weight Generation Configuration
        weight_generation_num_samples=10, # Number of samples for weight generation
        weight_generation_learning_rate=0.01, # Learning rate for weight generation

        # Miscellaneous
        seed=42,                  # Random seed
        device="cuda",            # Device to use for training
        
        # Additional Configuration
        grad_norm_clip=1.0,       # Gradient clipping norm
        checkpoint_interval=1000, # Interval for checkpoint saving
        checkpoint_dir='checkpoints/' # Directory for checkpoints
    )
    betas = np.linspace(config.beta_start, config.beta_end, config.diffusion_num_timesteps)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    # Calculate betas, alphas, and alphas_cumprod
    config.betas = np.linspace(config.beta_start, config.beta_end, config.diffusion_num_timesteps, dtype=np.float32)
    config.alphas = 1. - config.betas
    config.alphas_cumprod = np.cumprod(config.alphas, axis=0)

    # Ensure these are float32
    config.betas = config.betas.astype(np.float32)
    config.alphas = config.alphas.astype(np.float32)
    config.alphas_cumprod = config.alphas_cumprod.astype(np.float32)

    # Validate the configuration
    validate_config(config)

    # Set up logger
    logger = Logger(config.log_dir, 'main')

    # Set up visualizer
    visualizer = Visualizer(config.visualization_dir)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    logger.info(f"Using device: {device}")

    # Initialize GPT-2 model and trainer
    gpt2_model = GPT2Model(config).to(device)
    gpt2_trainer = GPT2Trainer(config, device)

    # Train GPT-2 model
    logger.info("Starting GPT-2 training")
    gpt2_trainer.train()
    logger.info("GPT-2 training completed")

    # Initialize Diffusion model and trainer
    diffusion_model = DiffusionModel(config).to(device)
    diffusion_trainer = DiffusionTrainer(config, device)

    # Train Diffusion model
    logger.info("Starting Diffusion model training")
    diffusion_trainer.train()
    logger.info("Diffusion model training completed")

    # Evaluate models
    evaluator = Evaluator(config, device)
    gpt2_metrics = evaluator.evaluate_gpt2(gpt2_model)
    logger.info(f"GPT-2 evaluation metrics: {gpt2_metrics}")

    # Generate and apply weight updates
    weight_generator = WeightGenerator(config, device)
    weight_applier = WeightApplier(config, device)

    for i in range(config.num_weight_updates):
        logger.info(f"Generating weight update {i+1}/{config.num_weight_updates}")
        weight_update = weight_generator.generate_weight_update(gpt2_model.state_dict())
        weight_applier.apply_weight_update(gpt2_model, weight_update)

        # Re-evaluate GPT-2 model
        updated_metrics = evaluator.evaluate_gpt2(gpt2_model)
        logger.info(f"Updated GPT-2 metrics after update {i+1}: {updated_metrics}")

        # Visualize results
        visualizer.plot_loss_curve(gpt2_trainer.losses, title=f'GPT-2 Training Loss (Update {i+1})')
        visualizer.plot_weight_distribution(weight_update['transformer.h.0.attn.c_attn.weight'], 
                                            title=f'Weight Update Distribution (Update {i+1})')

    logger.info("Training and evaluation completed")

def validate_config(config):
    required_attrs = ['vocab_size', 'n_embd', 'block_size', 'n_layer', 'n_head', 'dropout']
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise AttributeError(f"Configuration is missing required attribute: {attr}")

if __name__ == "__main__":
    main()
