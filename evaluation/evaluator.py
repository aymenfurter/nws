import torch
from tqdm import tqdm
from evaluation.metrics import perplexity, accuracy
from gpt2_training.gpt2_model import GPT2Model
from diffusion_model.diffusion_model import DiffusionModel
from gpt2_training.data_loader import get_data_loader

class Evaluator:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.gpt2_model = GPT2Model(config).to(device)
        self.diffusion_model = DiffusionModel(config).to(device)
        self.data_loader = get_data_loader(config)

    def load_models(self, gpt2_checkpoint, diffusion_checkpoint):
        gpt2_state_dict = torch.load(gpt2_checkpoint, map_location=self.device)
        self.gpt2_model.load_state_dict(gpt2_state_dict['model_state_dict'])
        
        diffusion_state_dict = torch.load(diffusion_checkpoint, map_location=self.device)
        self.diffusion_model.load_state_dict(diffusion_state_dict['model_state_dict'])

    def evaluate_gpt2(self, model=None):
        if model is None:
            model = self.gpt2_model
        model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        with torch.no_grad():
            for x, y in tqdm(self.data_loader, desc="Evaluating GPT-2"):
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = model(x, y)
                total_loss += loss.item()
                total_accuracy += accuracy(logits, y)
                num_batches += 1
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        ppl = perplexity(avg_loss)
        return {
            'loss': avg_loss,
            'perplexity': ppl,
            'accuracy': avg_accuracy
        }

    def evaluate_weight_generation(self, num_iterations=10):
        original_weights = self.gpt2_model.state_dict()
        performance_delta = []
        for _ in tqdm(range(num_iterations), desc="Evaluating weight generation"):
            # Generate new weights
            flattened_weights = self.diffusion_model.generate(shape=(1, self.config.total_params))
            new_weights = self.unflatten_weights(flattened_weights.squeeze())
            
            # Evaluate original weights
            original_performance = self.evaluate_gpt2()
            
            # Apply new weights and evaluate
            self.gpt2_model.load_state_dict(new_weights)
            new_performance = self.evaluate_gpt2()
            
            # Calculate performance delta
            delta = {
                'loss': original_performance['loss'] - new_performance['loss'],
                'perplexity': original_performance['perplexity'] - new_performance['perplexity'],
                'accuracy': new_performance['accuracy'] - original_performance['accuracy']
            }
            performance_delta.append(delta)
            
            # Restore original weights
            self.gpt2_model.load_state_dict(original_weights)
        
        # Calculate average performance delta
        avg_delta = {
            'loss': sum(d['loss'] for d in performance_delta) / num_iterations,
            'perplexity': sum(d['perplexity'] for d in performance_delta) / num_iterations,
            'accuracy': sum(d['accuracy'] for d in performance_delta) / num_iterations
        }
        return avg_delta

    def unflatten_weights(self, flattened_weights):
        unflattened = {}
        idx = 0
        for name, param in self.gpt2_model.named_parameters():
            unflattened[name] = flattened_weights[idx:idx+param.numel()].view(param.shape)
            idx += param.numel()
        return unflattened

    def evaluate_diffusion_model(self):
        pass

    def generate_and_evaluate_weights(self, num_samples=10):
        original_performance = self.evaluate_gpt2()
        generated_performances = []

        for _ in tqdm(range(num_samples), desc="Generating and evaluating weights"):
            # Generate new weights
            flattened_weights = self.diffusion_model.generate(shape=(1, self.config.total_params))
            new_weights = self.unflatten_weights(flattened_weights.squeeze())
            
            # Apply new weights and evaluate
            self.gpt2_model.load_state_dict(new_weights)
            new_performance = self.evaluate_gpt2()
            generated_performances.append(new_performance)

        # Compute average performance of generated weights
        avg_generated_performance = {
            'loss': sum(p['loss'] for p in generated_performances) / num_samples,
            'perplexity': sum(p['perplexity'] for p in generated_performances) / num_samples,
            'accuracy': sum(p['accuracy'] for p in generated_performances) / num_samples
        }

        return {
            'original_performance': original_performance,
            'avg_generated_performance': avg_generated_performance
        }