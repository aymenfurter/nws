import torch
from gpt2_training.gpt2_model import GPT2Model

class WeightApplier:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.gpt2_model = GPT2Model(config).to(device)

    def load_gpt2_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gpt2_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded GPT-2 model from {checkpoint_path}")

    def apply_weight_update(self, weight_update, learning_rate=1.0):
        current_weights = self.gpt2_model.state_dict()
        
        for name, param in current_weights.items():
            if name in weight_update:
                param.data += learning_rate * weight_update[name]
        
        self.gpt2_model.load_state_dict(current_weights)
        print(f"Applied weight update with learning rate {learning_rate}")

    def apply_weight_update_with_momentum(self, weight_update, momentum=0.9, learning_rate=1.0):
        if not hasattr(self, 'momentum_buffer'):
            self.momentum_buffer = {name: torch.zeros_like(param) for name, param in self.gpt2_model.named_parameters()}
        
        current_weights = self.gpt2_model.state_dict()
        
        for name, param in current_weights.items():
            if name in weight_update:
                self.momentum_buffer[name] = momentum * self.momentum_buffer[name] + (1 - momentum) * weight_update[name]
                param.data += learning_rate * self.momentum_buffer[name]
        
        self.gpt2_model.load_state_dict(current_weights)
        print(f"Applied weight update with momentum {momentum} and learning rate {learning_rate}")

    def apply_adaptive_weight_update(self, weight_update, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=1.0):
        if not hasattr(self, 'm'):
            self.m = {name: torch.zeros_like(param) for name, param in self.gpt2_model.named_parameters()}
            self.v = {name: torch.zeros_like(param) for name, param in self.gpt2_model.named_parameters()}
            self.t = 0
        
        self.t += 1
        current_weights = self.gpt2_model.state_dict()
        
        for name, param in current_weights.items():
            if name in weight_update:
                g = weight_update[name]
                self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
                self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)
                
                m_hat = self.m[name] / (1 - beta1 ** self.t)
                v_hat = self.v[name] / (1 - beta2 ** self.t)
                
                param.data += learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
        
        self.gpt2_model.load_state_dict(current_weights)
        print(f"Applied adaptive weight update with learning rate {learning_rate}")

    def get_updated_model(self):
        return self.gpt2_model

    def save_updated_model(self, save_path):
        torch.save({
            'model_state_dict': self.gpt2_model.state_dict(),
            'config': self.config,
        }, save_path)
        print(f"Saved updated GPT-2 model to {save_path}")