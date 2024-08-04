import yaml
from types import SimpleNamespace

class ConfigManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config_dict = yaml.safe_load(file)
        self.config = SimpleNamespace(**self.config_dict)

    def get_config(self):
        return self.config

    def update_config(self, key, value):
        setattr(self.config, key, value)
        self.config_dict[key] = value

    def save_config(self, path):
        with open(path, 'w') as file:
            yaml.dump(self.config_dict, file)

    def __getattr__(self, name):
        return getattr(self.config, name)