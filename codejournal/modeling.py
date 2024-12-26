from .imports import *
from .config import ConfigBase
from .trainer import Trainer, TrainerArgs


class ModelBase(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    @property
    def device(self):
        return next(self.parameters()).device
    def to(self, device):
        return super().to(device)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def to_dtype(self, dtype):
        return super().to(dtype)
    
    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


    def save(self, path):
        """Save model and configuration to the specified path using safetensors."""
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        chk_path = os.path.join(path, 'weights.safetensors')
        config_path = os.path.join(path, 'config.json')
        
        # Move state_dict to CPU and save with safetensors
        cpu_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        save_file(cpu_state_dict, chk_path)
        
        # Save configuration as JSON
        self.config.to_json(config_path)
    
    def load_from_safetensors(self, path):
        """Load a model and its configuration from a safetensors checkpoint."""
        from safetensors.torch import load_file
        if os.path.isdir(path):
            path = os.path.join(path, 'weights.safetensors')
        # Load state_dict using safetensors
        state_dict = load_file(path)
        self.load_state_dict(state_dict)
        return self

    @classmethod
    def from_pretrained(cls, path, device=None, config=None, dtype=None):
        """Load a model and its configuration from a safetensors checkpoint."""
        from safetensors.torch import load_file
        assert os.path.isdir(path), f'Path {path} is not a directory'
        
        # File paths
        chk_path = os.path.join(path, 'weights.safetensors')
        config_path = os.path.join(path, 'config.json')
        
        # Load state_dict using safetensors
        state_dict = load_file(chk_path)
        
        # Optionally cast tensors to a specific dtype
        if dtype is not None:
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        
        # Load configuration
        config = config or Config.from_json(config_path)
        
        # Initialize the model
        model = cls(config)
        model.load_state_dict(state_dict)
        
        # Move the model to the specified device, if any
        if device is not None:
            model.to(device)
        return model

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def log(self,key, value):
        wandb.log({key: value})
    
    def get_trainable_state_dict(self):
        trainable_params = {
            name: param for name, param in self.named_parameters() if param.requires_grad
        }
        trainable_state_dict = {k: v for k, v in self.state_dict().items() if k in trainable_params}
        return trainable_state_dict
    
    def get_trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
    
    def get_trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_optimizer(self,trainer):
        module = getattr(torch.optim, trainer.args.optimizer)
        optimizer = module(self.get_trainable_parameters(), lr=trainer.args.lr, **trainer.args.optimizer_kwargs)
        return optimizer
    
    def get_scheduler(self, optimizer, trainer):
        if not trainer.args.scheduler:
            return None
        raise NotImplementedError

    def notify(self, message, level="info"):
        webhook = os.environ.get("SLACK_WEBHOOK_URL")
        if not webhook: 
            print(f"Slack webhook not found. Skipping notification.")
            return
        if webhook:
            if level == "info":
                notify_info(webhook, message)
            elif level == "warn":
                notify_warn(webhook, message)
            elif level == "priority":
                notify_priority(webhook, message)
            else:
                print(f"Invalid level: {level}. Skipping notification.")
                return


__all__ = ["ModelBase","ConfigBase","Trainer","TrainerArgs"]