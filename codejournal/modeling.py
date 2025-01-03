from .imports import *
from .config import ConfigBase
from .trainer import Trainer, TrainerArgs
from huggingface_hub import HfApi
from .slack import notify_info, notify_warn, notify_priority
    
class ModelUtils:
    """Utility functions for model-related operations."""

    @staticmethod
    def freeze_parameters(module):
        """Freeze all parameters of a module."""
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_parameters(module):
        """Unfreeze all parameters of a module."""
        for param in module.parameters():
            param.requires_grad = True

    @staticmethod
    def is_module_frozen(module):
        """Check if all parameters in a module are frozen."""
        for param in module.parameters():
            if param.requires_grad:
                return False
        return True

    @staticmethod
    def get_device(module):
        """Get the device of the module."""
        return next(module.parameters()).device

    @staticmethod
    def get_dtype(module):
        """Get the dtype of the module."""
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return None


    @staticmethod
    def get_parameter_count(module):
        """Count the total number of parameters in a module."""
        return sum(p.numel() for p in module.parameters())

    @staticmethod
    def get_trainable_parameter_count(module):
        """Count the number of trainable parameters in a module."""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    @staticmethod
    def save_model_and_config(model, path):
        """Save model weights and configuration."""
        from safetensors.torch import save_file, save_model
        os.makedirs(path, exist_ok=True)
        chk_path = os.path.join(path, "weights.safetensors")
        config_path = os.path.join(path, "config.json")

        # Save weights using safetensors
        # state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        # save_file(state_dict, chk_path)
        save_model(model, chk_path)

        # Save config as JSON
        model.config.to_json(config_path)

    @staticmethod
    def load_model_weights_from_safetensors(model, path):
        """Load model weights from safetensors."""
        if os.path.isdir(path):
            path = os.path.join(path, "weights.safetensors")
        from safetensors.torch import load_file, load_model
        # state_dict = load_file(path)
        # model.load_state_dict(state_dict)
        load_model(model, path)

    @staticmethod
    def push_model_to_hub(model, model_id, token=None, temp_dir="./temp/", private=True,commit_message="Push model.", push_kwargs=None):
        """Push model to Hugging Face Hub."""
        if token is None:
            token = os.environ.get("HUGGINGFACE_TOKEN")
        if token is None:
            raise ValueError("Token is required to push to the Hub.")
        if push_kwargs is None:
            push_kwargs = {}

        path = os.path.join(temp_dir, model_id)
        ModelUtils.save_model_and_config(model, path)

        api = HfApi(token=token)
        url = api.create_repo(repo_id=model_id, repo_type="model", exist_ok=True, private=private)
        api.upload_folder(repo_id=url.repo_id, folder_path=path, commit_message=commit_message, repo_type="model", **push_kwargs)
        shutil.rmtree(temp_dir)
        logger.info(f"Model pushed to the Hub: {url}")
        return url

    @staticmethod
    def load_model_from_hub(cls, model_id, token=None, device=None, config=None, dtype=None):
        """Load a model from Hugging Face Hub."""
        if token is None:
            token = os.environ.get("HUGGINGFACE_TOKEN")
        if token is None:
            raise ValueError("Token is required to load from the Hub.")
        api = HfApi(token=token)
        path = api.snapshot_download(repo_id=model_id)
        model = cls.from_pretrained(path, device=device, config=config, dtype=dtype)
        return model


class ModelBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    # Utility methods directly integrated
    freeze = ModelUtils.freeze_parameters
    unfreeze = ModelUtils.unfreeze_parameters
    isfrozen = ModelUtils.is_module_frozen
    get_device = ModelUtils.get_device
    get_dtype = ModelUtils.get_dtype
    get_parameter_count = ModelUtils.get_parameter_count
    get_trainable_parameter_count = ModelUtils.get_trainable_parameter_count

    device = property(ModelUtils.get_device)
    dtype = property(ModelUtils.get_dtype)
    parameter_count = property(ModelUtils.get_parameter_count)
    trainable_parameter_count = property(ModelUtils.get_trainable_parameter_count)
    
    save = ModelUtils.save_model_and_config
    load_from_safetensors = ModelUtils.load_model_weights_from_safetensors
    push_to_hub = ModelUtils.push_model_to_hub
    # load_from_hub = ModelUtils.load_model_from_hub

    @classmethod
    def load_from_hub(cls, model_id, token=None, device=None, config=None, dtype=None,private=True):
        """Load a model from Hugging Face Hub."""
        ModelUtils.load_model_from_hub(cls, model_id, token=token, device=device, config=config, dtype=dtype,private=private)

    @classmethod
    def from_pretrained(cls, path, device=None, config=None, dtype=None):
        """Load a model and its configuration from a safetensors checkpoint."""
        from safetensors.torch import load_file, load_model
        assert os.path.isdir(path), f'Path {path} is not a directory'
        
        # File paths
        chk_path = os.path.join(path, 'weights.safetensors')
        config_path = os.path.join(path, 'config.json')
        
        # Load state_dict using safetensors
        # state_dict = load_file(chk_path)
        
        # Optionally cast tensors to a specific dtype
        if dtype is not None:
            model.to(dtype)
        
        # Load configuration
        config = config or ConfigBase.from_json(config_path)
        
        # Initialize the model
        model = cls(config)
        load_model(model, chk_path)
        
        # Move the model to the specified device, if any
        if device is not None:
            model.to(device)
        return model

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
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
    
    def log(self,key, value):
        wandb.log({key: value})

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

__all__ = ["ModelBase", "ConfigBase", "Trainer", "TrainerArgs", "ModelUtils"]