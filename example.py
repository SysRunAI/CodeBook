from codejournal.imports import *
from codejournal.modeling import *

import torchvision.models as models
from torchvision import datasets, transforms

# os.envor["SLACK_WEBHOOK_URL"] = ""

class Config(ConfigBase):
    resnet: int = 18
    pretrained: bool = True
    num_classes: int = 10

class ResNet(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.model = getattr(models, f'resnet{self.config.resnet}')(pretrained=self.config.pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.config.num_classes)
        
    def forward(self, x):
        x = x.repeat(1,3,1,1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / y.size(0)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / y.size(0)
        return {'loss': loss, 'acc': acc}
    
    def get_optimizer(self, trainer):
        # Override the default optimizer if needed
        return super().get_optimizer(trainer)
    
    def get_scheduler(self, optimizer, trainer):
        args = trainer.args # For configuring÷
        return super().get_scheduler(optimizer, trainer)


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

train_dataset = datasets.MNIST('./data', train=True, download=True,
                    transform=transform)
val_dataset = datasets.MNIST('./data', train=False, download=True,
                    transform=transform)

training_args = TrainerArgs(
    # Core Training Configuration
    batch_size=32,
    max_epochs=11,
    train_steps_per_epoch=1000,
    val_steps_per_epoch=500,
    grad_accumulation_steps=1,
    lr=1e-5,
    optimizer="AdamW",
    optimizer_kwargs={},
    scheduler=None,
    scheduler_kwargs={},

    # Logging and Checkpointing
    log_every_n_steps=32,
    save_every_n_steps=100,
    n_best_checkpoints=3,  # Negative value for saving all checkpoints
    n_latest_checkpoints=2,  # Negative value for saving all checkpoints
    checkpoint_metric="loss",
    checkpoint_metric_type="val",
    checkpoint_metric_minimize=True,

    # Hardware and Precision
    device="mps",  # Auto inferred
    mixed_precision=False,
    grad_clip_norm=1.0,  # False for no clipping
    num_workers=0,

    # WandB Integration
    wandb_project="Demo",  # wandb login
    wandb_run_name="MNIST",
    wandb_run_id=None,
    wandb_resume="allow",
    wandb_kwargs={},
    disable_wandb=False,

    # Resume and Debugging
    resume_from_checkpoint=True,  # None for no resuming, True for resuming latest checkpoint, or path to checkpoint
    debug_mode=True,

    # Miscellaneous
    safe_dataloader=True,
    log_grad_norm=True,
    slack_notify=True,
    results_dir="results",
    val_data_shuffle=False,
)

if __name__ == "__main__":
    config = Config()
    model = ResNet(config)

    trainer = Trainer(training_args)
    trainer.train(model, train_dataset, val_dataset)

__all__ = ["Config","ResNet","Trainer","TrainerArgs","train_dataset","val_dataset","training_args"]