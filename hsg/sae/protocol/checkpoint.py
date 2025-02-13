import torch
from hsg.sae.dictionary import AutoEncoder
import os

class History():
    """
    Object that stores the recent training history of a SAE model and enables checkpointing and early stopping.
    """
    def __init__(self, patience: int = 10, checkpoint_dir: str = "./checkpoints", layer: int = 0):
        self.history = {
            "train": [],
            "val": [],
            "epoch": []
        }
        self.layer = layer
        self.best_loss = float('inf')
        self.patience = patience
        self.counter = 0
        self.checkpoint_dir = checkpoint_dir
        self.early_stop = False

    def checkpoint(self, model, checkpoint_dir: str):
        """
        Save the model to disk.
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        torch.save(model.state_dict(), f"{checkpoint_dir}/best_model{self.layer}.pt")

    def reload_checkpoint(self, model, checkpoint_dir: str) -> AutoEncoder:
        """
        Reload the model from disk.
        """
        model.from_pretrained(f"{checkpoint_dir}/best_model{self.layer}.pt")

    def update(self, model, train_value, val_value, epoch) -> AutoEncoder:
        # add new epoch values to history
        self.history["train"].append(train_value)
        self.history["val"].append(val_value)
        self.history["epoch"].append(epoch)

        # check performance
        if val_value < self.best_loss:
            self.best_loss = val_value
            self.counter = 0
            self.checkpoint(model, self.checkpoint_dir)
        else:
            self.counter += 1

        if len(self.history["epoch"]) > self.patience:
            self.history["train"].pop(0)
            self.history["val"].pop(0)
            self.history["epoch"].pop(0)

        # reload best model if early stopping
        if self.counter >= self.patience:
            self.early_stop = True
            return self.reload_checkpoint(model, self.checkpoint_dir)
        else:
            return model