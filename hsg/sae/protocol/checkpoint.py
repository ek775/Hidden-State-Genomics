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
        self.best_epoch = 0
        self.patience = patience
        self.counter = 0
        self.checkpoint_dir = checkpoint_dir
        self.early_stop = False

    def checkpoint(self, model):
        """
        Save the model to disk.
        """
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.best_epoch = self.history["epoch"][-1]
        self.best_loss = self.history["val"][-1]
        self.counter = 0

        torch.save(model.state_dict(), f"{self.checkpoint_dir}/best_model{self.layer}.pt")

    def best_metrics(self) -> dict:
        """
        Returns the best metrics of the model.
        """
        index = self.history["epoch"].index(self.best_epoch)
        metrics = {
            "train": self.history["train"][index],
            "val": self.history["train"][index],
            "epoch": self.history["epoch"][index]
        }

        return metrics

    def reload_checkpoint(self, model) -> AutoEncoder:
        """
        Reload the model from disk.
        """
        reloaded = model.from_pretrained(f"{self.checkpoint_dir}/best_model{self.layer}.pt")

        return reloaded

    def update(self, model, train_value, val_value, epoch) -> bool:
        # add new epoch values to history
        self.history["train"].append(train_value)
        self.history["val"].append(val_value)
        self.history["epoch"].append(epoch)

        # check performance
        if val_value < self.best_loss:
            self.checkpoint(model)
        else:
            self.counter += 1

        if len(self.history["epoch"]) > self.patience:
            self.history["train"].pop(0)
            self.history["val"].pop(0)
            self.history["epoch"].pop(0)

        # reload best model if early stopping
        if self.counter >= self.patience:
            self.early_stop = True
            return self.early_stop, 
        else:
            return self.early_stop