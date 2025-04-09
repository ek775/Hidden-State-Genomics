import torch
from hsg.pipelines.hidden_state import load_model
from hsg.sae.protocol.etl import extract_hidden_states
from hsg.sae.dictionary import AutoEncoder
from tqdm import tqdm
from umap import UMAP
import umap.plot as uplot
import plotly.express as px
import numpy as np

# built ins
import os, logging, sys

# Objects
class LatentModel(torch.nn.Module):
    """
    Package sae latents into a single model.
    """
    def __init__(self, parent_model, tokenizer, device, layer, sae):
        super().__init__()
        self.parent_model = parent_model
        self.tokenizer = tokenizer
        self.device = device
        # TODO: may be helpful to examine multiple layers at once
        self.layer = layer
        self.sae = sae
        # ensure everything is on the same device
        self.sae.to(self.device)
        if self.parent_model.device != self.device:
            self.parent_model.to(self.device)
    
    def forward(self, x: str, return_hidden_states=False, return_reconstructions=False, return_logits=False) -> torch.Tensor:
        """
        Forward pass of the model provides sae latents for a given layer + optional arguments: [hidden_states, reconstructions, logits]

        Where hidden_states are the hidden states of the parent model at the specified layer,
        reconstructions are the SAE reconstructions of the hidden states, and 
        logits are the final output of the parent model.
        """
        hidden_states, logits = extract_hidden_states(self.parent_model, x, self.tokenizer, self.layer, self.device, return_logits=True)
        reconstructions, latents = self.sae(hidden_states, output_features=True)
        
        results = [latents]
        if return_hidden_states:
            results.append(hidden_states)
        if return_reconstructions:
            results.append(reconstructions)
        if return_logits:
            results.append(logits)

        return results

# Functions
def get_latent_model(parent_model_path, layer_idx, sae_path) -> LatentModel:
    """
    Abstraction for easy loading of latent models
    """

    parent_model, tokenizer, device = load_model(parent_model_path)
    sae = AutoEncoder
    sae = AutoEncoder.from_pretrained(sae_path)
    latent_model = LatentModel
    latent_model = LatentModel(parent_model, tokenizer, device, layer_idx, sae)

    return latent_model

def generate_umap(embeddings: torch.Tensor, color: np.array, n_components:int = 2, **kwargs):
    """
    Generate a UMAP plot from a tensor of embeddings
    """
    umap = UMAP(n_components=n_components, **kwargs)
    umap_embeddings = umap.fit(embeddings)
    uplot.points(umap_embeddings, labels=color)

def interactive_umap(embeddings: torch.Tensor, color: np.array, n_components:int = 2, **kwargs):
    """
    Generate an interactive UMAP plot from a tensor of embeddings
    """
    umap = UMAP(n_components=n_components, **kwargs)
    umap_embeddings = umap.fit_transform(embeddings)
    fig = px.scatter(umap_embeddings, x=0, y=1, color=color, title=f"UMAP {kwargs}")
    fig.show()