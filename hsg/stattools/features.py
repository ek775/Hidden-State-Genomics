import torch
from transformers.models.esm.modeling_esm import EsmForMaskedLM
from transformers.models.esm.tokenization_esm import EsmTokenizer
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
from pathlib import Path

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
    
    def forward(self, x: str, return_hidden_states=False, return_tokens=False, return_reconstructions=False, return_logits=False) -> torch.Tensor:
        """
        Forward pass of the model provides sae latents for a given layer + optional arguments: [hidden_states, reconstructions, logits]

        Where hidden_states are the hidden states of the parent model at the specified layer,
        reconstructions are the SAE reconstructions of the hidden states, and 
        logits are the final output of the parent model.

        Args:
        - x: A string representing a DNA sequence.
        - return_hidden_states: A boolean indicating whether to return the hidden states of the parent model.
        - return_tokens: A boolean indicating whether to return the initial sequence string as parsed by the tokenizer.
        - return_reconstructions: A boolean indicating whether to return the SAE reconstructions of the hidden states.
        - return_logits: A boolean indicating whether to return the final transformer output logits.

        Returns:
        - latents: A tensor representing the latent space of the SAE.
        [Optional]
        - hidden_states: A tensor representing the hidden states of the parent model.
        - tokens: A list of tokens representing the initial sequence string after parsing.
        - reconstructions: A tensor representing the SAE reconstructions of the hidden states.
        - logits: A tensor representing the final transformer output logits.

        If only latents are returned, the function will return a tensor.

        If one or more optional arguments are specified, the function will return all arguments as a tuple in the order specified.
        Items not requested will be skipped without altering the order.
        """
        hidden_states, tokens, logits = extract_hidden_states(
            self.parent_model, 
            x, self.tokenizer, 
            self.layer, 
            self.device, 
            return_tokens=True,
            return_logits=True)
        
        reconstructions, latents = self.sae(hidden_states, output_features=True)
        
        # configure results to return
        results = [latents]
        if return_hidden_states:
            results.append(hidden_states)
        if return_tokens:
            results.append(tokens)
        if return_reconstructions:
            results.append(reconstructions)
        if return_logits:
            results.append(logits)

        # unpack results or return latents only
        if len(results) > 1:
            return tuple(results)
        else:
            return latents

class FullLatentModel(torch.nn.Module):
    """
    Integrates multiple SAEs for each layer of the parent model to allow for feature extraction from all layers simultaneously.
    
    Has a larger memory footprint and slower inference time.
    """
    def __init__(self, saedir: Path, parent_model: EsmForMaskedLM, tokenizer: EsmTokenizer, device: str):
        """
        Args:
            saedir (Path): Path to the directory containing the SAEs.
            parent_model (EsmForMaskedLM): The parent model to use.
            tokenizer (EsmTokenizer): The tokenizer to use.
            device (str): The device to use for computation (e.g., "cuda" or "cpu").
        """
        super().__init__()
        self.parent_model = parent_model
        self.tokenizer = tokenizer
        self.device = device

        # load all SAEs
        self.saes = {}
        for layer in range(len(self.parent_model.esm.encoder.layer)):
            sae_path = os.path.join(saedir, f"layer_{layer}.pt")
            self.saes[str(layer)] = AutoEncoder.from_pretrained(sae_path)

        # use the gpu or cpu as appropriate
        self.to(self.device)
        self.parent_model.to(self.device)
        for sae in self.saes:
            self.saes[sae].to(self.device)

    def forward(self, x: str) -> dict:
        """
        Forward pass of the model provides the next token LM prediction for a given sequence,
        as well as the SAE latents for each layer and the original tokens for interpretability.
        Args:
            x (str): The input sequence.
        Returns:
            dict: A dictionary containing the logits, hidden states, tokens, and latents for each layer.

            Form of the dictionary:
            {
                "logits": torch.Tensor,
                "hidden_states": torch.Tensor,
                "tokens": list[str],
                "latents": {
                    "layer": torch.Tensor
                }
                "reconstructions": {
                    "layer": torch.Tensor
                }
            }
        """
        results = {}

        # tokenization & preprocessing
        token_embeddings = self.tokenizer.encode_plus(
            x,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(token_embeddings.squeeze())

        mask = token_embeddings != self.tokenizer.pad_token_id

        token_embeddings = token_embeddings.to(self.device)
        mask = mask.to(self.device)

        # get output of the parent model
        base_prediction = self.parent_model(
            token_embeddings,
            attention_mask=mask,
            encoder_attention_mask=mask,
            output_hidden_states=True
        )

        results["logits"] = base_prediction.logits.cpu()
        results["hidden_states"] = base_prediction.hidden_states.cpu()
        results["tokens"] = tokens
        results["latents"] = {}

        # get the SAE latents for each layer
        for sae in self.saes:
            hidden_states = base_prediction.hidden_states[int(sae)]
            reconstructions, features = self.saes[sae](hidden_states, output_features=True)
            results["latents"][sae] = features.cpu()
            results["reconstructions"][sae] = reconstructions.cpu()

        # All Done!
        return results


#################################################################################
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