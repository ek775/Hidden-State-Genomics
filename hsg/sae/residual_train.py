# load environment variables
from dotenv import load_dotenv
load_dotenv()

# built ins
import os, logging

# external libs
import torch
from torch.utils.tensorboard import SummaryWriter
from biocommons.seqrepo import SeqRepo

from tap import tapify

# hsg libs
from hsg.pipelines.hidden_state import load_model, get_unique_refseqs
from hsg.pipelines.variantmap import DNAVariantProcessor
from hsg.sae.dictionary import AutoEncoder
from hsg.sae.interleave import intervention_output


#####################################################################################################
# Main Functions
#####################################################################################################
def train_sae(parent_model, layer_idx: int, log_dir: str, expansion_factor: int, **kwargs) -> AutoEncoder:
    """
    Train a SAE on the output of a layer of a parent model.
    """
    train_log_writer = SummaryWriter(log_dir=os.path.join(log_dir, f"layer_{layer_idx}"))
    activation_dim = parent_model.esm.encoder.layer[layer_idx].output.dense.out_features
    num_latents = activation_dim * expansion_factor
    SAE = AutoEncoder(activation_dim=activation_dim, dict_size=num_latents)
    
    return SAE

def train_all_layers(
        parent_model: str, 
        SAE_directory: str, 
        log_dir: str,
        variant_data: str = os.environ["CLIN_VAR_CSV"],
        **kwargs
    ):
    """
    Command line interface to train a series of SAEs on the output of each layer of a parent model.

    Args:
        parent_model (str): The name of the parent model to be loaded from huggingface.
        model_directory (str): The directory to save the trained SAEs to.
    """

    model, tokenizer, device = load_model(parent_model)
    train_seqs = get_unique_refseqs(variant_data, DNAVariantProcessor())
    seq_repo = SeqRepo(os.environ["SEQREPO_PATH"])
    n_layers = len(model.esm.encoder.layer)

    # core training loop - may attempt multiprocessing later
    for layer in range(n_layers):
        sae = train_sae(parent_model=model, layer_idx=layer, log_dir=log_dir)
        torch.save(sae.state_dict(), os.path.join(SAE_directory, f"layer_{layer}.pt"))
    pass

if __name__ == "__main__":
    tapify(train_all_layers)