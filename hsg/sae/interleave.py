"""Adapted from E. Simon: https://github.com/ElanaPearl/InterPLM/blob/main/interplm/sae/intervention.py"""


import nnsight
from nnsight import NNsight
import torch
from transformers.models.esm.modeling_esm import EsmForMaskedLM


def get_submodule(model: NNsight, submodule_index: int):
    """
    Get a submodule from Nucleotide Transformer via index position. Due to NTv2 containing multiple prediciton heads, some of the outputs are mapped
    to predefined outputs that nnsight has labelled with .nns_output, however, using this attribute has proven difficult. Instead, I have opted to
    follow E. Simon's approach and patch the following layer's input (i+1) which is equivalent.

    Args:
        model: The model from which to get the submodule.
        submodule_index: The index of the submodule to get.

    Returns:
        (1) The submodule. (2) access method (input or output): str
    """
    n_layers = len(model.esm.encoder.layer)

    if submodule_index > n_layers:
        raise ValueError("Submodule index out of range.")
    
    elif submodule_index == n_layers:
        return model.esm.encoder.emb_layer_norm_after, "output"
    
    else:
        return model.esm.encoder.layer[submodule_index], "input"
    

def intervention_output(model: EsmForMaskedLM, tokens: torch.Tensor, attention_mask: torch.Tensor, patch_layer: int, 
                        hidden_state_override: torch.Tensor | None = None):
    """
    Get the output of a NTv2 model with optional activation clamping for cause-effect analysis.

    Args:
        model: The model.
        tokens: Input tokens from associated tokenizer.
        attention_mask: Attention mask for the input tokens.
        patch_layer: The layer to patch.
        hidden_state_override: A vector to override the hidden state activations with. (Optional)

    If hidden_state_override is not provided, original activations and output are returned.

    Returns:
        (1) The model output. (2) The hidden state activation at layer[i].
    """
    with torch.no_grad():
        output = model(
            tokens,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )

        # return vanilla output
        if hidden_state_override is None:
            return output.logits, output.hidden_states[patch_layer]
        
        # patch the hidden state activations
        nnsight_model = NNsight(model)
        submodule, input_or_output = get_submodule(nnsight_model, patch_layer)

        with nnsight_model.trace(tokens, attention_mask=attention_mask, encoder_attention_mask=attention_mask) as tracer:
            patch = (submodule.input[0][0] if input_or_output == "input" else submodule.output)
            patch[:] = hidden_state_override
            patch_results = nnsight_model.output.save() 

        return patch_results, output.hidden_states[patch_layer]