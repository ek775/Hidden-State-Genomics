import torch
from hsg.sae.interleave import intervention_output

def extract_hidden_states(model, sequence:str, tokenizer, layer:int, device:str, return_tokens=False, return_logits=False) -> torch.Tensor:
    """
    Extracts the hidden states from the model for a given layer,
    
    (1) Pass a sequence through the model.
    (2) Strip the masked vectors from the hidden_states.
    (3) Arrange the resulting tensor in the shape (sequence_length//6-mer, hidden_size) for SAE training.

    Args:
    - model: A transformer model.
    - sequence: A string representing a DNA sequence.
    - tokenizer: The tokenizer associated with the model.
    - layer: The layer from which to extract hidden states.
    - device: The device on which to run the model.
    - return_tokens: A boolean indicating whether to return the initial sequence string as parsed by the tokenizer.
    - return_logits: A boolean indicating whether to return the final transformer output logits.

    Returns:
    - activations: A tensor representing the hidden states of the model.
    - (Optional) tokens: A list of tokens representing the initial sequence string after parsing.
    - (Optional) logits: A tensor representing the final transformer output logits.
    """
    tokenized_sequence = tokenizer.encode_plus(
        sequence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length
    )["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(tokenized_sequence.squeeze())

    mask = tokenized_sequence != tokenizer.pad_token_id

    tokenized_sequence = tokenized_sequence.to(device)
    mask = mask.to(device)

    logits, activations = intervention_output(
        model=model,
        tokens=tokenized_sequence,
        attention_mask=mask,
        patch_layer=layer,
        hidden_state_override=None
    )

    # remove batch dimension
    activations = activations.squeeze()

    # Remove padding from activations using the mask tensor
    activations = activations[mask.squeeze()]

    # configure results to return
    results = [activations]
    if return_tokens:
        results.append(tokens)
    if return_logits:
        results.append(logits)

    # return activations and optionally tokens and logits
    if len(results) > 1:
        return tuple(results)
    else:
        return activations