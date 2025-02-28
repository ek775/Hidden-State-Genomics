import torch
from hsg.sae.interleave import intervention_output

def extract_hidden_states(model, sequence:str, tokenizer, layer:int, device:str, return_logits=False) -> torch.Tensor:
    """
    Extracts the hidden states from the model for a given layer,
    
    (1) Pass a sequence through the model.
    (2) Strip the masked vectors from the hidden_states.
    (3) Arrange the resulting tensor in the shape (sequence_length//6-mer, hidden_size) for SAE training.
    """
    tokenized_sequence = tokenizer.encode_plus(
        sequence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length
    )["input_ids"]

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

    # optionally return logits
    if return_logits:
        return activations, logits
    else:
        return activations