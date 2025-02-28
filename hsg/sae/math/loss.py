import torch

def mse_reconstruction_loss(x, x_hat, latents, current_l1_penalty):
    """
    Compute the MSE loss between the original and reconstructed sequences.
    """
    mse = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")

    l1_loss = latents.norm(p=1, dim=-1).mean()

    return mse + current_l1_penalty * l1_loss
