import torch

def train(model, data: torch.Tensor, optimizer, loss_fn, l1_penalty) -> float:
    """
    Executes training optimization step for a given batch
    """
    model.train()
    optimizer.zero_grad()

    logits, latents = model(data, output_features=True)

    loss = loss_fn(x=data, x_hat=logits, latents=latents, current_l1_penalty=l1_penalty)
    loss.backward()

    accuracy = torch.nn.functional.mse_loss(logits, data, reduction="mean")

    optimizer.step()

    return loss, accuracy


def validate(model, data: torch.Tensor, loss_fn, l1_penalty) -> float:
    """
    Runs the model in eval mode and returns the loss
    """
    with torch.no_grad():

        logits, latents = model(data, output_features=True)
        loss = loss_fn(x=data, x_hat=logits, latents=latents, current_l1_penalty=l1_penalty)

        accuracy = torch.nn.functional.mse_loss(logits, data, reduction="mean")

        return loss, accuracy

