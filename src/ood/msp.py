import torch


def msp_score(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1).values
