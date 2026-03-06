import torch


def energy_confidence_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # Using log-sum-exp as a confidence proxy: higher score indicates ID-like input.
    return temperature * torch.logsumexp(logits / temperature, dim=1)
