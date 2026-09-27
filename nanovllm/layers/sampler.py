import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy = temperatures <= 1e-10
        safe_temperatures = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits / safe_temperatures.unsqueeze(dim=1), dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(greedy, logits.argmax(dim=-1), sample_tokens)
