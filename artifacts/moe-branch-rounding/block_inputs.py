"""Inputs whose real linear gate produces each prescribed expert distribution."""

import torch


def routed_inputs(tokens, distribution, seed=47, experts=128, topk=8):
    generator = torch.Generator().manual_seed(seed)
    if distribution == "uniform":
        ids = (torch.arange(tokens)[:, None] + torch.arange(topk)) % experts
    elif distribution == "hot":
        ids = torch.arange(topk).expand(tokens, topk)
    elif distribution == "multihot":
        ids = (
            (torch.arange(tokens) % 4)[:, None] * (3 * topk) + torch.arange(topk)
        ) % experts
    elif distribution == "longtail":
        scores = (
            torch.randn(tokens, experts, generator=generator)
            - torch.arange(experts).float().log1p()
        )
        ids = scores.topk(topk, dim=-1).indices
    else:
        raise ValueError(distribution)
    hidden = torch.randn(tokens, 2048, generator=generator, dtype=torch.bfloat16) * 0.02
    hidden[:, :experts] = 0
    # Power-of-two scaling preserves distinct BF16 scores at k=64 and k=256
    # while keeping the full softmax away from underflow ties.
    scores = torch.arange(topk, 0, -1, dtype=hidden.dtype) / max(1, topk // 8)
    hidden[:, :experts].scatter_(1, ids, scores.expand(tokens, topk))
    return hidden.unsqueeze(0), ids
