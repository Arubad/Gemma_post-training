"""Chunked, checkpointed LM-head + cross-entropy (Gemma 4: 262k vocab, final logit softcap).

Full logits for one 16k sequence are 16k x 262k = 4.3e9 values (8.6 GB bf16, 17 GB fp32), so the
head is applied in chunks under activation checkpointing: only one chunk's logits are alive at
a time, in forward and in backward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor, softcap: float | None) -> torch.Tensor:
    logits = (h @ w.t()).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(logits, y, ignore_index=-100, reduction="sum")


def chunked_lm_loss(hidden: torch.Tensor, lm_head_weight: torch.Tensor, labels: torch.Tensor,
                    softcap: float | None = None, chunk_size: int = 2048) -> tuple[torch.Tensor, int]:
    """hidden [B,T,H] (final-normed), labels [B,T] already aligned to *predict* position t
    (i.e. caller shifts). Returns (sum_loss, n_label_tokens)."""
    h = hidden.reshape(-1, hidden.shape[-1])
    y = labels.reshape(-1)
    keep = y != -100
    n = int(keep.sum())
    if n == 0:
        return hidden.sum() * 0.0, 0
    h = h[keep]
    y = y[keep]
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], chunk_size):
        total = total + checkpoint(_chunk_loss, h[s : s + chunk_size], lm_head_weight, y[s : s + chunk_size],
                                   softcap, use_reentrant=False)
    return total, n


def shift_labels(input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """labels[t] = input_ids[t+1] where mask[t+1]==1 else -100 (predict-next alignment)."""
    lab = torch.full_like(input_ids, -100)
    lab[:, :-1] = torch.where(mask[:, 1:].bool(), input_ids[:, 1:], torch.full_like(input_ids[:, 1:], -100))
    return lab
