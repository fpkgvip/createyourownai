"""PyTorch fallback for the FlashAttention interface used by train.py."""

import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), softmax_scale=None):
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if q.size(1) != k.size(1):
        repeats = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    nq, nk = q.size(-2), k.size(-2)
    query_pos = torch.arange(nq, device=q.device)[:, None] + nk - nq
    key_pos = torch.arange(nk, device=q.device)[None, :]
    mask = torch.ones((nq, nk), dtype=torch.bool, device=q.device)
    if causal:
        mask &= key_pos <= query_pos
    left, right = window_size
    if left >= 0:
        mask &= key_pos >= query_pos - left
    if right >= 0:
        mask &= key_pos <= query_pos + right
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=softmax_scale,
    ).transpose(1, 2)
