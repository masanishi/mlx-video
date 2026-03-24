"""Attention module for LTX-2."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx_2.config import LTXRopeType
from mlx_video.models.ltx_2.rope import apply_rotary_emb


def _slice_rope_freqs(
    freqs_cis: Optional[Tuple[mx.array, mx.array]],
    start: int,
    end: int,
) -> Optional[Tuple[mx.array, mx.array]]:
    """Slice RoPE frequencies along the sequence dimension for query chunking."""
    if freqs_cis is None:
        return None

    cos_freqs, sin_freqs = freqs_cis

    if cos_freqs.ndim == 3:
        return cos_freqs[:, start:end, :], sin_freqs[:, start:end, :]
    if cos_freqs.ndim == 4:
        return cos_freqs[:, :, start:end, :], sin_freqs[:, :, start:end, :]

    raise ValueError(f"Unsupported RoPE tensor rank for chunk slicing: {cos_freqs.ndim}")


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    mask: Optional[mx.array] = None,
    query_chunk_size: Optional[int] = None,
) -> mx.array:

    b, q_seq_len, dim = q.shape
    _, kv_seq_len, _ = k.shape
    dim_head = dim // heads

    # Reshape to (B, seq_len, heads, dim_head)
    q = mx.reshape(q, (b, q_seq_len, heads, dim_head))
    k = mx.reshape(k, (b, kv_seq_len, heads, dim_head))
    v = mx.reshape(v, (b, kv_seq_len, heads, dim_head))

    # Transpose to (B, heads, seq_len, dim_head)
    q = mx.swapaxes(q, 1, 2)
    k = mx.swapaxes(k, 1, 2)
    v = mx.swapaxes(v, 1, 2)

    # Handle mask dimensions
    if mask is not None:
        # Add batch dimension if needed
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, axis=0)
        # Add heads dimension if needed
        if mask.ndim == 3:
            mask = mx.expand_dims(mask, axis=1)

    # Compute scaled dot-product attention
    scale = 1.0 / math.sqrt(dim_head)

    if query_chunk_size is not None and query_chunk_size > 0 and q_seq_len > query_chunk_size:
        outputs = []
        for start in range(0, q_seq_len, query_chunk_size):
            end = min(start + query_chunk_size, q_seq_len)
            q_chunk = q[:, :, start:end, :]

            mask_chunk = mask
            if mask is not None and mask.ndim >= 4 and mask.shape[-2] == q_seq_len:
                mask_chunk = mask[..., start:end, :]

            out_chunk = mx.fast.scaled_dot_product_attention(
                q_chunk, k, v, scale=scale, mask=mask_chunk
            )
            outputs.append(out_chunk)

        out = mx.concatenate(outputs, axis=2)
    else:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    # Reshape back to (B, q_seq_len, heads * dim_head)
    out = mx.swapaxes(out, 1, 2)
    out = mx.reshape(out, (b, q_seq_len, heads * dim_head))

    return out


class Attention(nn.Module):
    """Multi-head attention with rotary position embeddings.

    Supports both self-attention and cross-attention.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        has_gate_logits: bool = False,
    ):
        super().__init__()

        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head
        self.query_chunk_size: Optional[int] = None

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        # Q, K, V projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        # Q and K normalization
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

        # Output projection
        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

        # Per-head gating (LTX-2.3)
        if has_gate_logits:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        pe: Optional[Tuple[mx.array, mx.array]] = None,
        k_pe: Optional[Tuple[mx.array, mx.array]] = None,
        skip_attention: bool = False,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Query input of shape (B, seq_len, query_dim)
            context: Context for cross-attention. If None, uses x (self-attention)
            mask: Attention mask
            pe: Position embeddings for query (and key if k_pe is None)
            k_pe: Position embeddings for key (optional, uses pe if None)
            skip_attention: If True, bypass Q*K*V attention and use value projection
                only (for STG perturbation). Matches PyTorch all_perturbed=True.

        Returns:
            Attention output of shape (B, seq_len, query_dim)
        """
        context = x if context is None else context
        v = self.to_v(context)

        if skip_attention:
            gate = None
            if hasattr(self, "to_gate_logits"):
                gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))  # (B, seq, heads)
            # STG: bypass Q*K*V attention, use value projection only
            out = v
            if gate is not None:
                b, seq_len, _ = out.shape
                out = mx.reshape(out, (b, seq_len, self.heads, self.dim_head))
                out = out * gate[..., None]
                out = mx.reshape(out, (b, seq_len, -1))
            return self.to_out(out)
        elif (
            self.query_chunk_size is not None
            and self.query_chunk_size > 0
            and x.shape[1] > self.query_chunk_size
        ):
            k = self.to_k(context)
            k = self.k_norm(k)

            if pe is not None:
                k_pe_to_use = pe if k_pe is None else k_pe
                k = apply_rotary_emb(k, k_pe_to_use, self.rope_type)

            out_chunks = []
            for start in range(0, x.shape[1], self.query_chunk_size):
                end = min(start + self.query_chunk_size, x.shape[1])
                x_chunk = x[:, start:end, :]
                q_chunk = self.to_q(x_chunk)
                q_chunk = self.q_norm(q_chunk)

                if pe is not None:
                    q_chunk = apply_rotary_emb(
                        q_chunk,
                        _slice_rope_freqs(pe, start, end),
                        self.rope_type,
                    )

                mask_chunk = mask
                if mask is not None and mask.ndim >= 4 and mask.shape[-2] == x.shape[1]:
                    mask_chunk = mask[..., start:end, :]

                out_chunk = scaled_dot_product_attention(
                    q_chunk,
                    k,
                    v,
                    self.heads,
                    mask_chunk,
                )

                if hasattr(self, "to_gate_logits"):
                    gate_chunk = 2.0 * mx.sigmoid(self.to_gate_logits(x_chunk))
                    b, seq_len, _ = out_chunk.shape
                    out_chunk = mx.reshape(
                        out_chunk, (b, seq_len, self.heads, self.dim_head)
                    )
                    out_chunk = out_chunk * gate_chunk[..., None]
                    out_chunk = mx.reshape(out_chunk, (b, seq_len, -1))

                out_chunks.append(self.to_out(out_chunk))

            return mx.concatenate(out_chunks, axis=1)
        else:
            # Compute per-head gate early (from original input)
            gate = None
            if hasattr(self, "to_gate_logits"):
                gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))  # (B, seq, heads)

            # Standard attention
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k_pe_to_use = pe if k_pe is None else k_pe
                k = apply_rotary_emb(k, k_pe_to_use, self.rope_type)

            out = scaled_dot_product_attention(
                q,
                k,
                v,
                self.heads,
                mask,
                query_chunk_size=self.query_chunk_size,
            )

            # Apply per-head gating
            if gate is not None:
                b, seq_len, _ = out.shape
                out = mx.reshape(out, (b, seq_len, self.heads, self.dim_head))
                out = out * gate[..., None]
                out = mx.reshape(out, (b, seq_len, -1))

            # Project output
            return self.to_out(out)
