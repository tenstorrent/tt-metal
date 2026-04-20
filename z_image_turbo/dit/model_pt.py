# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch reference wrapper for ZImageTransformer2DModel.

Loads the model from HuggingFace, applies RoPE patching required for
TT-MLIR compatibility, and provides a clean forward() interface.
"""

import os
import sys

import torch
import torch.nn as nn

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
PATCH_SIZE = 2
F_PATCH_SIZE = 1

HEAD_DIM = 128
ORIGINAL_HEADS = 30
PADDED_HEADS = 32
EXTRA_DIM = (PADDED_HEADS - ORIGINAL_HEADS) * HEAD_DIM  # 256


def _patch_rope_for_tt():
    """Patch diffusers to use real-valued RoPE (no complex tensors) and
    XLA-compatible sequence prep, unpatchify, and cumsum."""
    from diffusers.models.transformers.transformer_z_image import (
        RopeEmbedder,
        ZSingleStreamAttnProcessor,
        ZImageTransformer2DModel,
    )
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    @staticmethod
    def _precompute_freqs_cis_real(dim, end, theta=256.0):
        result = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, dtype=torch.float64, device="cpu")
            freqs = torch.outer(timestep, freqs).float()
            result.append(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
        return result

    def _rope_call_real(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            result.append(self.freqs_cis[i][ids[:, i]])
        return torch.cat(result, dim=-2)

    def _attn_call_real(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.reshape(*query.shape[:-1], attn.heads, -1)
        key = key.reshape(*key.shape[:-1], attn.heads, -1)
        value = value.reshape(*value.shape[:-1], attn.heads, -1)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if freqs_cis is not None:

            def _real_rope(x_in, fc):
                x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
                cos = fc[..., 0].unsqueeze(2)
                sin = fc[..., 1].unsqueeze(2)
                x_r, x_i = x[..., 0], x[..., 1]
                out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1)
                return out.flatten(-2).type_as(x_in)

            query = _real_rope(query, freqs_cis)
            key = _real_rope(key, freqs_cis)
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output

    RopeEmbedder.precompute_freqs_cis = _precompute_freqs_cis_real
    RopeEmbedder.__call__ = _rope_call_real
    ZSingleStreamAttnProcessor.__call__ = _attn_call_real
    print("Applied real-valued RoPE patch (no complex tensors) for TT backend compatibility")

    from torch.nn.utils.rnn import pad_sequence as _pad_sequence

    def _prepare_sequence_xla(self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None):
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)
        feats_cat = torch.cat(feats, dim=0)
        combined_mask = torch.cat(inner_pad_mask)
        feats_cat = torch.where(combined_mask.unsqueeze(-1), pad_token.expand_as(feats_cat), feats_cat)
        feats = list(feats_cat.split(item_seqlens, dim=0))
        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))
        feats = _pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = _pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1
        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = _pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, : feats.shape[1]]
        return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

    ZImageTransformer2DModel._prepare_sequence = _prepare_sequence_xla
    print("Applied XLA-compatible _prepare_sequence patch (torch.where for boolean indexing)")

    def _unpatchify_xla(self, x, size, patch_size, f_patch_size, x_pos_offsets=None):
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        if x_pos_offsets is not None:
            return ZImageTransformer2DModel._unpatchify_original(self, x, size, patch_size, f_patch_size, x_pos_offsets)
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            t = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
            )
            x[i] = t.reshape(t.shape[0], t.shape[1] * t.shape[2], t.shape[3] * t.shape[4], t.shape[5] * t.shape[6])
        return x

    ZImageTransformer2DModel._unpatchify_original = ZImageTransformer2DModel.unpatchify
    ZImageTransformer2DModel.unpatchify = _unpatchify_xla
    print("Applied XLA-compatible unpatchify patch (tensor-shape reshape, single graph)")

    try:
        from transformers import masking_utils as _masking_utils

        _original_find_packed = _masking_utils.find_packed_sequence_indices

        def _find_packed_sequence_indices_tt(position_ids: torch.Tensor):
            first_dummy_value = position_ids[:, :1] - 1
            position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
            packed_sequence_mask = (position_diff != 1).to(torch.int32).cumsum(-1)
            from transformers.utils.import_utils import is_tracing

            if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
                return None
            return packed_sequence_mask

        _masking_utils.find_packed_sequence_indices = _find_packed_sequence_indices_tt
        print("Applied cumsum u8 fix (bool→int32 cast before cumsum in masking_utils)")
    except AttributeError:
        pass


def load_model():
    """Load ZImageTransformer2DModel from HuggingFace in bfloat16.

    Also patches the RoPE embedder to use real-valued operations instead of
    torch.polar / view_as_complex (not supported by TT-MLIR).

    Returns:
        transformer: ZImageTransformer2DModel in eval mode.
    """
    _patch_rope_for_tt()

    from diffusers import ZImageTransformer2DModel

    transformer = ZImageTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    transformer.eval()
    print(f"  Loaded transformer ({sum(p.numel() for p in transformer.parameters())/1e9:.2f}B params)")
    return transformer


def pad_heads(transformer):
    """Pad attention heads from 30 → 32 with zero-weight dummy heads (in place).

    Zero-weight dummy heads are mathematically transparent — they contribute
    zero to all attention outputs.  Padding is necessary so that 32 / 4 = 8
    heads per device divides evenly in the 4-way TP setup.
    """

    def _pad_layer(layer):
        attn = layer.attention
        in_dim = attn.to_q.weight.shape[1]
        for proj in (attn.to_q, attn.to_k, attn.to_v):
            w = proj.weight.data
            proj.weight = nn.Parameter(
                torch.cat([w, torch.zeros(EXTRA_DIM, in_dim, dtype=w.dtype)], dim=0),
                requires_grad=False,
            )
            if proj.bias is not None:
                b = proj.bias.data
                proj.bias = nn.Parameter(
                    torch.cat([b, torch.zeros(EXTRA_DIM, dtype=b.dtype)]),
                    requires_grad=False,
                )
        proj = attn.to_out[0]
        w = proj.weight.data
        proj.weight = nn.Parameter(
            torch.cat([w, torch.zeros(w.shape[0], EXTRA_DIM, dtype=w.dtype)], dim=1),
            requires_grad=False,
        )
        attn.heads = PADDED_HEADS

    all_layers = list(transformer.layers) + list(transformer.noise_refiner) + list(transformer.context_refiner)
    for layer in all_layers:
        _pad_layer(layer)

    print(f"  Head padding: {ORIGINAL_HEADS} → {PADDED_HEADS} heads")


def forward(transformer, latents, timestep, cap_feats, patch_size=PATCH_SIZE, f_patch_size=F_PATCH_SIZE):
    """Run a CPU forward pass through the transformer.

    Args:
        transformer: ZImageTransformer2DModel (must have patch_rope_for_tt applied).
        latents: List of [C, F, H, W] bfloat16 tensors (one per batch item).
        timestep: [1] float tensor (e.g. torch.tensor([0.5])).
        cap_feats: [seq_len, 2560] bfloat16 caption embeddings.
        patch_size: spatial patch size (default 2).
        f_patch_size: temporal patch size (default 1).

    Returns:
        List of output tensors, one per batch item.
    """
    with torch.no_grad():
        result = transformer(
            x=latents,
            t=timestep,
            cap_feats=cap_feats if isinstance(cap_feats, list) else [cap_feats],
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            return_dict=False,
        )
    outputs = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(outputs, list):
        outputs = [outputs]
    return outputs
