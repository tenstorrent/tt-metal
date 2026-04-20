# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Print ZImageTransformer2DModel architecture.

Usage:
  python z_image_turbo/print_arch.py
"""

import torch
from diffusers import ZImageTransformer2DModel

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

print(f"Loading transformer from {MODEL_ID}/transformer ...")
transformer = ZImageTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

print("\n" + "=" * 80)
print("ZImageTransformer2DModel Architecture")
print("=" * 80)
print(transformer)

print("\n" + "=" * 80)
print("Parameter counts by top-level module")
print("=" * 80)
for name, module in transformer.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name:<30} {params/1e6:>8.1f}M  ({type(module).__name__})")

total = sum(p.numel() for p in transformer.parameters())
print(f"\n  {'TOTAL':<30} {total/1e9:>8.2f}B")

print("\n" + "=" * 80)
print("Config")
print("=" * 80)
for k, v in transformer.config.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 80)
print("Single transformer layer (layers[0])")
print("=" * 80)
print(transformer.layers[0])

print("\n" + "=" * 80)
print("Attention head summary")
print("=" * 80)
attn = transformer.layers[0].attention
print(f"  heads:    {attn.heads}")
print(f"  to_q:     {attn.to_q.weight.shape}  (out, in)")
print(f"  to_k:     {attn.to_k.weight.shape}")
print(f"  to_v:     {attn.to_v.weight.shape}")
print(f"  to_out:   {attn.to_out[0].weight.shape}")
head_dim = attn.to_q.weight.shape[0] // attn.heads
print(f"  head_dim: {head_dim}")

print("\n" + "=" * 80)
print("FFN summary (SwiGLU)")
print("=" * 80)
ff = transformer.layers[0].feed_forward
print(f"  w1 (gate): {ff.w1.weight.shape}")
print(f"  w3 (up):   {ff.w3.weight.shape}")
print(f"  w2 (down): {ff.w2.weight.shape}")
