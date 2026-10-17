# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Repro: sparse_sdpa ignores math_approx_mode=False for softmax P."""

import torch
import ttnn


def run(device):
    heads, sequence, head_dim, value_dim = 32, 32, 64, 32
    q = torch.zeros((1, heads, sequence, head_dim), dtype=torch.bfloat16)
    kv = torch.zeros((1, 1, 32, head_dim), dtype=torch.bfloat16)
    indices = torch.full((1, 1, sequence, 32), -1, dtype=torch.int32)
    q[..., value_dim] = 1
    kv[0, 0, 1, value_dim] = 2
    kv[0, 0, 1, :value_dim] = 1
    indices[..., 0], indices[..., 1] = 0, 1

    def put(x, dtype):
        return ttnn.from_torch(
            x,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    output = ttnn.transformer.sparse_sdpa(
        put(q, ttnn.bfloat16),
        put(kv, ttnn.bfloat16),
        put(indices, ttnn.uint32),
        value_dim,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=0.5,
        k_chunk_size=32,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
    )
    expected = torch.sigmoid(torch.tensor(1.0)).item()
    actual = ttnn.to_torch(output).float()
    error = (actual - expected).abs().max().item()
    passed = error <= 0.005
    print(
        f"observed_min={actual.min().item():.7f} observed_max={actual.max().item():.7f} "
        f"expected={expected:.7f} max_error={error:.7f} {'PASS' if passed else 'FAIL'}"
    )
    return passed


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        raise SystemExit(0 if run(device) else 1)
    finally:
        ttnn.close_device(device)
