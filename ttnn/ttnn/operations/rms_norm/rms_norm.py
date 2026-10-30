# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm: the starting stub of benchmark experiment E1 (generation). No kernels; every call is refused.

Specification
-------------
    rms_norm(x, gamma=None, epsilon=1e-6) = x / sqrt(mean(x**2, dim=-1) + epsilon) * gamma

computed over the last dimension of `x`, as a single-device TTNN op on Blackhole.

Inputs
    input_tensor  rank 2, 3 or 4; dtype float32, bfloat16 or bfloat8_b; TILE or ROW_MAJOR layout (bfloat8_b is
                  TILE only); any H and W, tile-aligned or not; DRAM interleaved, or L1 HEIGHT, WIDTH or BLOCK sharded.
    gamma         optional; shape [1, 1, 1, W]; its own dtype (float32, bfloat16, bfloat8_b) and layout (TILE,
                  ROW_MAJOR; bfloat8_b is TILE only).
    epsilon       added to the mean of squares before the reciprocal square root.
    compute_kernel_config
                  a ttnn.ComputeConfigDescriptor; None means default_compute_kernel_config(). The op honors its
                  math_fidelity, fp32_dest_acc_en and math_approx_mode.
    memory_config the output's placement; None means the input's. A sharded input's output keeps its shard spec.

Output
    Same shape, dtype and layout as the input.

Refusals
    A cell the op does not support raises NotImplementedError (UnsupportedAxisValue or ExcludedCell from
    ttnn.operations._op_contract). A refused cell is unsupported, not failed; any other exception, or an output
    outside the accuracy targets, is a failure.

Accuracy targets (tt_ops_code_gen bc965d1, eval/golden_tests/rms_norm/helpers.py TOLERANCES)
    PCC and RMS error relative to the reference's standard deviation, per input dtype:
    float32 (0.999, 0.02), bfloat16 (0.995, 0.04), bfloat8_b (0.99, 0.10). The reference is computed in float32.

Target cells
    cartesian(TARGET, INPUT_SHAPES) minus the suite's INVALID list (tt_ops_code_gen bc965d1,
    eval/golden_tests/rms_norm/feature_spec.py, sha256 0a0705d8de82...), plus the suite's loose, translated and
    regression tests. alignment and rank are derived from the input shape by INPUT_TAGGERS. The benchmark
    evaluator owns the support lists it scores with; SUPPORTED and EXCLUSIONS below are for the generator's own
    bookkeeping and are not read when scoring.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue

TILE_DIM = 32

TARGET = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
}

INPUT_SHAPES = [
    (1, 1, 32, 64), (1, 1, 64, 128), (4, 8, 32, 256), (2, 4, 128, 512), (1, 1, 2048, 256), (4, 1, 512, 512),
    (1, 1, 32, 50), (1, 1, 64, 17), (4, 8, 32, 47), (2, 1, 128, 100), (1, 1, 17, 64), (1, 1, 50, 128),
    (4, 8, 47, 256), (1, 1, 17, 50), (2, 1, 100, 47), (1, 1, 32, 4096), (1, 1, 32, 8192), (1, 1, 128, 4096),
    (2, 1, 64, 4096), (1, 32, 128), (4, 128, 512), (2, 512, 1024), (1, 32, 4096), (1, 32, 8192), (1, 32, 50),
    (4, 128, 47), (1, 17, 128), (32, 64), (128, 512), (1024, 1024), (32, 4096), (128, 8192), (32, 17), (128, 100),
    (17, 64),
]


def tag_alignment(inputs, axes):
    """tile_aligned (H%32==0 and W%32==0) / w_non_aligned (W%32!=0) / h_non_aligned (W aligned, H not)."""
    shape = inputs[0]
    H, W = shape[-2], shape[-1]
    if W % TILE_DIM != 0:
        return "w_non_aligned"
    if H % TILE_DIM != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {"alignment": tag_alignment, "rank": tag_rank}

SUPPORTED = {axis: [] for axis in TARGET}

EXCLUSIONS = []


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def validate(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.ComputeConfigDescriptor:
    raise UnsupportedAxisValue("rms_norm: not implemented")


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    validate(input_tensor, gamma=gamma, compute_kernel_config=compute_kernel_config, memory_config=memory_config)
