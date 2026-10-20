// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Types of the fused KDA (Kimi Delta Attention) single-token decode block: causal conv (kernel 4) over
// the device-owned history + SiLU, per-head L2 norms, the per-key-channel decay / beta / output gates,
// the gated delta-rule recurrence on the device-owned state (in place) and the gated RMSNorm, for all
// users and heads in one program. See the compute kernel for the math.

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct KdaDecodeParams {
    uint32_t B;              // users (rows of the packed inputs)
    uint32_t H;              // heads (local to this device)
    uint32_t key_dim;        // K (multiple of 32)
    uint32_t val_dim;        // V (multiple of 32)
    float gate_lower_bound;  // g = lb * sigmoid(.)  (GLM/Kimi: -5)
    float l2_eps;            // l2norm eps inside the root (FLA: 1e-6)
    float norm_eps;          // gated RMSNorm eps
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// Shapes (all interleaved DRAM/L1; B rows are tile-padded with zeros):
//   qkv          [1, 1, B, 3*H*K]            bf16 TILE   raw projections [q | k | v], head-major inside a branch
//   gates        [1, 1, B, H*K + H*V + Hb]   fp32 TILE   [forget-gate pre-activation | output-gate pre-activation |
//   beta logits],
//                                                        Hb = H rounded up to 32
//   conv_state   [B, 3, 3*H*K]               bf16 RM     the three previous raw rows (oldest first); updated IN PLACE
//   recurrent    [B, H, K, V]                fp32 TILE   the recurrent state; updated IN PLACE
//   taps         [1, 1, 4, 3*H*K]            fp32 TILE   conv taps, row j = tap j (3 = the current token)
//   chan_consts  [1, 1, 2, H*K]              fp32 TILE   row 0 = dt_bias, row 1 = exp(A_log) per key channel
//   o_norm_w     [1, 1, 1, V]                fp32 TILE   gated RMSNorm weight
//   selectors    [33*32, 32]                 bf16 TILE   tile r < 32: 1 at (0, r); tile 32: all ones
// Output: o [1, 1, B, H*V] bf16 ROW_MAJOR (tilize it for o_proj); recurrent and conv_state are returned as-is
// (aliased).
struct KdaDecodeInputs {
    Tensor qkv;
    Tensor gates;
    Tensor conv_state;
    Tensor recurrent;
    Tensor taps;
    Tensor chan_consts;
    Tensor o_norm_w;
    Tensor selectors;
};

}  // namespace ttnn::prim
