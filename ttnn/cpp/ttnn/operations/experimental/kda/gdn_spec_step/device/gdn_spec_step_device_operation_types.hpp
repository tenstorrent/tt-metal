// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Fused GDN spec-verify step (M2 stage 1): one dispatch per GDN layer replaces the composite verify chain
// (conv window rebuild, depthwise causal conv + SiLU, l2norms, gates, T-step delta rule on a per-token state ring,
// gated RMSNorm, silu(z) gate). One core per (user u, value head h); the state stays in L1 across the T tokens.
// T = 1 with ring := rec_state and an identity ctrl page is the seed step (same kernel, same arithmetic).
#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GdnSpecStepParams {
    uint32_t num_value_heads;
    uint32_t num_key_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t T;            // candidate tokens per user (rows u*T + t of qkvzab)
    uint32_t B;            // users
    uint32_t conv_kernel;  // K (4): window length Lw = K - 1 + T
    uint32_t qkvz_dim;     // column offset of the a|b block (= 2*Nk*Dk + 2*Nv*Dv)
    float scale;
    float l2_epsilon;
    float norm_epsilon;
    uint32_t hnew_depth;  // states buffered between compute and the ring writer (2 or 4)
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct GdnSpecStepInputs {
    Tensor
        qkvzab;  // [1, B*T <= R <= round_up(B*T,32), W >= qkvz_dim + 2*Nv] bf16 TILE, RAW rows u*T + t of [q|k|v|z|a|b]
    Tensor win_a;    // [B, Lw <= 32, C] bf16 TILE: conv window ping-pong pair; the op reads win[par] (rows 0..Lw-1 of
    Tensor win_b;    // user u = the previous window) and writes win[1 - par] (rows 0..Lw-1 = [E_prev[mi+1:mi+K]; new])
    Tensor ring;     // fp32 TILE, >= T*B*Nv blocks of [Dk, Dv]: block (t*B*Nv + u*Nv + h) = state after token t; the
                     // initial block per (u,h) comes from the ctrl page. rec_state [B, Nv, Dk, Dv] at T = 1 (seed)
    Tensor ctrl;     // uint32 ROW_MAJOR, one page [1, N >= 1 + B + B*Nv] (N*4 % 64 == 0): word 0 = window parity,
                     // words [1, 1+B) = mi[u], words [1+B, 1+B+B*Nv) = initial ring block per (u,h) or HOLD sentinel
                     // (HOLD must cover all Nv heads of a user; contents are data, not validated)
    Tensor taps;     // [1, K, C] bf16 TILE: conv taps, tap j in row j of tile column c (row-broadcast operand)
    Tensor dt_bias;  // [1, 1, Nv] fp32/bf16
    Tensor neg_exp_A;  // [1, 1, Nv] fp32/bf16
    Tensor weight;     // [Dv] bf16 gated-norm weight
};

}  // namespace ttnn::experimental::prim
