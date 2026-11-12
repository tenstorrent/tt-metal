// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Fused GDN speculative-verify step: one dispatch per GDN layer runs the conv window rebuild, the depthwise causal conv
// + SiLU, the l2norms, the gates, T gated delta-rule steps with the state after every token written out, the gated
// RMSNorm and the silu(z) gate. One core per (user u, value head h); the state stays in L1 across the T tokens.
#pragma once

#include <cstdint>
#include <optional>

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
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    // fast_mode: gdn_decode_step's fast_mode arithmetic. The T beta/decay gate pairs are computed in software fp32 on
    // the reader (gdn_step_gate_math.hpp) and handed to the compute as ready scalar tiles, instead of the SFPU
    // sigmoid/softplus/exp; the state add packs hnew from the same DST tile as hn (the full fp32 sum).
    bool fast_mode = false;
    // fp32_state (fast_mode only): gdn_decode_step's fp32_state arithmetic. The recurrent-state decay multiply and
    // the state add run on the SFPU in exact fp32 (state_in / dec / hd32 / hn32 are UnpackToDest DFBs) instead of the
    // FPU, whose SrcA path truncates the fp32 state every token.
    bool fp32_state = false;
};

struct GdnSpecStepInputs {
    // [1, B*T <= R <= round_up(B*T,32), W >= qkvz_dim + 2*Nv] bf16 TILE, raw rows u*T + t of [q|k|v|z|a|b]
    Tensor qkvzab;
    Tensor win_a;    // [B, Lw <= 32, C] bf16 TILE: conv window ping-pong pair; the op reads win[par] (rows 0..Lw-1 of
    Tensor win_b;    // user u = the previous window) and writes win[1 - par] (rows 0..Lw-1 = [E_prev[mi+1:mi+K]; new])
    Tensor ring;     // fp32 TILE, >= T*B*Nv blocks of [Dk, Dv]: block (t*B*Nv + u*Nv + h) = state after token t; the
                     // start block per (u,h) comes from the ctrl page
    Tensor ctrl;     // uint32 ROW_MAJOR, one page [1, N >= 1 + B + B*Nv] (N*4 % 64 == 0): word 0 = window parity,
                     // words [1, 1+B) = mi[u], words [1+B, 1+B+B*Nv) = start ring block per (u,h) or HOLD sentinel
                     // (HOLD must cover all Nv heads of a user; contents are data, not validated)
    Tensor taps;     // [1, K, C] bf16 TILE: conv taps, tap j in row j of tile column c (row-broadcast operand)
    Tensor dt_bias;  // [1, 1, Nv] fp32/bf16
    Tensor neg_exp_A;  // [1, 1, Nv] fp32/bf16
    Tensor weight;     // [Dv] bf16 gated-norm weight
    // optional: [>= B, T*Nv] ROW_MAJOR (u)int32. Entry (u, t*Nv + h) is the first ring page (KV = Dk*Dv/1024 fp32
    // tiles) of value head h's state after token t of user u, so the per-token states can go to arbitrary pages of a
    // pool. Absent: ring block (t*B*Nv + u*Nv + h). With it, the ctrl page's start word of (u, h) is the first fp32
    // ring page of that head's start state (any page) instead of a block index.
    std::optional<Tensor> ring_page_idx;
    // optional second fp32 pool (ring_page_idx only): ring pages >= ring's page count (in ring_page_idx and in the ctrl
    // start words) address ring_pool2 at (page - that count), so one user's heads can span two buffers.
    std::optional<Tensor> ring_pool2;
};

}  // namespace ttnn::experimental::prim
