// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// M0a SCRATCH prototype (not a product): row-batched T-loop recurrence for the GDN spec verify, one core per
// (user, value head). Measures the M2 fused op's cost model parameters (profiles/m2_fused_gdn_spec_op.md, 6.1).
#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GdnSpecTloopProtoParams {
    uint32_t num_value_heads;
    uint32_t num_key_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t T;         // candidate tokens per user (rows u*T + t of qkv)
    uint32_t B;         // users
    uint32_t qkvz_dim;  // column offset of the a|b block (= 2*Nk*Dk + 2*Nv*Dv)
    uint32_t s0_slot;   // initial state of (u,h) = ring block (s0_slot*B*Nv + u*Nv + h)
    float scale;
    float l2_epsilon;
    float norm_epsilon;
    bool row_batched;  // pre/post (l2norms, gates, kt, rmsnorm, output gate) once for the T rows; else per token
    bool write_ring;   // per-token fp32 state write to ring block (t*BH + bh); false = ablation (no writes)
    // JIT-only A/B switches: bit0 = reader two-phase (small reads pushed before the state read so PRE overlaps it);
    // bit1 = dual-pack S_next into hn and hnew (no 16 copy ops/token); bit2 = writer-only mode (compute just copies
    // state_in -> hnew T times: the pure ring-write stream); bit3 = output rows written one 32 B face-row at a time
    // (odd r0 / odd T allowed: the M2 spec 2.7 probe); bit4 = hnew 4 deep instead of 2 (write-overlap probe)
    uint32_t opt_flags;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct GdnSpecTloopProtoInputs {
    Tensor qkv;  // [1, R >= B*T, W >= qkvz_dim + 2*Nv] bf16 TILE, rows u*T + t of [q | k | v | z | a | b] (post-conv);
                 // a user's T rows must lie in one 32-row tile row (B*T <= 32 or 32 % T == 0)
    Tensor dt_bias;    // [1, 1, Nv] fp32/bf16
    Tensor neg_exp_A;  // [1, 1, Nv] fp32/bf16
    Tensor ring;       // [T*B*Nv, Dk, Dv] fp32 TILE: per-token state ring, updated in place
    Tensor weight;     // [Dv] bf16 gated-norm weight
};

}  // namespace ttnn::experimental::prim
