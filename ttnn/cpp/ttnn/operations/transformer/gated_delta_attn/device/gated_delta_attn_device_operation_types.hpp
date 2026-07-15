// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

// Attributes (non-tensor config) for the gated-delta-attention sequential scan.
struct GatedDeltaAttnSeqParams {
    uint32_t num_heads;   // BH — number of heads per device (e.g. 12)
    uint32_t num_chunks;  // number of sequence chunks (seq_len / chunk_size)
    uint32_t chunk_size;  // tokens per chunk (must be divisible by 32)
    uint32_t key_dim;     // Dk (e.g. 128)
    uint32_t val_dim;     // Dv (e.g. 128)
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // Token-major output. When true, output[0] is [B, seq_len, num_v_heads*Dv] token-major instead of the
    // [BH, NC, C, Dv] head-major layout.
    bool token_major_output = false;
    uint32_t num_v_heads = 0;  // H (value heads per batch); only used when token_major_output.
    uint32_t seq_len = 0;      // T (logical seq length, pre-chunk-pad); only used when token_major_output.
};

// Input tensors for the sequential scan kernel (Path A — Python pre-computes L_inv).
//
// All tensors are float32, TILE_LAYOUT, DRAM.
// Shape conventions (num_chunks axis is dim-1):
//   L_unit      : [BH, NC, C, C]   unit-diagonal lower-tri matrix (= D^{-1} L, where L = I + kk*mask)
//   v_beta_sc   : [BH, NC, C, Dv]  v_beta row-scaled by D^{-1}  (= D^{-1} @ v_beta)
//   k_bd_sc     : [BH, NC, C, Dk]  k_beta_decay row-scaled by D^{-1}
//   intra_attn  : [BH, NC, C, C]   intra-chunk attention matrix (precomputed: q@k.T * mask)
//   q_decay     : [BH, NC, C, Dk]  queries with cumulative decay
//   k_decay_t   : [BH, NC, Dk, C]  transposed keys with cumulative decay
//   dl_exp      : [BH, NC, 1, 1]   per-chunk state decay scalar (exp(sum_g))
//   L_inv       : [BH, NC, C, 32]  4 precomputed diagonal block inverses per chunk
//                 Tile i holds L^{-1}[i*32:(i+1)*32, 0:32] for block i in [0..Ct).
//   initial_state: [BH, Dk, Dv] or nullopt — recurrent state; zeros if absent
struct GatedDeltaAttnSeqInputs {
    Tensor L_unit;
    Tensor v_beta_sc;
    Tensor k_bd_sc;
    Tensor intra_attn;
    Tensor q_decay;
    Tensor k_decay_t;
    Tensor dl_exp;
    Tensor L_inv;
    std::optional<Tensor> initial_state;
};

}  // namespace ttnn::prim
