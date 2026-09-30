// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct QkvCausalConv1dSiluParams {
    uint32_t sequence;
    uint32_t q_width;
    uint32_t k_width;
    uint32_t v_width;
    // ROW_MAJOR input: channels per work item. TILE input: 32 * B, where B is the tiled-path block size.
    uint32_t channel_chunk_size;
    // TILE input only: also return new_state = TILE [1,3,Q+K+V] that holds x[T-3..T-1].
    bool return_conv_state = false;
    // TILE input only: the pre-allocated new_state output (conv_state_out) shares its buffer with
    // history (in-place conv-state update). Then the core that reads a column block's history tiles
    // also writes that block's new_state, after its history read (reader define QKV_CONV_STATE_INPLACE).
    // Set by the launcher from the tensors; it is an attribute so that aliased and non-aliased calls
    // get separate program-cache entries (their tensor bindings resolve differently).
    bool conv_state_inplace = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // TILE input only, channel_chunk_size 128 (one 128-channel head per step), q/k widths multiples of 128.
    // Selects qkv_causal_conv1d_silu_tiled_fast.cpp (the 4 taps accumulate in dest; two-row interleaved TTI SiLU;
    // NOT bit-identical to the default kernel) with a q/k epilogue that L2-normalizes q and k per 128-channel head
    // and token (q also * 1/sqrt(128); eps 1e-6), as ChunkGdnFused's in-kernel QK norm. q and k are returned as
    // FLOAT32 TILE tensors; v and new_state stay bf16.
    bool fused_qk_l2_norm = false;
};

struct QkvCausalConv1dSiluInputs {
    Tensor input;
    // Required for ROW_MAJOR input. Optional for TILE input: std::nullopt means three zero rows.
    std::optional<Tensor> history;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor tap3;
    // TILE input with return_conv_state only: pre-allocated new_state output (TILE bf16 interleaved
    // [1,3,Q+K+V]). std::nullopt allocates a new DRAM tensor. It may be the history tensor itself.
    std::optional<Tensor> conv_state_out;
};

}  // namespace ttnn::experimental::prim
