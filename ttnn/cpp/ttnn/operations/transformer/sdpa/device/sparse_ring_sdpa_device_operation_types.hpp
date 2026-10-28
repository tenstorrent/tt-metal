// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <utility>

#include <tt-metalium/constants.hpp>

#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_fusion.hpp"

namespace ttnn::prim {

struct SparseRingSDPAParams {
    std::optional<float> scale;
    std::size_t logical_n = 0;
    std::size_t ring_size = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    experimental::prim::RingAttentionAllGatherAsyncParams all_gather_operation_attributes;
    experimental::prim::RingAttentionAllGatherAsyncInputs all_gather_tensor_args;
    CoreCoord ccl_core_grid_offset;
    // Square block of the mask, in tokens: one mask row per Q block of block_size rows, one mask column per K block
    // of block_size tokens. A compile-time parameter of the kernels; only 64 is supported for now.
    uint32_t block_size = 64;
    // Q blocks per core per pass (resident slots). nullopt: the largest count whose CBs fit L1. Each core's blocks
    // are split into the same number of passes; the K/V all-gather runs once.
    std::optional<uint32_t> q_blocks_per_pass = std::nullopt;

    // We need a constructor, because all_gather_struct is not default initializable.
    SparseRingSDPAParams(
        std::optional<float> scale,
        std::size_t logical_n,
        std::size_t ring_size,
        tt::tt_metal::MemoryConfig output_memory_config,
        std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
        DeviceComputeKernelConfig compute_kernel_config,
        experimental::prim::RingAttentionAllGatherAsyncParams all_gather_operation_attributes,
        experimental::prim::RingAttentionAllGatherAsyncInputs all_gather_tensor_args,
        CoreCoord ccl_core_grid_offset,
        uint32_t block_size,
        std::optional<uint32_t> q_blocks_per_pass) :
        scale(scale),
        logical_n(logical_n),
        ring_size(ring_size),
        output_memory_config(std::move(output_memory_config)),
        program_config(std::move(program_config)),
        compute_kernel_config(compute_kernel_config),
        all_gather_operation_attributes(std::move(all_gather_operation_attributes)),
        all_gather_tensor_args(std::move(all_gather_tensor_args)),
        ccl_core_grid_offset(ccl_core_grid_offset),
        block_size(block_size),
        q_blocks_per_pass(q_blocks_per_pass) {}

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }

    std::uint32_t block_tiles() const { return block_size / tt::constants::TILE_HEIGHT; }

    static constexpr auto attribute_names = std::forward_as_tuple(
        "scale",
        "logical_n",
        "ring_size",
        "compute_kernel_config",
        "program_config",
        "ccl_core_grid_offset",
        "all_gather_operation_attributes",
        "all_gather_tensor_args",
        "block_size",
        "q_blocks_per_pass");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(scale),
            std::cref(logical_n),
            std::cref(ring_size),
            std::cref(compute_kernel_config),
            std::cref(program_config),
            std::cref(ccl_core_grid_offset),
            std::cref(all_gather_operation_attributes),
            std::cref(all_gather_tensor_args),
            std::cref(block_size),
            std::cref(q_blocks_per_pass));
    }
};

// Q / K / V are sequence-sharded along the ring; the fused all-gather fills gathered_k / gathered_v. The block
// mask and pooled K / V come from the routing pre-pass (see ttnn::transformer::sparse_ring_sdpa):
//   block_mask  uint32 row-major [1, NH, nb_pad, n_blocks_pad]: != 0 computes (local block_size-row Q block, global
//               block_size-token K block) exactly; global block s * nb_pad + b is block b of shard s.
//   pooled_k/v  OPTIONAL bf16 tile [1, NH, n_blocks_pad, DH]: per-block mean K / mean V over the valid tokens, whole
//               sequence. Present (Sol-Attn): every block the mask drops enters the softmax as one pooled entry
//               weighted by its token count, through the column bias the reader generates (pooled_bias_gen.hpp).
//               Absent (kept blocks only, e.g. the VSA fine branch): the softmax normalizes over the kept blocks.
struct SparseRingSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    Tensor gathered_k;
    Tensor gathered_v;
    Tensor block_mask;
    std::optional<Tensor> pooled_k;
    std::optional<Tensor> pooled_v;

    // The pooled route (Sol-Attn's correction for the dropped blocks) runs iff pooled K / V are given.
    bool pooled_route() const { return pooled_k.has_value(); }

    uint32_t local_kv_seq_len() const { return static_cast<uint32_t>(input_k.logical_shape()[2]); }
};

// SparseRingSDPAResult holds one tensor, the attention output.
constexpr size_t SPARSE_RING_SDPA_OUTPUT_IDX = 0;
using SparseRingSDPAResult = Tensors;
using SparseRingSDPAResultSpec = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::prim
