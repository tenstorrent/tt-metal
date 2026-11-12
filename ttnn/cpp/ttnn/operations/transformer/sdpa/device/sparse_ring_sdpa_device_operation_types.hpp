// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    uint32_t block_size = 64;
    std::optional<uint32_t> q_blocks_per_pass = std::nullopt;
    // Dense Q blocks (prefix spread, Sol's exact prefix rows): global Q-block indices shard * Q blocks per shard + local
    // block whose mask rows keep every full key block. With the dynamic split on, each unit's cores split it by cost
    // (sparse_dyn_split.hpp weighted_range). Empty: the equal counts. Part of the program hash.
    std::vector<uint32_t> dense_q_blocks;
    // Token-major output (concatenate_heads layout) [B, 1, N, H * D] and its dtype (nullopt = bf16). Set after
    // construction; keyed in compute_program_hash.
    bool output_token_major = false;
    std::optional<DataType> output_dtype = std::nullopt;

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
        std::optional<uint32_t> q_blocks_per_pass,
        std::vector<uint32_t> dense_q_blocks = {}) :
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
        q_blocks_per_pass(q_blocks_per_pass),
        dense_q_blocks(std::move(dense_q_blocks)) {}

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
        "q_blocks_per_pass",
        "dense_q_blocks");
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
            std::cref(q_blocks_per_pass),
            std::cref(dense_q_blocks));
    }
};

struct SparseRingSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    Tensor gathered_k;
    Tensor gathered_v;
    Tensor block_mask;
    std::optional<Tensor> pooled_k;
    std::optional<Tensor> pooled_v;
    std::optional<Tensor> block_counts;
    std::optional<Tensor> head_weights;

    bool pooled_route() const { return pooled_k.has_value(); }

    uint32_t local_kv_seq_len() const { return static_cast<uint32_t>(input_k.logical_shape()[2]); }
};

constexpr size_t SPARSE_RING_SDPA_OUTPUT_IDX = 0;
using SparseRingSDPAResult = Tensors;
using SparseRingSDPAResultSpec = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::prim
