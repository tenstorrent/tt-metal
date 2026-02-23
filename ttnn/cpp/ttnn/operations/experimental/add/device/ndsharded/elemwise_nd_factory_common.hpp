
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/add/device/ndsharded/kernels/elt_nd_sharded_add_args.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <optional>
#include <unordered_map>

namespace ttnn::experimental::prim {

template <bool initialize_args>
inline void set_elt_nd_sharded_add_runtime_args(
    tt::tt_metal::Program& program,
    const Tensor& a_tensor,
    const Tensor& b_tensor,
    const Tensor& output,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::KernelHandle compute_kernel_id,
    const CoreRangeSet& all_device_cores,
    const std::vector<CoreCoord>* ordered_cores_opt = nullptr) {
    using namespace eltwise_nd_dram_optimized;
    using namespace ttnn::kernel_utils;
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    auto fallback_cores = corerange_to_cores(all_device_cores, {}, true);
    const bool use_ordered = ordered_cores_opt && ordered_cores_opt->size() == fallback_cores.size();
    const std::vector<CoreCoord> cores = use_ordered ? *ordered_cores_opt : fallback_cores;
    auto distribution_spec = a_tensor.tensor_spec().compute_buffer_sharding_args().buffer_distribution_spec();
    TT_FATAL(distribution_spec.has_value(), "Buffer distribution spec must be set");

    std::vector<std::vector<uint32_t>> reader_args_array{
        cores.size(), std::vector<uint32_t>{amount_of_fields<EltwiseReaderArgs>(), 0}};
    std::vector<std::vector<uint32_t>> compute_args_array{
        cores.size(), std::vector<uint32_t>(amount_of_fields<EltwiseComputeArgs>(), 0)};
    std::vector<std::vector<uint32_t>> writer_args_array{
        cores.size(), std::vector<uint32_t>(amount_of_fields<EltwiseWriterArgs>(), 0)};

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    // const auto num_tiles_per_shard_width = distribution_spec->num_tiles_per_shard() / tt::constants::TILE_WIDTH;
    const auto num_tiles_per_shard_width = a_tensor.shard_spec()->shape[-1] / tt::constants::TILE_WIDTH;
    const auto num_tiles_per_shard_height = a_tensor.shard_spec()->shape[-2] / tt::constants::TILE_HEIGHT;
    const auto num_tiles_per_shard = num_tiles_per_shard_height * num_tiles_per_shard_width;

    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        const CoreCoord& core = cores[core_id];
        // Worker i from get_optimal_dram_bank_to_logical_worker_assignment is optimal for DRAM bank i.
        // Buffer shard k is on buffer core at index k (same order as allocation).
        const uint32_t start_shard_id = core_id;

        const uint32_t num_shards_this_core = num_tiles_per_shard_width;
        // Reader run-time args
        EltwiseReaderArgs reader_runtime_args = {
            .a_tensor_base_addr = a_tensor.buffer()->address(),
            .b_tensor_base_addr = b_tensor.buffer()->address(),
            .num_shards = num_shards_this_core,
            .shard_id = start_shard_id,
            .next_shard_offset = static_cast<uint32_t>(cores.size())};

        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, ttnn::kernel_utils::to_vector(reader_runtime_args));

        // Compute processes all tiles for all shards this core handles

        const uint32_t total_tiles_this_core = static_cast<uint32_t>(num_tiles_per_shard) * num_shards_this_core;
        EltwiseComputeArgs compute_kern_args = {.num_tiles = total_tiles_this_core};

        const uint32_t num_cycles_per_shard = static_cast<uint32_t>(num_tiles_per_shard) /
                                              num_tiles_per_shard_width;  // tiles per shard / tiles per cycle
        EltwiseWriterArgs write_kern_args = {
            .dst_base_addr = output.buffer()->address(),
            .num_shards = num_shards_this_core,
            .shard_id = start_shard_id,
            .next_shard_offset = static_cast<uint32_t>(cores.size()),
            .num_cycles_per_shard = num_cycles_per_shard};

        reader_args_array[core_id] = to_vector(reader_runtime_args);
        compute_args_array[core_id] = to_vector(compute_kern_args);
        writer_args_array[core_id] = to_vector(write_kern_args);
        if constexpr (!initialize_args) {
            auto& core_reader_args = cached_reader_args.at(core.x).at(core.y);
            std::ranges::copy(reader_args_array[core_id], core_reader_args.data());

            auto& core_eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            std::ranges::copy(compute_args_array[core_id], core_eltwise_args.data());

            auto& core_writer_args = cached_writer_args.at(core.x).at(core.y);
            std::ranges::copy(writer_args_array[core_id], core_writer_args.data());
        }
    }

    if constexpr (initialize_args) {
        SetRuntimeArgs(program, reader_kernel_id, cores, reader_args_array);
        SetRuntimeArgs(program, compute_kernel_id, cores, compute_args_array);
        SetRuntimeArgs(program, writer_kernel_id, cores, writer_args_array);
    }
}
}  // namespace ttnn::experimental::prim
