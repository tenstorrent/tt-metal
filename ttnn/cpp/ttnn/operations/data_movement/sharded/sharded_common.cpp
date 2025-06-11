// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/tensor/tensor.hpp"

#include "sharded_common.hpp"

namespace ttnn::operations::data_movement::detail {

// Utility function
uint32_t calculate_starting_idx_h(const Tensor& tensor, uint32_t num_slices, uint32_t slice_index) {
    if (num_slices <= 1) {
        return 0;
    }

    uint32_t num_tiles_height = tensor.physical_volume() / tensor.padded_shape()[-1] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_width = tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    uint32_t total_num_tiles = num_tiles_height * num_tiles_width;

    uint32_t num_tiles_per_slice = total_num_tiles / num_slices;
    uint32_t starting_tile_in_slice = num_tiles_per_slice * slice_index;
    return starting_tile_in_slice;
}

std::tuple<std::vector<std::vector<WidthShardingReshardSegment>>, uint32_t, uint32_t, uint32_t>
compute_width_sharding_reshard_segments(
    const std::array<uint32_t, 2>& local_shard_shape,
    const std::array<uint32_t, 2>& remote_shard_shape,
    const std::vector<CoreCoord>& local_cores,
    const std::vector<CoreCoord>& remote_cores,
    const tt::tt_metal::BufferType& remote_buffer_type,
    const CoreType& remote_core_type,
    tt::tt_metal::IDevice* device,
    uint32_t element_size) {
    const uint32_t num_local_shards = local_cores.size();
    const uint32_t num_remote_shards = remote_cores.size();

    const uint32_t local_shard_height = local_shard_shape[0];
    const uint32_t local_shard_width = local_shard_shape[1];
    const uint32_t remote_shard_height = remote_shard_shape[0];
    const uint32_t remote_shard_width = remote_shard_shape[1];

    using WidthShardingReshardSegmentForSingleCore = std::vector<WidthShardingReshardSegment>;

    TT_FATAL(
        local_shard_height == remote_shard_height,
        "Unexpected mismatch in shard heights ({} != {}",
        local_shard_height,
        remote_shard_height);

    const uint32_t total_num_sticks = local_shard_height;
    const uint32_t local_stride_bytes = element_size * local_shard_width;
    const uint32_t remote_stride_bytes = element_size * remote_shard_width;

    std::vector<WidthShardingReshardSegmentForSingleCore> runtime_args_for_each_core;

    bool is_final_transfer = false;
    uint32_t local_shard_offset = 0;
    uint32_t remote_shard_offset = 0;
    uint32_t current_remote_core_idx = 0;
    for (uint32_t current_local_core_idx = 0; current_local_core_idx < local_cores.size(); current_local_core_idx++) {
        const auto& core = local_cores[current_local_core_idx];
        WidthShardingReshardSegmentForSingleCore core_args;
        while (local_shard_offset < local_shard_width) {
            const uint32_t remaining_input = local_shard_width - local_shard_offset;
            const uint32_t remaining_output = remote_shard_width - remote_shard_offset;

            // The last core might have some garbage in it because of uneven shards
            is_final_transfer = (current_local_core_idx >= local_cores.size() - 1) &&
                                (current_remote_core_idx >= remote_cores.size() - 1);
            const uint32_t transfer_size =
                is_final_transfer ? remaining_output : std::min(remaining_input, remaining_output);

            const auto bank_id = device->allocator()->get_bank_ids_from_logical_core(
                remote_buffer_type, remote_cores[current_remote_core_idx])[0];
            core_args.emplace_back(
                element_size * transfer_size,
                element_size * local_shard_offset,
                bank_id,
                element_size * remote_shard_offset);

            local_shard_offset += transfer_size;
            remote_shard_offset += transfer_size;

            // If the current output shard is full, move to the next one
            if (remote_shard_offset == remote_shard_width) {
                ++current_remote_core_idx;
                remote_shard_offset = 0;
            }
            if (is_final_transfer) {
                break;
            }
        }
        local_shard_offset = 0;
        runtime_args_for_each_core.push_back(core_args);
    }

    TT_FATAL(
        runtime_args_for_each_core.size() == num_local_shards,
        "Expect to have one set of runtime args per local core (expected {} but was {})",
        num_local_shards,
        runtime_args_for_each_core.size());  // sanity check

    return {runtime_args_for_each_core, total_num_sticks, local_stride_bytes, remote_stride_bytes};
}

}  // namespace ttnn::operations::data_movement::detail
