// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement::detail {

enum class ReshardStridesInRange { ALL_STRIDES, FIRST_HALF, SECOND_HALF };
// start is inclusive, end is exclusive
struct PageRange {
    uint32_t start;
    uint32_t end;
};

struct Stride {
    CoreCoord core;
    uint32_t data;
};

struct PageStride {
    CoreCoord start_core;
    uint32_t start_data;
    uint32_t stride_size;  // number of pages per stride
    Stride stride;
    uint32_t num_strides;
    bool skip;
};

struct CorePageRange {
    CoreCoord core;
    PageRange range;
};

struct CorePageStride {
    CoreCoord core;
    PageStride page_stride;
};

// Utility function
uint32_t calculate_starting_idx_h(const Tensor& tensor, uint32_t num_slices, uint32_t slice_index);

struct WidthShardingReshardSegment {
    uint32_t write_size = 0;
    uint32_t read_offset = 0;
    uint32_t bank_id = 0;
    uint32_t write_offset = 0;
};

// Precompute a set of reads/writes for each core needed to perform a width-sharded reshard operations
std::tuple<std::vector<std::vector<WidthShardingReshardSegment>>, uint32_t, uint32_t, uint32_t>
compute_width_sharding_reshard_segments(
    const std::array<uint32_t, 2>& local_shard_shape,
    const std::array<uint32_t, 2>& remote_shard_shape,
    const std::vector<CoreCoord>& local_cores,
    const std::vector<CoreCoord>& remote_cores,
    const tt::tt_metal::BufferType& remote_buffer_type,
    const CoreType& remote_core_type,
    tt::tt_metal::IDevice* device,
    uint32_t element_size);

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn::operations::data_movement {

bool is_valid_for_2d_reshard(const Tensor& input_tensor, const MemoryConfig& out_mem_config);
std::vector<uint32_t> get_runtime_args_for_given_ranges(
    const std::vector<uint32_t>& physical_core_coords,
    const std::vector<detail::PageStride>& page_stride_vector,
    const uint32_t output_page_offset,
    const uint32_t& input_addr,
    const uint32_t starting_range,
    const uint32_t ending_range,
    const detail::ReshardStridesInRange reshard_strides_in_range = detail::ReshardStridesInRange::ALL_STRIDES);
std::unordered_map<CoreCoord, std::vector<detail::PageStride>> create_map_for_reshard(
    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page,
    Buffer* input_buffer,
    Buffer* output_buffer);
std::unordered_map<CoreCoord, std::vector<detail::PageStride>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer);
std::unordered_map<CoreCoord, std::vector<detail::PageStride>> get_core_page_ranges_diff_width(
    Buffer* input_buffer, Buffer* output_buffer, const Tensor& input);
Tensor construct_per_core_host_tensor(const std::unordered_map<CoreCoord, std::vector<uint32_t>>& core_to_data);
Tensor move_per_core_config_to_device(
    const Tensor& host_tensor, const CoreRangeSet& grid, distributed::MeshDevice* device);

}  // namespace ttnn::operations::data_movement
