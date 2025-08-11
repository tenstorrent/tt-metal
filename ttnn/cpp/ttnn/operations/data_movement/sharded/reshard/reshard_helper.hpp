// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
