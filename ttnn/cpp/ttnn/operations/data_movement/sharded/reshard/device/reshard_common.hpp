// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include <tt-metalium/allocator.hpp>

namespace ttnn::operations::data_movement::detail {

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

// Utility functions used by program factories
enum class ReshardStridesInRange { ALL_STRIDES, FIRST_HALF, SECOND_HALF };

std::unordered_map<CoreCoord, std::vector<PageStride>> get_core_page_ranges(
    tt::tt_metal::Buffer* input_buffer, tt::tt_metal::Buffer* output_buffer);

std::vector<uint32_t> get_runtime_args_for_given_ranges(
    const std::vector<uint32_t>& physical_core_coords,
    const std::vector<PageStride>& page_stride_vector,
    const uint32_t output_page_offset,
    const uint32_t& input_addr,
    const uint32_t starting_range,
    const uint32_t ending_range,
    const ReshardStridesInRange reshard_strides_in_range = ReshardStridesInRange::ALL_STRIDES);

tt::tt_metal::operation::ProgramWithCallbacks reshard_multi_core(const Tensor& input, Tensor& output);

}  // namespace ttnn::operations::data_movement::detail
