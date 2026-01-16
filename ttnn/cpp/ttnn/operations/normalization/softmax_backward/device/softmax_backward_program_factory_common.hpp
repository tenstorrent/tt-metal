// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_backward_operation_types.hpp"

#include <tt-metalium/kernel_types.hpp>
#include <ttnn/tensor/tensor.hpp>

#include <utility>  // for std::pair

namespace ttnn::operations::normalization::softmax_backward {

// Estimate L1 memory usage for non-streaming kernel
// Returns: (use_non_streaming_kernel, estimated_memory_bytes)
std::pair<bool, uint32_t> should_use_non_streaming_kernel(uint32_t width_tiles, uint32_t tile_size);

// Helper function to get common tensor properties
void get_tensor_properties(
    const ttnn::Tensor& softmax_output,
    const operation_attributes_t& operation_attributes,
    uint32_t& num_rows,
    uint32_t& width_tiles,
    uint32_t& mask_w,
    tt::DataFormat& input_data_format,
    tt::DataFormat& output_data_format,
    tt::DataFormat& intermed_data_format,
    uint32_t& input_tile_size,
    uint32_t& output_tile_size,
    uint32_t& intermed_tile_size,
    const ttnn::Tensor& tensor_return_value);

// Helper function to create precise compute config
tt::tt_metal::ComputeConfig precise(
    std::vector<uint32_t> compile_time_args, std::map<std::string, std::string> defines);

}  // namespace ttnn::operations::normalization::softmax_backward
