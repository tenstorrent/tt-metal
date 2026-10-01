// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/non_zero_indices.hpp"

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation.hpp"

namespace ttnn {

std::vector<Tensor> nonzero(
    const Tensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    // Default to DRAM INTERLEAVED — do NOT inherit the input memory config because the input
    // may be sharded, and the output is always variable-length (precludes sharding).
    const tt::tt_metal::MemoryConfig default_mc{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto [output_0, output_1] = ttnn::prim::nonzero(input_tensor, memory_config.value_or(default_mc));
    std::vector<Tensor> output_tensor_vec;
    output_tensor_vec.reserve(2);
    output_tensor_vec.push_back(std::move(output_0));
    output_tensor_vec.push_back(std::move(output_1));
    return output_tensor_vec;
}

}  // namespace ttnn
