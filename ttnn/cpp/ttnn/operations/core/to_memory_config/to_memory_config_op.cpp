// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_memory_config_op.hpp"

#include "ttnn/core.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

Tensor to_memory_config(
    const Tensor& tensor,
    const MemoryConfig& memory_config,
    std::optional<DataType> dtype,
    const std::optional<Tensor>& output_tensor) {
    // Temporary until we see why buffer data not being populated
    const auto original_memory_config = ttnn::get_memory_config(tensor);
    if (original_memory_config.has_value() && original_memory_config.value() == memory_config &&
        !output_tensor.has_value()) {
        return tensor;
    }
    std::vector<std::optional<Tensor>> optional_output_tensors;
    if (output_tensor.has_value()) {
        optional_output_tensors.push_back(output_tensor);
    }

    // The specialized sharded<->interleaved and reshard paths have been removed; ttnn::copy is the
    // general fallback that handles arbitrary memory-config conversions.
    return ttnn::prim::copy(
        tensor,
        memory_config,
        dtype.value_or(tensor.dtype()),
        optional_output_tensors.empty() ? std::nullopt : optional_output_tensors.at(0));
}

}  // namespace ttnn
