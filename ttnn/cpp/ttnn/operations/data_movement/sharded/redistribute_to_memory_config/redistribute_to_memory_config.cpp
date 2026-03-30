// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "device/redistribute_to_memory_config_device_operation.hpp"
#include "redistribute_to_memory_config.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn {

ttnn::Tensor redistribute_to_memory_config(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& output_memory_config,
    const std::optional<DataType>& data_type_arg,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::prim::redistribute_to_memory_config(
        input_tensor, output_memory_config, data_type_arg.value_or(input_tensor.dtype()), preallocated_output);
}

}  // namespace ttnn
