// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "device/to_sharded_device_operation.hpp"
#include "to_sharded.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn {

ttnn::Tensor to_sharded(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& sharded_memory_config,
    const std::optional<DataType>& data_type_arg,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::prim::to_sharded(
        input_tensor, sharded_memory_config, data_type_arg.value_or(input_tensor.dtype()), preallocated_output);
}

}  // namespace ttnn
