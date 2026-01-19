// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "device/reshard_device_operation.hpp"
#include "reshard.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ReshardOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::reshard(input_tensor, memory_config, optional_output_tensor);
}

}  // namespace ttnn::operations::data_movement
