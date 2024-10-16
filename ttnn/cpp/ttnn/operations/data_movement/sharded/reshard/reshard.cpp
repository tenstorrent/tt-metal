// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "device/reshard_op.hpp"
#include "reshard.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ReshardOperation::invoke(uint8_t queue_id,
                                      const ttnn::Tensor& input_tensor,
                                      const MemoryConfig& memory_config,
                                      const std::optional<Tensor>& optional_output_tensor) {
    return operation::run(
               ReshardDeviceOperation{.output_mem_config = memory_config}, {input_tensor}, {}, {optional_output_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
