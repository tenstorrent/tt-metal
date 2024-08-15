// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct DataTransferToHostOperation {
    static Tensor invoke(const Tensor &input_tensor);
};

struct DataTransferToDeviceOperation {
    static Tensor invoke(const Tensor &input_tensor, Device* device, const MemoryConfig& memory_config);
};

} // operations::data_movement

constexpr auto data_transfer_to_host = ttnn::register_operation_with_auto_launch_op<"ttnn::data_transfer_to_host", ttnn::operations::data_movement::DataTransferToHostOperation>();
constexpr auto data_transfer_to_device = ttnn::register_operation_with_auto_launch_op<"ttnn::data_transfer_to_device", ttnn::operations::data_movement::DataTransferToDeviceOperation>();

}  // namespace ttnn
