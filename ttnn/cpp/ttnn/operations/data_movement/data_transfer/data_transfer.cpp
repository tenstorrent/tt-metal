// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement {

Tensor DataTransferToHostOperation::operator()(const Tensor &input_tensor) {
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return input_tensor;
    }

    return input_tensor.cpu();
}


Tensor DataTransferToDeviceOperation::operator()(const Tensor &input_tensor, Device* device, const std::optional<MemoryConfig>& option_memory_config) {
    TT_FATAL(device != nullptr);

    if(input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(input_tensor.get_legacy_shape()[-1] * input_tensor.element_size() % sizeof(uint32_t) == 0);
    }

    if (input_tensor.storage_type() == StorageType::DEVICE && input_tensor.device() == device) {
        return {input_tensor};
    }

    const auto memory_config = option_memory_config.value_or(input_tensor.memory_config());
    return input_tensor.to(device, memory_config);
}

}  // namespace ttnn::operations::data_movement
