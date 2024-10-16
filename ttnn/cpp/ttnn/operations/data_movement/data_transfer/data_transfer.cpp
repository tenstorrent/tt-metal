// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement {

Tensor DataTransferToHostOperation::invoke(const Tensor &input_tensor) {
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return input_tensor;
    }

    return input_tensor.cpu();
}

Tensor DataTransferToDeviceOperation::invoke(const Tensor &input_tensor,
                                             Device *device,
                                             const MemoryConfig &memory_config) {
    TT_FATAL(device != nullptr, "Error");

    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(input_tensor.get_legacy_shape()[-1] * input_tensor.element_size() % sizeof(uint32_t) == 0, "Error");
    }

    if (input_tensor.storage_type() == StorageType::DEVICE && input_tensor.device() == device) {
        return {input_tensor};
    }

    return input_tensor.to(device, memory_config);
}

}  // namespace ttnn::operations::data_movement
