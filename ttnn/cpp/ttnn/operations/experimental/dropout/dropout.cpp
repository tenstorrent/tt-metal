
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/dropout_device_operation.hpp"
#include "dropout.hpp"

namespace ttnn::operations::experimental {

Tensor DropoutOperation::invoke(
    const Tensor& input_tensor,
    float prob,
    float scale,
    uint32_t seed,
    bool use_per_device_seed,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::dropout(
        input_tensor,
        prob,
        scale,
        seed,
        use_per_device_seed,
        DataType::BFLOAT16,
        output_memory_config,
        optional_output_tensor);
}

}  // namespace ttnn::operations::experimental
