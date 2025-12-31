// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads.hpp"
#include "device/nlp_concat_heads_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

Tensor nlp_concat_heads(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    using OperationType = operations::experimental::nlp_concat_heads::NLPConcatHeadsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
