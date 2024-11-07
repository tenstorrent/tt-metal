
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng.hpp"
#include "device/binary_ng_device_operation.hpp"

namespace ttnn::operations::binary_ng {

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const DataType> &output_dtype,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return ttnn::prim::binary_ng(
        queue_id, input_tensor_a, input_tensor_b, binary_op_type, output_dtype, memory_config, optional_output_tensor);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const std::optional<const DataType> &output_dtype,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return invoke(DefaultQueueId, input_tensor_a, input_tensor_b, output_dtype, memory_config, optional_output_tensor);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor &input_tensor_a,
    float scalar,
    const std::optional<const DataType> &output_dtype,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return ttnn::prim::binary_ng(queue_id, input_tensor_a, scalar, binary_op_type, output_dtype, memory_config, optional_output_tensor);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    const Tensor &input_tensor_a,
    float scalar,
    const std::optional<const DataType> &output_dtype,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return invoke(DefaultQueueId, input_tensor_a, scalar, output_dtype, memory_config, optional_output_tensor);
}

template struct BinaryNg<BinaryOpType::ADD>;

}  // namespace ttnn::operations::binary_ng
