
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hypot.hpp"
#include "ttnn/operations/eltwise/binary_ng/hypot/device/hypot_device_operation.hpp"

namespace ttnn::operations::binary_ng {

namespace utils {
inline Tensor typecast_to(DataType dtype, const Tensor& input) {
    return input.get_dtype() == dtype ? input : ttnn::typecast(input, dtype);
}

inline bool needs_typecast_to_bfloat16(const Tensor& input) {
    return (input.get_dtype() == DataType::BFLOAT8_B || input.get_dtype() == DataType::BFLOAT4_B);
}
}

Tensor Hypot::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {

    bool typecast_a = utils::needs_typecast_to_bfloat16(input_tensor_a);
    bool typecast_b = utils::needs_typecast_to_bfloat16(input_tensor_b);
    Tensor input_a = typecast_a ? utils::typecast_to(DataType::BFLOAT16, input_tensor_a) : input_tensor_a;
    Tensor input_b = typecast_b ? utils::typecast_to(DataType::BFLOAT16, input_tensor_b) : input_tensor_b;
    return ttnn::prim::hypot(
        queue_id,
        input_a,
        input_b,
        memory_config,
        optional_output_tensor);
}

Tensor Hypot::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke(DefaultQueueId, input_tensor_a, input_tensor_b, memory_config, optional_output_tensor);
}

}  // namespace ttnn::operations::binary_ng
