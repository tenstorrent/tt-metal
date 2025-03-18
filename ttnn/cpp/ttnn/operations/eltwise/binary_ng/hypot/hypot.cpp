
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
    const bool output_preallocated = optional_output_tensor.has_value();

    // RM is never BFLOAT8 or BFLOAT4 so we can assume it goes in here.
    if (!typecast_a && !typecast_b) {
        bool input_a_rm = input_tensor_a.get_layout() == Layout::ROW_MAJOR;
        bool input_b_rm = input_tensor_b.get_layout() == Layout::ROW_MAJOR;
        Tensor input_a =
            input_a_rm ? ttnn::to_layout(input_tensor_a, Layout::TILE, std::nullopt, std::nullopt, (IDevice*)nullptr)
                        : input_tensor_a;
        Tensor input_b =
            input_b_rm ? ttnn::to_layout(input_tensor_b, Layout::TILE, std::nullopt, std::nullopt, (IDevice*)nullptr)
                        : input_tensor_b;

        if (input_a_rm && input_b_rm) {
            // we don't support to_layout with optional output tensor
            TT_FATAL(
                !output_preallocated,
                "Optional output tensor with Row Major input is not supported right now for Elementwise operations");
        }

        Tensor result = prim::hypot(
            queue_id,
            input_a,
            input_b,
            memory_config,
            optional_output_tensor);

        // if both inputs are in row major, convert the output to row major
        // since there's no consensus here, avoiding the conversion if we have an excuse to is likely the best option
        // since it leads to better perf
        if (input_a_rm && input_b_rm) {
            result = ttnn::to_layout(result, Layout::ROW_MAJOR, std::nullopt, memory_config, (IDevice*)nullptr);
        }

        return result;
    }
    else{
        Tensor input_a = typecast_a ? utils::typecast_to(DataType::BFLOAT16, input_tensor_a) : input_tensor_a;
        Tensor input_b = typecast_b ? utils::typecast_to(DataType::BFLOAT16, input_tensor_b) : input_tensor_b;
        return ttnn::prim::hypot(
            queue_id,
            input_a,
            input_b,
            memory_config,
            optional_output_tensor);
    }
}

Tensor Hypot::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke(DefaultQueueId, input_tensor_a, input_tensor_b, memory_config, optional_output_tensor);
}

}  // namespace ttnn::operations::binary_ng
