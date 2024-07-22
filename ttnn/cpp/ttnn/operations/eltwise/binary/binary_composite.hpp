// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_composite_op.hpp"

namespace ttnn {

namespace operations {

namespace binary {

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOps
{
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function<binary_comp_op_type>();
            return op_type(input_tensor_a, input_tensor_b, memory_config);
        }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsFloat
{
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        float alpha,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_with_float<binary_comp_op_type>();
            return op_type(input_tensor_a, input_tensor_b, alpha, memory_config);
        }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsIsClose
{
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        float rtol,
        float atol,
        const bool equal_nan,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_isclose<binary_comp_op_type>();
            return op_type(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
        }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsOverload
{
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_divnonan_floordiv<binary_comp_op_type>();
            return op_type(input_tensor_a, input_tensor_b, memory_config);
        }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_divnonan_floordiv_overload<binary_comp_op_type>();
            return op_type(input_tensor_a, value, memory_config);
        }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsDiv
{
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        bool accurate_mode = false,
        string round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_div<binary_comp_op_type>();
            return op_type(input_tensor_a, input_tensor_b, accurate_mode, round_mode, memory_config);
        }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        float value,
        bool accurate_mode = false,
        string round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_div_overload<binary_comp_op_type>();
            return op_type(input_tensor_a, value, accurate_mode, round_mode, memory_config);
        }
};

}  // namespace binary
}  // namespace operations

// newly imported
constexpr auto hypot = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::HYPOT>>("ttnn::hypot");
constexpr auto xlogy = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::XLOGY>>("ttnn::xlogy");
constexpr auto minimum = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::MINIMUM>>("ttnn::minimum");
constexpr auto maximum = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::MAXIMUM>>("ttnn::maximum");
constexpr auto atan2 = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::ATAN2>>("ttnn::atan2");
constexpr auto logical_xor = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::LOGICAL_XOR>>("ttnn::logical_xor");
constexpr auto nextafter = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::NEXTAFTER>>("ttnn::nextafter");
constexpr auto binary_remainder = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::BINARY_REMAINDER>>("ttnn::binary_remainder");
constexpr auto binary_fmod = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::BINARY_FMOD>>("ttnn::binary_fmod");
constexpr auto addalpha = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOpsFloat<operations::binary::BinaryCompositeOpType::ADDALPHA>>("ttnn::addalpha");
constexpr auto subalpha = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOpsFloat<operations::binary::BinaryCompositeOpType::SUBALPHA>>("ttnn::subalpha");
constexpr auto isclose = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOpsIsClose<operations::binary::BinaryCompositeOpType::ISCLOSE>>("ttnn::isclose");
constexpr auto div = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOpsDiv<operations::binary::BinaryCompositeOpType::DIV>>("ttnn::div");
constexpr auto div_no_nan = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOpsOverload<operations::binary::BinaryCompositeOpType::DIV_NO_NAN>>("ttnn::div_no_nan");
constexpr auto floor_div = ttnn::register_operation<operations::binary::ExecuteBinaryCompositeOpsOverload<operations::binary::BinaryCompositeOpType::FLOOR_DIV>>("ttnn::floor_div");

}  // namespace ttnn
