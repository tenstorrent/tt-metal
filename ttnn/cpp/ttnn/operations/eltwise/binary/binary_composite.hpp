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
    static Tensor operator()(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, memory_config);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsFloat
{
    static Tensor operator()(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        float alpha,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, alpha, memory_config);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsIsClose
{
    static Tensor operator()(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        float rtol,
        float atol,
        const bool equal_nan,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteDivLikeOps
{
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_binary_div_like_ops<binary_comp_op_type>();
            return op_type(input_tensor_a, input_tensor_b, memory_config);
        }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_binary_div_like_ops_overload<binary_comp_op_type>();
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
            auto op_type = get_binary_div<binary_comp_op_type>();
            return op_type(input_tensor_a, input_tensor_b, accurate_mode, round_mode, memory_config);
        }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a,
        float value,
        bool accurate_mode = false,
        string round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_binary_div_overload<binary_comp_op_type>();
            return op_type(input_tensor_a, value, accurate_mode, round_mode, memory_config);
        }
};

}  // namespace binary
}  // namespace operations

    // newly imported
    constexpr auto hypot = ttnn::register_operation_with_auto_launch_op<
        "ttnn::hypot",
        operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::HYPOT>>();
constexpr auto xlogy = ttnn::register_operation_with_auto_launch_op<
    "ttnn::xlogy",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::XLOGY>>();
constexpr auto minimum = ttnn::register_operation_with_auto_launch_op<
    "ttnn::minimum",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::MINIMUM>>();
constexpr auto maximum = ttnn::register_operation_with_auto_launch_op<
    "ttnn::maximum",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::MAXIMUM>>();
constexpr auto atan2 = ttnn::register_operation_with_auto_launch_op<
    "ttnn::atan2",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::ATAN2>>();
constexpr auto logical_xor = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_xor",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::LOGICAL_XOR>>();
constexpr auto nextafter = ttnn::register_operation_with_auto_launch_op<
    "ttnn::nextafter",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::NEXTAFTER>>();
constexpr auto addalpha = ttnn::register_operation_with_auto_launch_op<
    "ttnn::addalpha",
    operations::binary::ExecuteBinaryCompositeOpsFloat<operations::binary::BinaryCompositeOpType::ADDALPHA>>();
constexpr auto subalpha = ttnn::register_operation_with_auto_launch_op<
    "ttnn::subalpha",
    operations::binary::ExecuteBinaryCompositeOpsFloat<operations::binary::BinaryCompositeOpType::SUBALPHA>>();
constexpr auto isclose = ttnn::register_operation_with_auto_launch_op<
    "ttnn::isclose",
    operations::binary::ExecuteBinaryCompositeOpsIsClose<operations::binary::BinaryCompositeOpType::ISCLOSE>>();
constexpr auto binary_remainder = ttnn::register_operation<
    "ttnn::binary_remainder",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::BINARY_REMAINDER>>();
constexpr auto binary_fmod = ttnn::register_operation<
    "ttnn::binary_fmod",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::BINARY_FMOD>>();
constexpr auto div = ttnn::register_operation<
    "ttnn::div",
    operations::binary::ExecuteBinaryCompositeOpsDiv<operations::binary::BinaryCompositeOpType::DIV>>();
constexpr auto div_no_nan = ttnn::register_operation<
    "ttnn::div_no_nan",
    operations::binary::ExecuteBinaryCompositeOpsOverload<operations::binary::BinaryCompositeOpType::DIV_NO_NAN>>();
constexpr auto floor_div = ttnn::register_operation<
    "ttnn::floor_div",
    operations::binary::ExecuteBinaryCompositeOpsOverload<operations::binary::BinaryCompositeOpType::FLOOR_DIV>>();
constexpr auto logical_and_ = ttnn::register_operation<
    "ttnn::logical_and_",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::LOGICAL_AND_>>();
constexpr auto logical_or_ = ttnn::register_operation<
    "ttnn::logical_or_",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::LOGICAL_OR_>>();
constexpr auto logical_xor_ = ttnn::register_operation<
    "ttnn::logical_xor_",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::LOGICAL_XOR_>>();
// newly imported

}  // namespace ttnn
