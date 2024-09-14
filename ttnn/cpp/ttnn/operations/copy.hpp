// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/copy/typecast/typecast.hpp"

namespace ttnn {
namespace operations {
namespace copy {

namespace detail {

inline Tensor copy_impl(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::vector<ttnn::operations::unary::UnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    DataType output_dtype = (op_chain[0].op_type == ttnn::operations::unary::UnaryOpType::TYPECAST) ? static_cast<DataType>(op_chain[0].params[1]) : input_tensor.get_dtype();
    bool preserve_fp32_precision = (op_chain[0].op_type == ttnn::operations::unary::UnaryOpType::TYPECAST) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision or
                            output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or
                            output_dtype == DataType::FLOAT32 or
                            input_tensor.get_dtype() == DataType::UINT32 or
                            input_tensor.get_dtype() == DataType::INT32; // MT: Currently only uint32/int32 is moved to
                                                                          // DST directly, fp32 is converted to fp16b

    auto output_memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config.value_or(input_tensor.memory_config());
    return prim::unary(queue_id, input_tensor, op_chain, output_dtype, output_memory_config, fp32_dest_acc_en, preserve_fp32_precision, optional_output_tensor);
}
}  // namespace detail

struct Typecast {
    static Tensor invoke(
        const uint8_t queue_id,
        const Tensor& input,
        const DataType& output_dtype,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        if (optional_output_tensor.has_value()) {
            TT_FATAL(
                output_dtype == optional_output_tensor.value().get_dtype(),
                "If both output dtype and output tensor provided dtype should match");
        }
        if (DeviceArch(input.device()) == tt::ARCH::GRAYSKULL) {
            return ttnn::experimental::typecast(queue_id, input, output_dtype, memory_config_arg, optional_output_tensor);
        }
        DataType input_dtype = input.get_dtype();
        return detail::copy_impl(
            queue_id,
            input,
            {ttnn::operations::unary::UnaryWithParam(
                ttnn::operations::unary::UnaryOpType::TYPECAST,
                {static_cast<float>(input_dtype), static_cast<float>(output_dtype)})},
            memory_config_arg,
            optional_output_tensor);
    }

    static Tensor invoke(
        const Tensor& input,
        const DataType& output_dtype,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(DefaultQueueId, input, output_dtype, memory_config_arg, optional_output_tensor);
    }

    // eltwise_typecast implementation in tt_eager :
    // ---------------------------------------------
    // inline Tensor eltwise_typecast(
    //     const Tensor& input_tensor,
    //     uint32_t tt_input_dtype,
    //     uint32_t tt_output_dtype,
    //     const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG)

    static ttnn::Tensor invoke(
        const uint8_t queue_id,
        const Tensor& input_tensor,
        const DataType& tt_input_dtype,
        const DataType& tt_output_dtype,
        const std::optional<MemoryConfig>& memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        TT_ASSERT(
            DeviceArch(input_tensor.device()) != tt::ARCH::GRAYSKULL,
            "eltwise_typecast is not currently supported on Grayskull");
        TT_FATAL(
            tt_input_dtype == input_tensor.get_dtype(), "input dtype and input tensor's dtype provided should match");
        if (optional_output_tensor.has_value()) {
            TT_FATAL(
                tt_output_dtype == optional_output_tensor.value().get_dtype(),
                "If both output dtype and output tensor provided dtype should match");
        }
        return detail::copy_impl(
            queue_id,
            input_tensor,
            {ttnn::operations::unary::UnaryWithParam(
                ttnn::operations::unary::UnaryOpType::TYPECAST,
                {static_cast<float>(tt_input_dtype), static_cast<float>(tt_output_dtype)})},
            memory_config,
            optional_output_tensor);
    }
};
}  // namespace copy
}  // namespace operations

constexpr auto typecast = ttnn::register_operation_with_auto_launch_op<"ttnn::typecast", ttnn::operations::copy::Typecast>();

}  // namespace ttnn
