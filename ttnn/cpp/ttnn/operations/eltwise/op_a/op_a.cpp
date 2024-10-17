// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "op_a.hpp"

#include "ttnn/common/constants.hpp"
#include "device/unary_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/pool/downsample/device/downsample_op.hpp"
#include "ttnn/operations/core/core.hpp"


namespace ttnn::operations::op_a {

using ttnn::operations::unary::UnaryWithParam;
using ttnn::operations::unary::UnaryOpType;

namespace detail {

inline Tensor unary_impl(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& op_chain,
    const float param,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    DataType output_dtype = (op_chain[0].op_type == UnaryOpType::TYPECAST) ? static_cast<DataType>(op_chain[0].params[1]) : input_tensor.get_dtype();
    bool preserve_fp32_precision = (op_chain[0].op_type == UnaryOpType::TYPECAST) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision or
                            output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or
                            output_dtype == DataType::FLOAT32 or
                            input_tensor.get_dtype() == DataType::UINT32 or
                            input_tensor.get_dtype() == DataType::INT32;  // MT: Currently only uint32/int32 is moved to
                                                                          // DST directly, fp32 is converted to fp16b

    auto output_memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config.value_or(input_tensor.memory_config());
    return prim::op_a(queue_id, input_tensor, op_chain, param, output_dtype, output_memory_config, fp32_dest_acc_en, preserve_fp32_precision, optional_output_tensor);
}

}  // namespace detail


Tensor ExecuteUnaryWithFloatParameter::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const float parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam{UnaryOpType::FILL, static_cast<float>(parameter)}},
        parameter,
        memory_config,
        optional_output_tensor);
}

Tensor ExecuteUnaryWithFloatParameter::invoke(
    const Tensor& input_tensor,
    const float parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
        input_tensor,
        {UnaryWithParam{UnaryOpType::FILL, static_cast<float>(parameter)}},
        parameter,
        memory_config,
        optional_output_tensor);
}

}
