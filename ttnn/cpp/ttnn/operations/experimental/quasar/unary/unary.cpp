// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary.hpp"

#include "ttnn/operations/experimental/quasar/unary/device/unary_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

namespace detail {

Tensor unary_impl(
    const Tensor& input_tensor,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(!op_chain.empty(), "Quasar unary: op_chain must not be empty");
    const DataType input_dtype = input_tensor.dtype();
    // A preallocated output may switch between the two float formats; the device op validates both dtypes.
    const DataType output_dtype = optional_output_tensor.has_value() ? optional_output_tensor->dtype() : input_dtype;
    // Same derivation as the upstream unary_impl, restricted to the float dtypes this port supports.
    const bool preserve_fp32_precision = input_dtype == DataType::FLOAT32;
    const bool fp32_dest_acc_en = preserve_fp32_precision || output_dtype == DataType::FLOAT32;
    const MemoryConfig output_memory_config = optional_output_tensor.has_value()
                                                  ? optional_output_tensor->memory_config()
                                                  : memory_config.value_or(input_tensor.memory_config());
    return ttnn::prim::qsr::unary(
        input_tensor,
        op_chain,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        optional_output_tensor);
}

}  // namespace detail

Tensor cos(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;
    return detail::unary_impl(input_tensor, {UnaryWithParam{UnaryOpType::COS}}, memory_config, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::quasar
