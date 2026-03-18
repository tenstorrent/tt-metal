// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng.hpp"
#include "device/unary_ng_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn::operations::unary_ng {

namespace detail {

inline Tensor unary_ng_impl(
    const Tensor& input_tensor,
    const std::vector<unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    TT_FATAL(!op_chain.empty(), "Op chain cannot be empty");
    DataType input_dtype = input_tensor.dtype();
    DataType output_dtype = input_dtype;
    if (op_chain.back().type() == unary::UnaryOpType::TYPECAST ||
        op_chain.back().type() == unary::UnaryOpType::BITCAST) {
        output_dtype = static_cast<DataType>(*op_chain.back().get_param_if<float>(1));
    }
    bool preserve_fp32_precision = (input_dtype == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision || output_dtype == DataType::UINT32 ||
                            output_dtype == DataType::INT32 || output_dtype == DataType::FLOAT32 ||
                            output_dtype == DataType::UINT8 || input_dtype == DataType::UINT8 ||
                            input_dtype == DataType::UINT32 || input_dtype == DataType::INT32;
    bool bfp8_pack_precise =
        (op_chain.back().type() == unary::UnaryOpType::TYPECAST && output_dtype == DataType::BFLOAT8_B);

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());

    return prim::unary_ng(
        input_tensor,
        op_chain,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor,
        sub_core_grids);
}

}  // namespace detail

Tensor Abs::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    unary::UnaryOpType op_type = unary::UnaryOpType::ABS;
    if (input_tensor.dtype() == DataType::INT32) {
        op_type = unary::UnaryOpType::ABS_INT32;
    }
    Tensor out =
        detail::unary_ng_impl(input_tensor, {unary::UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
    return out;
}

Tensor Abs::invoke(const ComplexTensor& input_tensor, const MemoryConfig& output_mem_config) {
    return ttnn::hypot(input_tensor[0], input_tensor[1], output_mem_config);
}

}  // namespace ttnn::operations::unary_ng
