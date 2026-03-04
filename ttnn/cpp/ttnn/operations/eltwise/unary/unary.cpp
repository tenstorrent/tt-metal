// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary.hpp"

#include "common/unary_op_types.hpp"
#include "device/unary_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::operations::unary {

namespace detail {

Tensor unary_impl(
    const Tensor& input_tensor,
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_FATAL(!op_chain.empty(), "Op chain cannot be empty");
    DataType input_dtype = input_tensor.dtype();
    // TYPECAST/BITCAST should always be the last operation in the chain when present; use its output dtype (param 1)
    DataType output_dtype = input_dtype;
    if (op_chain.back().type() == UnaryOpType::TYPECAST || op_chain.back().type() == UnaryOpType::BITCAST) {
        output_dtype = static_cast<DataType>(*op_chain.back().get_param_if<float>(1));
    }
    bool preserve_fp32_precision = input_dtype == DataType::FLOAT32;
    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            output_dtype == DataType::UINT8 or input_dtype == DataType::UINT8 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;
    bool bfp8_pack_precise = (op_chain.back().type() == UnaryOpType::TYPECAST && output_dtype == DataType::BFLOAT8_B);

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    return prim::unary(
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

}  // namespace ttnn::operations::unary

namespace ttnn {

Tensor abs(const operations::unary::ComplexTensor& input_tensor, const MemoryConfig& output_mem_config) {
    return ttnn::hypot(input_tensor[0], input_tensor[1], output_mem_config);
}

ComplexTensor reciprocal(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    Tensor a_plus_b = ttnn::add(input[0], input[1], std::nullopt, output_mem_config);
    Tensor a_minus_b = ttnn::subtract(input[0], input[1], std::nullopt, output_mem_config);
    Tensor asqr_plus_bsqr = ttnn::add(
        ttnn::square(input[0], output_mem_config),
        ttnn::square(input[1], output_mem_config),
        std::nullopt,
        output_mem_config);
    Tensor inv_dr = ttnn::reciprocal(asqr_plus_bsqr, output_mem_config);
    Tensor conj_im = ttnn::multiply(ttnn::neg(input[1], output_mem_config), inv_dr, std::nullopt, output_mem_config);
    Tensor conj_re = ttnn::multiply(input[0], inv_dr, std::nullopt, output_mem_config);
    return operations::complex::ComplexTensor({conj_re, conj_im});
}

Tensor deg2rad(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    constexpr float DEG_TO_RAD = 0.017453292519943295f;  // pi/180
    return operations::binary::BinaryOperation<operations::binary::BinaryOpType::MUL>::invoke(
        input_tensor,
        DEG_TO_RAD,
        input_tensor.dtype(),
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        std::nullopt);
}

Tensor rad2deg(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    constexpr float RAD_TO_DEG = 57.29577951308232f;  // 180/pi
    return operations::binary::BinaryOperation<operations::binary::BinaryOpType::MUL>::invoke(
        input_tensor,
        RAD_TO_DEG,
        input_tensor.dtype(),
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        std::nullopt);
}

Tensor bitcast(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }
    // Use unary infrastructure with BITCAST op type
    // BITCAST uses identity kernel (copy_tile + pack_tile) with output format for both CBs
    operations::unary::EltwiseUnaryWithParam bitcast_op(
        operations::unary::UnaryOpType::BITCAST,
        {static_cast<float>(input_tensor.dtype()), static_cast<float>(output_dtype)});
    return operations::unary::detail::unary_impl(input_tensor, {bitcast_op}, memory_config, optional_output_tensor);
}

Tensor rdiv(
    const Tensor& input_tensor,
    float value,
    const std::optional<std::string>& rounding_mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");
    operations::unary::UnaryOpType op_type = operations::unary::UnaryOpType::RDIV;
    // Convert rounding_mode to numeric value: 0 = none, 1 = trunc, 2 = floor
    uint32_t rounding_mode_value = !rounding_mode ? 0 : (*rounding_mode == "trunc" ? 1 : 2);
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{op_type, {value, rounding_mode_value}}},
        memory_config,
        optional_output_tensor);
}

Tensor where_tss(
    const Tensor& condition,
    const operations::unary::ScalarVariant& value_true,
    const operations::unary::ScalarVariant& value_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    Tensor input = condition;
    // Check if we have any float scalars
    bool has_float_scalar = std::holds_alternative<float>(value_true) || std::holds_alternative<float>(value_false);

    // Convert input tensor to float32 only if input is INT32/UINT32 and scalars are float
    if ((condition.dtype() == DataType::INT32 || condition.dtype() == DataType::UINT32) && has_float_scalar) {
        input = ttnn::typecast(condition, DataType::FLOAT32);
    }
    operations::unary::UnaryOpType op_type = operations::unary::UnaryOpType::WHERE_TSS;
    auto param = std::visit(
        [op_type](const auto& val_true, const auto& val_false) {
            using T = std::decay_t<decltype(val_true)>;
            return operations::unary::EltwiseUnaryWithParam{op_type, std::vector<T>{val_true, val_false}};
        },
        value_true,
        value_false);

    return operations::unary::detail::unary_impl(input, {param}, memory_config, optional_output_tensor);
}

// xIELU (Expanded Integral of the Exponential Linear Unit)
// With beta = 0.5 and eps = -1e-6:
//     x > 0 :  alpha_p * x^2 + beta * x
//     x <= 0:  alpha_n * (expm1(minimum(x, eps))) - (alpha_n * x) + 0.5 * x
Tensor invoke(
    const Tensor& input,
    const float alpha_p,
    const float alpha_n,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::XIELU, {alpha_p, alpha_n}}},
        memory_config,
        optional_output_tensor);
}

}  // namespace ttnn
