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
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::unary {

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryTSVariant<unary_op_type>::invoke(
    const Tensor& input_tensor,
    ScalarVariant parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return std::visit(
        [&](auto param) {
            return detail::unary_impl(
                input_tensor, {EltwiseUnaryWithParam{unary_op_type, (param)}}, memory_config, optional_output_tensor);
        },
        parameter);
}

template struct ExecuteUnaryTSVariant<UnaryOpType::MINIMUM>;
template struct ExecuteUnaryTSVariant<UnaryOpType::MAXIMUM>;

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithFloatParameter<unary_op_type>::invoke(
    const Tensor& input_tensor,
    const float parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return detail::unary_impl(
        input_tensor,
        {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

template struct ExecuteUnaryWithFloatParameter<UnaryOpType::FMOD>;

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

Tensor abs(const ComplexTensor& input_tensor, const MemoryConfig& output_mem_config) {
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
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
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

    return operations::unary::detail::unary_impl(input, {param}, memory_config, optional_output_tensor, sub_core_grids);
}

Tensor xielu(
    const Tensor& input,
    float alpha_p,
    float alpha_n,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::XIELU, {alpha_p, alpha_n}}},
        memory_config,
        optional_output_tensor);
}

Tensor unary_fmod(
    const Tensor& input_tensor,
    float parameter,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::FMOD, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor);
}

Tensor threshold(
    const Tensor& input_tensor,
    float parameter_a,
    float parameter_b,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{
            operations::unary::UnaryOpType::THRESHOLD,
            {static_cast<float>(parameter_a), static_cast<float>(parameter_b)}}},
        memory_config,
        optional_output_tensor);
}

Tensor round(
    const Tensor& input_tensor,
    const std::optional<int32_t>& parameter,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::ROUND, parameter.value_or(0)}},
        memory_config,
        optional_output_tensor);
}

Tensor identity(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::IDENTITY}},
        memory_config,
        optional_output_tensor);
}

Tensor abs(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto op_type = input_tensor.dtype() == tt::tt_metal::DataType::INT32 ? operations::unary::UnaryOpType::ABS_INT32
                                                                         : operations::unary::UnaryOpType::ABS;
    return operations::unary::detail::unary_impl(
        input_tensor, {operations::unary::UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor eqz(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::EQZ}},
        memory_config,
        optional_output_tensor);
}

Tensor hardmish(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDMISH}},
        memory_config,
        optional_output_tensor);
}

Tensor hardshrink(
    const Tensor& input_tensor,
    float lambda,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDSHRINK, static_cast<float>(lambda)}},
        memory_config,
        optional_output_tensor);
}

Tensor logit(
    const Tensor& input_tensor,
    std::optional<float> eps,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOGIT, {eps.value_or(-1.0f)}}},
        memory_config,
        optional_output_tensor);
}

Tensor elu(
    const Tensor& input,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ELU, static_cast<float>(alpha)}},
        memory_config,
        optional_output_tensor);
}

Tensor hardtanh(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDTANH, {min_val, max_val}}},
        memory_config,
        optional_output_tensor);
}

Tensor softshrink(
    const Tensor& input_tensor,
    float lambda,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SOFTSHRINK, static_cast<float>(lambda)}},
        memory_config,
        optional_output_tensor);
}

Tensor clamp_tss(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        memory_config,
        optional_output_tensor);
}

Tensor clamp_tss(
    const Tensor& input_tensor,
    int32_t min_val,
    int32_t max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        memory_config,
        optional_output_tensor);
}

Tensor softplus(
    const Tensor& input,
    float beta,
    float threshold,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SOFTPLUS, {beta, threshold}}},
        memory_config,
        optional_output_tensor);
}

Tensor tanh(
    const Tensor& input,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool approx) {
    return operations::unary::detail::unary_impl(
        input,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TANH, static_cast<float>(approx)}},
        memory_config,
        optional_output_tensor);
}

Tensor tanhshrink(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool /*approx*/) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TANHSHRINK}},
        memory_config,
        optional_output_tensor);
}

Tensor prelu_sfpu(
    const Tensor& input,
    float value,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::PRELU_SFPU, value}},
        memory_config,
        optional_output_tensor);
}

Tensor selu(
    const Tensor& input_tensor,
    float scale,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SELU, {scale, alpha}}},
        memory_config,
        optional_output_tensor);
}

Tensor swish(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::SILU}},
        memory_config,
        optional_output_tensor);
}

Tensor power_iterative(
    const Tensor& input_tensor,
    uint32_t exponent,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::POWER_ITERATIVE, exponent}},
        memory_config,
        optional_output_tensor);
}

Tensor sigmoid(
    const Tensor& input,
    int vector_mode,
    operations::unary::SigmoidMode mode,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    std::vector<operations::unary::EltwiseUnaryWithParam> op_chain;
    switch (mode) {
        case operations::unary::SigmoidMode::FAST_APPROXIMATE:
            op_chain = {operations::unary::UnaryWithParam(
                operations::unary::UnaryOpType::SIGMOID, {static_cast<float>(vector_mode), 1.0f})};
            break;
        case operations::unary::SigmoidMode::ACCURATE_FAST_EXP:
            op_chain = {
                operations::unary::UnaryWithParam(operations::unary::UnaryOpType::NEG),
                operations::unary::UnaryWithParam(operations::unary::UnaryOpType::EXP, 1.0f),
                operations::unary::UnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, 1.0f),
                operations::unary::UnaryWithParam(operations::unary::UnaryOpType::RECIP)};
            break;
        case operations::unary::SigmoidMode::ACCURATE: [[fallthrough]];
        default:
            op_chain = {operations::unary::UnaryWithParam(
                operations::unary::UnaryOpType::SIGMOID, {static_cast<float>(vector_mode), 0.0f})};
    }
    return operations::unary::detail::unary_impl(input, op_chain, memory_config, optional_output_tensor);
}

Tensor sigmoid_accurate(
    const Tensor& input,
    bool fast_and_approximate_mode,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto
        op_chain = fast_and_approximate_mode
                       ? std::vector<
                             operations::unary::EltwiseUnaryWithParam>{operations::unary::UnaryWithParam(operations::unary::UnaryOpType::NEG), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::EXP, 1.0f), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, 1.0f), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::RECIP)}
                       : std::vector<operations::unary::EltwiseUnaryWithParam>{operations::unary::UnaryWithParam(
                             operations::unary::UnaryOpType::SIGMOID,
                             {static_cast<float>(operations::unary::VecMode::RC), 0.0f})};
    return operations::unary::detail::unary_impl(input, op_chain, memory_config, optional_output_tensor);
}

Tensor log_sigmoid(
    const Tensor& input,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOGSIGMOID}},
        memory_config,
        optional_output_tensor);
}

Tensor unary_chain(
    const Tensor& input_tensor,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& ops_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(input_tensor, ops_chain, memory_config, optional_output_tensor);
}

template <typename T>
Tensor rsub_sfpu(
    const Tensor& input_tensor,
    T param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSUB, {param}}},
        memory_config,
        optional_output_tensor);
}

template Tensor ttnn::rsub_sfpu<float>(
    const Tensor&, float, const std::optional<tt::tt_metal::MemoryConfig>&, const std::optional<Tensor>&);

template Tensor ttnn::rsub_sfpu<int>(
    const Tensor&, int, const std::optional<tt::tt_metal::MemoryConfig>&, const std::optional<Tensor>&);

Tensor add_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, (param))},
        memory_config,
        optional_output_tensor);
}

Tensor add_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, (param))},
        memory_config,
        optional_output_tensor);
}

Tensor mul_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::MUL_UNARY_SFPU, (param))},
        memory_config,
        optional_output_tensor);
}

Tensor mul_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::MUL_UNARY_SFPU, (param))},
        memory_config,
        optional_output_tensor);
}

Tensor sub_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::SUB_UNARY_SFPU, (param)}},
        memory_config,
        optional_output_tensor);
}

Tensor sub_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSUB, (param)}},
        memory_config,
        optional_output_tensor);
}

Tensor div_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::DIV_UNARY_SFPU, (param)}},
        memory_config,
        optional_output_tensor);
}

Tensor div_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RDIV, (param)}},
        memory_config,
        optional_output_tensor);
}

Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& input_tensor,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::unary::detail::unary_impl(
        input_tensor,
        {operations::unary::EltwiseUnaryWithParam{op_type, param}},
        memory_config,
        optional_output_tensor);
}

}  // namespace ttnn
