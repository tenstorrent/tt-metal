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
    const Tensor& t, float p, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::FMOD, static_cast<float>(p)}}, m, o);
}

Tensor threshold(
    const Tensor& t,
    float pa,
    float pb,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{
            operations::unary::UnaryOpType::THRESHOLD, {static_cast<float>(pa), static_cast<float>(pb)}}},
        m,
        o);
}

Tensor round(
    const Tensor& t,
    const std::optional<int32_t>& p,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::ROUND, p.value_or(0)}}, m, o);
}

Tensor identity(const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::IDENTITY}}, m, o);
}

Tensor abs(const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    auto op_type = t.dtype() == tt::tt_metal::DataType::INT32 ? operations::unary::UnaryOpType::ABS_INT32
                                                              : operations::unary::UnaryOpType::ABS;
    return operations::unary::detail::unary_impl(t, {operations::unary::UnaryWithParam{op_type}}, m, o);
}

Tensor eqz(const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::EQZ}}, m, o);
}

Tensor hardmish(const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDMISH}}, m, o);
}

Tensor hardshrink(
    const Tensor& t, float lambda, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDSHRINK, static_cast<float>(lambda)}},
        m,
        o);
}

Tensor logit(
    const Tensor& t,
    std::optional<float> eps,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOGIT, {eps.value_or(-1.0f)}}}, m, o);
}

Tensor elu(
    const Tensor& t, float alpha, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ELU, static_cast<float>(alpha)}}, m, o);
}

Tensor hardtanh(
    const Tensor& t,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDTANH, {min_val, max_val}}}, m, o);
}

Tensor softshrink(
    const Tensor& t, float lambda, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SOFTSHRINK, static_cast<float>(lambda)}},
        m,
        o);
}

Tensor clamp_tss(
    const Tensor& t,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::CLAMP_TSS, {min_val, max_val}}}, m, o);
}

Tensor clamp_tss(
    const Tensor& t,
    int32_t min_val,
    int32_t max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        m,
        o);
}

Tensor softplus(
    const Tensor& t,
    float beta,
    float threshold,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SOFTPLUS, {beta, threshold}}}, m, o);
}

Tensor tanh(
    const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o, bool approx) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TANH, static_cast<float>(approx)}}, m, o);
}

Tensor tanhshrink(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o,
    bool /*approx*/) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TANHSHRINK}}, m, o);
}

Tensor prelu_sfpu(
    const Tensor& t, float value, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::PRELU_SFPU, value}}, m, o);
}

Tensor selu(
    const Tensor& t,
    float scale,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SELU, {scale, alpha}}}, m, o);
}

Tensor swish(const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::SILU}}, m, o);
}

Tensor power_iterative(
    const Tensor& t,
    uint32_t exponent,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::POWER_ITERATIVE, exponent}}, m, o);
}

Tensor sigmoid(
    const Tensor& t,
    int vector_mode,
    operations::unary::SigmoidMode mode,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
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
    return operations::unary::detail::unary_impl(t, op_chain, m, o);
}

Tensor sigmoid_accurate(
    const Tensor& t,
    bool fast_and_approximate_mode,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    auto
        op_chain = fast_and_approximate_mode
                       ? std::vector<
                             operations::unary::EltwiseUnaryWithParam>{operations::unary::UnaryWithParam(operations::unary::UnaryOpType::NEG), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::EXP, 1.0f), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, 1.0f), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::RECIP)}
                       : std::vector<operations::unary::EltwiseUnaryWithParam>{operations::unary::UnaryWithParam(
                             operations::unary::UnaryOpType::SIGMOID,
                             {static_cast<float>(operations::unary::VecMode::RC), 0.0f})};
    return operations::unary::detail::unary_impl(t, op_chain, m, o);
}

Tensor log_sigmoid(
    const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOGSIGMOID}}, m, o);
}

Tensor unary_chain(
    const Tensor& t,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& chain,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(t, chain, m, o);
}

template <typename T>
Tensor rsub_sfpu(
    const Tensor& t, T param, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSUB, {param}}}, m, o);
}

template Tensor ttnn::rsub_sfpu<float>(
    const Tensor&, float, const std::optional<tt::tt_metal::MemoryConfig>&, const std::optional<Tensor>&);

template Tensor ttnn::rsub_sfpu<int>(
    const Tensor&, int, const std::optional<tt::tt_metal::MemoryConfig>&, const std::optional<Tensor>&);

Tensor add_sfpu(
    const Tensor& t, float p, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, (p))}, m, o);
}

Tensor add_sfpu(
    float p, const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, (p))}, m, o);
}

Tensor mul_sfpu(
    const Tensor& t, float p, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::MUL_UNARY_SFPU, (p))}, m, o);
}

Tensor mul_sfpu(
    float p, const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::MUL_UNARY_SFPU, (p))}, m, o);
}

Tensor sub_sfpu(
    const Tensor& t, float p, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::SUB_UNARY_SFPU, (p)}}, m, o);
}

Tensor sub_sfpu(
    float p, const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSUB, (p)}}, m, o);
}

Tensor div_sfpu(
    const Tensor& t, float p, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::DIV_UNARY_SFPU, (p)}}, m, o);
}

Tensor div_sfpu(
    float p, const Tensor& t, const std::optional<tt::tt_metal::MemoryConfig>& m, const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RDIV, (p)}}, m, o);
}

Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& t,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& m,
    const std::optional<Tensor>& o) {
    return operations::unary::detail::unary_impl(t, {operations::unary::EltwiseUnaryWithParam{op_type, param}}, m, o);
}

}  // namespace ttnn
