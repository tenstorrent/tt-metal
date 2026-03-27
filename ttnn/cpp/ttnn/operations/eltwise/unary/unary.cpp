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

using namespace ttnn::operations::unary;

namespace ttnn::detail {

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

}  // namespace ttnn::detail

namespace ttnn {

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
    return ttnn::multiply(
        input_tensor,
        DEG_TO_RAD,
        std::optional(input_tensor.dtype()),
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

Tensor rad2deg(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    constexpr float RAD_TO_DEG = 57.29577951308232f;  // 180/pi
    return ttnn::multiply(
        input_tensor,
        RAD_TO_DEG,
        std::optional(input_tensor.dtype()),
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        std::nullopt,
        std::nullopt,
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
    EltwiseUnaryWithParam bitcast_op(
        UnaryOpType::BITCAST, {static_cast<float>(input_tensor.dtype()), static_cast<float>(output_dtype)});
    return ttnn::detail::unary_impl(input_tensor, {bitcast_op}, memory_config, optional_output_tensor);
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
    UnaryOpType op_type = UnaryOpType::RDIV;
    // Convert rounding_mode to numeric value: 0 = none, 1 = trunc, 2 = floor
    uint32_t rounding_mode_value = !rounding_mode ? 0 : (*rounding_mode == "trunc" ? 1 : 2);
    return ttnn::detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, {value, rounding_mode_value}}}, memory_config, optional_output_tensor);
}

Tensor where_tss(
    const Tensor& condition,
    const ScalarVariant& value_true,
    const ScalarVariant& value_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    Tensor input = condition;
    // Check if we have any float scalars
    bool has_float_scalar = std::holds_alternative<float>(value_true) || std::holds_alternative<float>(value_false);

    // Convert input tensor to float32 only if input is INT32/UINT32 and scalars are float
    if ((condition.dtype() == DataType::INT32 || condition.dtype() == DataType::UINT32) && has_float_scalar) {
        input = ttnn::typecast(condition, DataType::FLOAT32, std::nullopt, std::nullopt, sub_core_grids);
    }
    UnaryOpType op_type = UnaryOpType::WHERE_TSS;
    auto param = std::visit(
        [op_type](const auto& val_true, const auto& val_false) {
            using T = std::decay_t<decltype(val_true)>;
            return EltwiseUnaryWithParam{op_type, std::vector<T>{val_true, val_false}};
        },
        value_true,
        value_false);

    return ttnn::detail::unary_impl(input, {param}, memory_config, optional_output_tensor, sub_core_grids);
}

Tensor xielu(
    const Tensor& input,
    float alpha_p,
    float alpha_n,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input, {UnaryWithParam{UnaryOpType::XIELU, {alpha_p, alpha_n}}}, memory_config, optional_output_tensor);
}

Tensor unary_fmod(
    const Tensor& input_tensor,
    float parameter,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::FMOD, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor);
}

Tensor threshold(
    const Tensor& input_tensor,
    float parameter_a,
    float parameter_b,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::THRESHOLD, {static_cast<float>(parameter_a), static_cast<float>(parameter_b)}}},
        memory_config,
        optional_output_tensor);
}

Tensor round(
    const Tensor& input_tensor,
    const std::optional<int32_t>& parameter,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::ROUND, parameter.value_or(0)}},
        memory_config,
        optional_output_tensor);
}

Tensor identity(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {UnaryWithParam{UnaryOpType::IDENTITY}}, memory_config, optional_output_tensor);
}

Tensor eqz(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {UnaryWithParam{UnaryOpType::EQZ}}, memory_config, optional_output_tensor);
}

Tensor hardmish(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {UnaryWithParam{UnaryOpType::HARDMISH}}, memory_config, optional_output_tensor);
}

Tensor hardshrink(
    const Tensor& input_tensor,
    float lambda,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::HARDSHRINK, static_cast<float>(lambda)}},
        memory_config,
        optional_output_tensor);
}

Tensor logit(
    const Tensor& input_tensor,
    std::optional<float> eps,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::LOGIT, {eps.value_or(-1.0f)}}},
        memory_config,
        optional_output_tensor);
}

Tensor elu(
    const Tensor& input,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input, {UnaryWithParam{UnaryOpType::ELU, static_cast<float>(alpha)}}, memory_config, optional_output_tensor);
}

Tensor hardtanh(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::HARDTANH, {min_val, max_val}}},
        memory_config,
        optional_output_tensor);
}

Tensor softshrink(
    const Tensor& input_tensor,
    float lambda,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::SOFTSHRINK, static_cast<float>(lambda)}},
        memory_config,
        optional_output_tensor);
}

Tensor clamp_tss(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        memory_config,
        optional_output_tensor);
}

Tensor clamp_tss(
    const Tensor& input_tensor,
    int32_t min_val,
    int32_t max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        memory_config,
        optional_output_tensor);
}

Tensor softplus(
    const Tensor& input,
    float beta,
    float threshold,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input, {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}}, memory_config, optional_output_tensor);
}

Tensor tanh(
    const Tensor& input,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool approx) {
    return ttnn::detail::unary_impl(
        input, {UnaryWithParam{UnaryOpType::TANH, static_cast<float>(approx)}}, memory_config, optional_output_tensor);
}

Tensor tanhshrink(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool /*approx*/) {
    return ttnn::detail::unary_impl(
        input_tensor, {UnaryWithParam{UnaryOpType::TANHSHRINK}}, memory_config, optional_output_tensor);
}

Tensor prelu_sfpu(
    const Tensor& input,
    float value,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input, {UnaryWithParam{UnaryOpType::PRELU_SFPU, value}}, memory_config, optional_output_tensor);
}

Tensor selu(
    const Tensor& input_tensor,
    float scale,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {UnaryWithParam{UnaryOpType::SELU, {scale, alpha}}}, memory_config, optional_output_tensor);
}

Tensor swish(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{UnaryOpType::SILU}}, memory_config, optional_output_tensor);
}

Tensor power_iterative(
    const Tensor& input_tensor,
    uint32_t exponent,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::POWER_ITERATIVE, exponent}},
        memory_config,
        optional_output_tensor);
}

Tensor sigmoid(
    const Tensor& input,
    int vector_mode,
    SigmoidMode mode,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    std::vector<EltwiseUnaryWithParam> op_chain;
    switch (mode) {
        case SigmoidMode::FAST_APPROXIMATE:
            op_chain = {UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(vector_mode), 1.0f})};
            break;
        case SigmoidMode::ACCURATE_FAST_EXP:
            op_chain = {
                UnaryWithParam(UnaryOpType::NEG),
                UnaryWithParam(UnaryOpType::EXP, 1.0f),
                UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
                UnaryWithParam(UnaryOpType::RECIP)};
            break;
        case SigmoidMode::ACCURATE: [[fallthrough]];
        default: op_chain = {UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(vector_mode), 0.0f})};
    }
    return ttnn::detail::unary_impl(input, op_chain, memory_config, optional_output_tensor);
}

Tensor sigmoid_accurate(
    const Tensor& input,
    bool fast_and_approximate_mode,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto op_chain =
        fast_and_approximate_mode
            ? std::vector<
                  EltwiseUnaryWithParam>{UnaryWithParam(UnaryOpType::NEG), UnaryWithParam(UnaryOpType::EXP, 1.0f), UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f), UnaryWithParam(UnaryOpType::RECIP)}
            : std::vector<EltwiseUnaryWithParam>{
                  UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(VecMode::RC), 0.0f})};
    return ttnn::detail::unary_impl(input, op_chain, memory_config, optional_output_tensor);
}

Tensor log_sigmoid(
    const Tensor& input,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input, {UnaryWithParam{UnaryOpType::LOGSIGMOID}}, memory_config, optional_output_tensor);
}

Tensor unary_chain(
    const Tensor& input_tensor,
    const std::vector<EltwiseUnaryWithParam>& ops_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(input_tensor, ops_chain, memory_config, optional_output_tensor);
}

template <typename T>
Tensor rsub_sfpu(
    const Tensor& input_tensor,
    T param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{UnaryOpType::RSUB, {param}}}, memory_config, optional_output_tensor);
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
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, (param))},
        memory_config,
        optional_output_tensor);
}

Tensor add_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, (param))},
        memory_config,
        optional_output_tensor);
}

Tensor mul_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam(UnaryOpType::MUL_UNARY_SFPU, (param))},
        memory_config,
        optional_output_tensor);
}

Tensor mul_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam(UnaryOpType::MUL_UNARY_SFPU, (param))},
        memory_config,
        optional_output_tensor);
}

Tensor sub_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, (param)}},
        memory_config,
        optional_output_tensor);
}

Tensor sub_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{UnaryOpType::RSUB, (param)}}, memory_config, optional_output_tensor);
}

Tensor div_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::DIV_UNARY_SFPU, (param)}},
        memory_config,
        optional_output_tensor);
}

Tensor div_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{UnaryOpType::RDIV, (param)}}, memory_config, optional_output_tensor);
}

Tensor unary_with_int32_param(
    UnaryOpType op_type,
    const Tensor& input_tensor,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{op_type, param}}, memory_config, optional_output_tensor);
}

}  // namespace ttnn
