// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::unary {

enum class SigmoidMode {
    FAST_APPROXIMATE,
    ACCURATE_FAST_EXP,
    ACCURATE,
};

}  // namespace ttnn::operations::unary

namespace ttnn::operations::unary_ng::detail {

Tensor unary_ng_impl(
    const Tensor& input_tensor,
    const std::vector<unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::operations::unary_ng::detail

namespace ttnn {

#define DECLARE_UNARY_NG_OP(op_name)                                                   \
    Tensor op_name(                                                                    \
        const Tensor& input_tensor,                                                    \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt, \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,            \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

DECLARE_UNARY_NG_OP(abs)
DECLARE_UNARY_NG_OP(neg)
DECLARE_UNARY_NG_OP(acos)
DECLARE_UNARY_NG_OP(asin)
DECLARE_UNARY_NG_OP(asinh)
DECLARE_UNARY_NG_OP(atan)
DECLARE_UNARY_NG_OP(atanh)
DECLARE_UNARY_NG_OP(cos)
DECLARE_UNARY_NG_OP(acosh)
DECLARE_UNARY_NG_OP(cosh)
DECLARE_UNARY_NG_OP(sinh)
DECLARE_UNARY_NG_OP(erfinv)
DECLARE_UNARY_NG_OP(exp2)
DECLARE_UNARY_NG_OP(expm1)
DECLARE_UNARY_NG_OP(gez)
DECLARE_UNARY_NG_OP(gtz)
DECLARE_UNARY_NG_OP(i0)
DECLARE_UNARY_NG_OP(i1)
DECLARE_UNARY_NG_OP(isfinite)
DECLARE_UNARY_NG_OP(isinf)
DECLARE_UNARY_NG_OP(isnan)
DECLARE_UNARY_NG_OP(isneginf)
DECLARE_UNARY_NG_OP(isposinf)
DECLARE_UNARY_NG_OP(lez)
DECLARE_UNARY_NG_OP(logical_not)
DECLARE_UNARY_NG_OP(ltz)
DECLARE_UNARY_NG_OP(nez)
DECLARE_UNARY_NG_OP(reciprocal)
DECLARE_UNARY_NG_OP(relu)
DECLARE_UNARY_NG_OP(relu6)
DECLARE_UNARY_NG_OP(sign)
DECLARE_UNARY_NG_OP(signbit)
DECLARE_UNARY_NG_OP(silu)
DECLARE_UNARY_NG_OP(sin)
DECLARE_UNARY_NG_OP(square)
DECLARE_UNARY_NG_OP(tan)
DECLARE_UNARY_NG_OP(tiled_prod)
DECLARE_UNARY_NG_OP(bitwise_not)
DECLARE_UNARY_NG_OP(alt_complex_rotate90)
DECLARE_UNARY_NG_OP(floor)
DECLARE_UNARY_NG_OP(ceil)
DECLARE_UNARY_NG_OP(trunc)
DECLARE_UNARY_NG_OP(frac)
DECLARE_UNARY_NG_OP(hardsigmoid)
DECLARE_UNARY_NG_OP(hardswish)
DECLARE_UNARY_NG_OP(softsign)
DECLARE_UNARY_NG_OP(cbrt)
DECLARE_UNARY_NG_OP(lgamma)
DECLARE_UNARY_NG_OP(eqz)
DECLARE_UNARY_NG_OP(hardmish)
DECLARE_UNARY_NG_OP(identity)
DECLARE_UNARY_NG_OP(log_sigmoid)
DECLARE_UNARY_NG_OP(swish)
DECLARE_UNARY_NG_OP(tanhshrink)

#undef DECLARE_UNARY_NG_OP

#define DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(op_name)                    \
    Tensor op_name(                                                                    \
        const Tensor& input_tensor,                                                    \
        bool fast_and_approximate_mode = false,                                        \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt, \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,            \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(exp)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(erf)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(erfc)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(gelu)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(log)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(log10)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(log2)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(log1p)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(sqrt)
DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(mish)

#undef DECLARE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE

#define DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(op_name)                                  \
    Tensor op_name(                                                                    \
        const Tensor& input_tensor,                                                    \
        float parameter,                                                               \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt, \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,            \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(heaviside)
DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(leaky_relu)
DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(relu_max)
DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(relu_min)
DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(unary_remainder)
DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(celu)
DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(rpow)
DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(unary_fmod)
DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM(prelu_sfpu)

#undef DECLARE_UNARY_NG_OP_WITH_FLOAT_PARAM

Tensor hardshrink(
    const Tensor& input_tensor,
    float parameter = 0.5f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor elu(
    const Tensor& input_tensor,
    float parameter = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor softshrink(
    const Tensor& input_tensor,
    float parameter = 0.5f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

#define DECLARE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS(op_name)                             \
    Tensor op_name(                                                                    \
        const Tensor& input_tensor,                                                    \
        float parameter_a,                                                             \
        float parameter_b,                                                             \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt, \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,            \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

DECLARE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS(threshold)
DECLARE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS(softplus)

#undef DECLARE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS

Tensor hardtanh(
    const Tensor& input_tensor,
    float parameter_a = -1.0f,
    float parameter_b = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor selu(
    const Tensor& input_tensor,
    float parameter_a = 1.050700987f,
    float parameter_b = 1.673263242f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

#define DECLARE_UNARY_NG_OP_SCALAR_VARIANT(op_name)                                    \
    Tensor op_name(                                                                    \
        const Tensor& input_tensor,                                                    \
        operations::unary::ScalarVariant parameter,                                    \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt, \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,            \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

DECLARE_UNARY_NG_OP_SCALAR_VARIANT(fill)
DECLARE_UNARY_NG_OP_SCALAR_VARIANT(power)
DECLARE_UNARY_NG_OP_SCALAR_VARIANT(gt_unary)
DECLARE_UNARY_NG_OP_SCALAR_VARIANT(lt_unary)
DECLARE_UNARY_NG_OP_SCALAR_VARIANT(ne_unary)
DECLARE_UNARY_NG_OP_SCALAR_VARIANT(eq_unary)
DECLARE_UNARY_NG_OP_SCALAR_VARIANT(ge_unary)
DECLARE_UNARY_NG_OP_SCALAR_VARIANT(le_unary)

#undef DECLARE_UNARY_NG_OP_SCALAR_VARIANT

Tensor abs(const ComplexTensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config);
ComplexTensor reciprocal(const ComplexTensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

// -----------------------------------------------------------------------------
// Ops with unique parameter signatures (manual declarations)
// -----------------------------------------------------------------------------

Tensor xielu(
    const Tensor& input,
    float alpha_p = 0.8f,
    float alpha_n = 0.8f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor round(
    const Tensor& input_tensor,
    const std::optional<int32_t>& parameter = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor logit(
    const Tensor& input_tensor,
    std::optional<float> eps = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor deg2rad(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor rad2deg(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor clamp_tss(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor clamp_tss(
    const Tensor& input_tensor,
    int32_t min_val,
    int32_t max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor tanh(
    const Tensor& input,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    bool approx = false,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor where_tss(
    const Tensor& condition,
    const operations::unary::ScalarVariant& value_true,
    const operations::unary::ScalarVariant& value_false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor bitcast(
    const Tensor& input_tensor,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor rdiv(
    const Tensor& input_tensor,
    float value,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor power_iterative(
    const Tensor& input_tensor,
    uint32_t exponent,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor sigmoid(
    const Tensor& input,
    int vector_mode = static_cast<int32_t>(operations::unary::VecMode::RC),
    operations::unary::SigmoidMode mode = operations::unary::SigmoidMode::ACCURATE,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor sigmoid_accurate(
    const Tensor& input,
    bool fast_and_approximate_mode = false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor unary_chain(
    const Tensor& input_tensor,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& ops_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

template <typename T>
Tensor rsub_sfpu(
    const Tensor& input_tensor,
    T param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor add_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor add_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor mul_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor mul_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor sub_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor sub_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor div_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor div_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& input_tensor,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn
