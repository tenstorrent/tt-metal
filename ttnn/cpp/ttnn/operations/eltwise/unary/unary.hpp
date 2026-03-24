// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/eltwise/unary_ng/unary_ng.hpp"

namespace ttnn {
namespace operations::unary {

enum class SigmoidMode {
    FAST_APPROXIMATE,
    ACCURATE_FAST_EXP,
    ACCURATE,
};

}  // namespace operations::unary

namespace detail {

Tensor unary_impl(
    const Tensor& input_tensor,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace detail

#define REGISTER_UNARY_OPERATION(op_name, op_type)                                        \
    inline Tensor op_name(                                                                \
        const Tensor& input_tensor,                                                       \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,    \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,               \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {               \
        return ttnn::detail::unary_impl(                                                  \
            input_tensor,                                                                 \
            {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::op_type}}, \
            memory_config,                                                                \
            optional_output_tensor,                                                       \
            sub_core_grids);                                                              \
    }

#define REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(op_name, op_type)                         \
    inline Tensor op_name(                                                                                \
        const Tensor& input_tensor,                                                                       \
        bool fast_and_approximate_mode = false,                                                           \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,                    \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,                               \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {                               \
        return ttnn::detail::unary_impl(                                                                  \
            input_tensor,                                                                                 \
            {operations::unary::UnaryWithParam{                                                           \
                operations::unary::UnaryOpType::op_type, static_cast<float>(fast_and_approximate_mode)}}, \
            memory_config,                                                                                \
            optional_output_tensor,                                                                       \
            sub_core_grids);                                                                              \
    }

#define REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(op_name, op_type)                   \
    inline Tensor op_name(                                                                \
        const Tensor& input_tensor,                                                       \
        float parameter,                                                                  \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,    \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,               \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {               \
        return ttnn::detail::unary_impl(                                                  \
            input_tensor,                                                                 \
            {operations::unary::UnaryWithParam{                                           \
                operations::unary::UnaryOpType::op_type, static_cast<float>(parameter)}}, \
            memory_config,                                                                \
            optional_output_tensor,                                                       \
            sub_core_grids);                                                              \
    }

#define UNARY_OP_SCALAR_VARIANT(op_name, op_type)                                                                 \
    inline Tensor op_name(                                                                                        \
        const Tensor& input_tensor,                                                                               \
        operations::unary::ScalarVariant parameter,                                                               \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,                            \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {                                     \
        return std::visit(                                                                                        \
            [&](auto param) {                                                                                     \
                return ttnn::detail::unary_impl(                                                                  \
                    input_tensor,                                                                                 \
                    {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::op_type, (param)}}, \
                    memory_config,                                                                                \
                    optional_output_tensor);                                                                      \
            },                                                                                                    \
            parameter);                                                                                           \
    }

// Unaries with fast_and_approximate_mode
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(exp, EXP)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erf, ERF)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erfc, ERFC)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(gelu, GELU)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log, LOG)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log10, LOG10)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log2, LOG2)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log1p, LOG1P)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt, RSQRT)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(sqrt, SQRT)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(mish, MISH)

REGISTER_UNARY_OPERATION(lgamma, LGAMMA)

// Unaries with float parameter
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(heaviside, HEAVISIDE)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(leaky_relu, LEAKY_RELU)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_max, RELU_MAX)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_min, RELU_MIN)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(unary_remainder, REMAINDER)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(celu, CELU)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rpow, RPOW)

UNARY_OP_SCALAR_VARIANT(fill, FILL)
UNARY_OP_SCALAR_VARIANT(power, POWER)
UNARY_OP_SCALAR_VARIANT(gt_unary, UNARY_GT)
UNARY_OP_SCALAR_VARIANT(lt_unary, UNARY_LT)
UNARY_OP_SCALAR_VARIANT(ne_unary, UNARY_NE)
UNARY_OP_SCALAR_VARIANT(eq_unary, UNARY_EQ)
UNARY_OP_SCALAR_VARIANT(ge_unary, UNARY_GE)
UNARY_OP_SCALAR_VARIANT(le_unary, UNARY_LE)

// -----------------------------------------------------------------------------
// Functions defined without macros
// -----------------------------------------------------------------------------

Tensor xielu(
    const Tensor& input,
    float alpha_p = 0.8f,
    float alpha_n = 0.8f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

ComplexTensor reciprocal(const ComplexTensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

Tensor unary_fmod(
    const Tensor& input_tensor,
    float parameter,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor threshold(
    const Tensor& input_tensor,
    float parameter_a,
    float parameter_b,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor round(
    const Tensor& input_tensor,
    const std::optional<int32_t>& parameter = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor identity(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor eqz(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor hardmish(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor hardshrink(
    const Tensor& input_tensor,
    float lambda = 0.5f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor logit(
    const Tensor& input_tensor,
    std::optional<float> eps = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor elu(
    const Tensor& input,
    float alpha = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor hardtanh(
    const Tensor& input_tensor,
    float min_val = -1.0f,
    float max_val = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor softshrink(
    const Tensor& input_tensor,
    float lambda = 0.5f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor deg2rad(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor rad2deg(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor clamp_tss(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor clamp_tss(
    const Tensor& input_tensor,
    int32_t min_val,
    int32_t max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor softplus(
    const Tensor& input,
    float beta,
    float threshold,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor tanh(
    const Tensor& input,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    bool approx = false);

Tensor tanhshrink(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    bool /*approx*/ = false);

Tensor prelu_sfpu(
    const Tensor& input,
    float value,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor where_tss(
    const Tensor& condition,
    const operations::unary::ScalarVariant& value_true,
    const operations::unary::ScalarVariant& value_false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor selu(
    const Tensor& input_tensor,
    float scale = 1.050700987f,
    float alpha = 1.673263242f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor bitcast(
    const Tensor& input_tensor,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor rdiv(
    const Tensor& input_tensor,
    float value,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor swish(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor power_iterative(
    const Tensor& input_tensor,
    uint32_t exponent,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor sigmoid(
    const Tensor& input,
    int vector_mode = (int32_t)operations::unary::VecMode::RC,
    operations::unary::SigmoidMode mode = operations::unary::SigmoidMode::ACCURATE,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor sigmoid_accurate(
    const Tensor& input,
    bool fast_and_approximate_mode = false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor log_sigmoid(
    const Tensor& input,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor unary_chain(
    const Tensor& input_tensor,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& ops_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

template <typename T>
Tensor rsub_sfpu(
    const Tensor& input_tensor,
    T param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor add_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor add_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor mul_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor mul_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor sub_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor sub_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor div_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor div_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& input_tensor,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
