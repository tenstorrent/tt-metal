// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn {

namespace operations {

namespace unary {

struct UnaryWithParam;

template <UnaryOpType... unary_op_types>
struct ExecuteUnaryInvokeResult {
    using type = ComplexTensor;
};

template <>
struct ExecuteUnaryInvokeResult<UnaryOpType::ABS> {
    using type = Tensor;
};

template <UnaryOpType... unary_op_types>
struct ExecuteUnary {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static typename ExecuteUnaryInvokeResult<unary_op_types...>::type invoke(
        const ComplexTensor& input_tensor, const MemoryConfig& memory_config);
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFastAndApproximateMode {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFloatParameter {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Sigmoid_accurate {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Unary_chain {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const std::vector<UnaryWithParam>& ops_chain,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const std::vector<UnaryWithParam>& ops_chain,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Softplus {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input,
        const float beta,
        const float threshold,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input,
        const float beta,
        const float threshold,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Dropout {
    static Tensor invoke(
        const Tensor& input,
        const uint32_t seed,
        const uint32_t probability,
        const uint32_t scale,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input,
        const uint32_t seed,
        const uint32_t probability,
        const uint32_t scale,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Identity {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type, typename T = int32_t>
struct ExecuteUnaryWithIntegerParameter {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        T parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        T parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type, typename T = float>
struct SymmetricBinop {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        T param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        T param,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        T param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        T param,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type>
struct AsymmetricBinop {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        float param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        float param,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        float param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        float param,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace unary
}  // namespace operations

#define REGISTER_UNARY_OPERATION(operation_name, operation_type)                  \
    constexpr auto operation_name = ttnn::register_operation_with_auto_launch_op< \
        "ttnn::" #operation_name,                                                 \
        ttnn::operations::unary::ExecuteUnary<ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_OVERLOAD(operation_name, operation_type) \
    constexpr auto operation_name = ttnn::register_operation<             \
        "ttnn::" #operation_name,                                         \
        ttnn::operations::unary::ExecuteUnary<ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(operation_name, operation_type) \
    constexpr auto operation_name = ttnn::register_operation_with_auto_launch_op<               \
        "ttnn::" #operation_name,                                                               \
        ttnn::operations::unary::ExecuteUnaryWithFastAndApproximateMode<                        \
            ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(operation_name, operation_type) \
    constexpr auto operation_name = ttnn::register_operation_with_auto_launch_op<     \
        "ttnn::" #operation_name,                                                     \
        ttnn::operations::unary::ExecuteUnaryWithFloatParameter<                      \
            ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(operation_name, operation_type, data_type) \
    constexpr auto operation_name = ttnn::register_operation_with_auto_launch_op<                  \
        "ttnn::" #operation_name,                                                                  \
        ttnn::operations::unary::                                                                  \
            ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::operation_type, data_type>>();

REGISTER_UNARY_OPERATION_OVERLOAD(abs, ABS);
REGISTER_UNARY_OPERATION(acos, ACOS);
REGISTER_UNARY_OPERATION(asin, ASIN);
REGISTER_UNARY_OPERATION(atan, ATAN);
REGISTER_UNARY_OPERATION(cos, COS);
REGISTER_UNARY_OPERATION(erfinv, ERFINV);
REGISTER_UNARY_OPERATION(exp2, EXP2);
REGISTER_UNARY_OPERATION(expm1, EXPM1);
REGISTER_UNARY_OPERATION(eqz, EQZ);
REGISTER_UNARY_OPERATION(floor, FLOOR);
REGISTER_UNARY_OPERATION(ceil, CEIL);
REGISTER_UNARY_OPERATION(gez, GEZ);
REGISTER_UNARY_OPERATION(gtz, GTZ);
REGISTER_UNARY_OPERATION(i0, I0);
REGISTER_UNARY_OPERATION(isfinite, ISFINITE);
REGISTER_UNARY_OPERATION(isinf, ISINF);
REGISTER_UNARY_OPERATION(isnan, ISNAN);
REGISTER_UNARY_OPERATION(isneginf, ISNEGINF);
REGISTER_UNARY_OPERATION(isposinf, ISPOSINF);
REGISTER_UNARY_OPERATION(lez, LEZ);
REGISTER_UNARY_OPERATION(log, LOG);
REGISTER_UNARY_OPERATION(log10, LOG10);
REGISTER_UNARY_OPERATION(log2, LOG2);
REGISTER_UNARY_OPERATION(logical_not, LOGICAL_NOT_UNARY);
REGISTER_UNARY_OPERATION(ltz, LTZ);
REGISTER_UNARY_OPERATION(neg, NEG);
REGISTER_UNARY_OPERATION(nez, NEZ);
REGISTER_UNARY_OPERATION_OVERLOAD(reciprocal, RECIP);
REGISTER_UNARY_OPERATION(relu, RELU);
REGISTER_UNARY_OPERATION(relu6, RELU6);
REGISTER_UNARY_OPERATION(sigmoid, SIGMOID);
REGISTER_UNARY_OPERATION(sign, SIGN);
REGISTER_UNARY_OPERATION(signbit, SIGNBIT);
REGISTER_UNARY_OPERATION(silu, SILU);
REGISTER_UNARY_OPERATION(sin, SIN);
REGISTER_UNARY_OPERATION(sqrt, SQRT);
REGISTER_UNARY_OPERATION(square, SQUARE);
REGISTER_UNARY_OPERATION(tan, TAN);
REGISTER_UNARY_OPERATION(tanh, TANH);
REGISTER_UNARY_OPERATION(tiled_prod, TILED_PROD);

constexpr auto log_sigmoid = ttnn::register_operation_with_auto_launch_op<
    "ttnn::log_sigmoid",
    ttnn::operations::unary::
        ExecuteUnary<ttnn::operations::unary::UnaryOpType::SIGMOID, ttnn::operations::unary::UnaryOpType::LOG>>();

// Unaries with fast_and_approximate_mode
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(exp, EXP);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erf, ERF);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erfc, ERFC);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(gelu, GELU);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt, RSQRT);

// Unaries with float parameter
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(elu, ELU);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rsub, RSUB);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(heaviside, HEAVISIDE);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(leaky_relu, LEAKY_RELU);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_max, RELU_MAX);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_min, RELU_MIN);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(unary_remainder, REMAINDER);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(unary_fmod, FMOD);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(gt_unary, UNARY_GT);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(lt_unary, UNARY_LT);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(ne_unary, UNARY_NE);

// Unaries with integer parameter
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(power, POWER, uint32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_left_shift, LEFT_SHIFT, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_right_shift, RIGHT_SHIFT, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_and, BITWISE_AND, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_or, BITWISE_OR, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_xor, BITWISE_XOR, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_not, BITWISE_NOT, int32_t);

// Other unaries
constexpr auto identity =
    ttnn::register_operation_with_auto_launch_op<"ttnn::identity", ttnn::operations::unary::Identity>();
constexpr auto softplus =
    ttnn::register_operation_with_auto_launch_op<"ttnn::softplus", ttnn::operations::unary::Softplus>();
constexpr auto dropout =
    ttnn::register_operation_with_auto_launch_op<"ttnn::dropout", ttnn::operations::unary::Dropout>();
constexpr auto sigmoid_accurate =
    ttnn::register_operation_with_auto_launch_op<"ttnn::sigmoid_accurate", ttnn::operations::unary::Sigmoid_accurate>();
constexpr auto unary_chain =
    ttnn::register_operation_with_auto_launch_op<"ttnn::unary_chain", ttnn::operations::unary::Unary_chain>();

constexpr auto add_sfpu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::add_sfpu",
    ttnn::operations::unary::SymmetricBinop<ttnn::operations::unary::UnaryOpType::ADD_UNARY_SFPU>>();
constexpr auto mul_sfpu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::mul_sfpu",
    ttnn::operations::unary::SymmetricBinop<ttnn::operations::unary::UnaryOpType::MUL_UNARY_SFPU>>();

constexpr auto sub_sfpu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::sub_sfpu",
    ttnn::operations::unary::AsymmetricBinop<
        ttnn::operations::unary::UnaryOpType::SUB_UNARY_SFPU,
        ttnn::operations::unary::UnaryOpType::RSUB>>();
constexpr auto div_sfpu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::div_sfpu",
    ttnn::operations::unary::AsymmetricBinop<
        ttnn::operations::unary::UnaryOpType::DIV_UNARY_SFPU,
        ttnn::operations::unary::UnaryOpType::RDIV>>();
}  // namespace ttnn
