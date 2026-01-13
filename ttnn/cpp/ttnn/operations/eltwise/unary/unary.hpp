// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {
namespace operations::unary {

template <UnaryOpType... unary_op_types>
struct ExecuteUnary {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static ComplexTensor invoke(const ComplexTensor& input_tensor, const MemoryConfig& memory_config);
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFastAndApproximateMode {
    static Tensor invoke(
        const Tensor& input_tensor,
        bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithVectorAndFastAndApproximateMode {
    static Tensor invoke(
        const Tensor& input_tensor,
        int vector_mode = (int32_t)VecMode::RC,
        bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFloatParameter {
    static Tensor invoke(
        const Tensor& input_tensor,
        float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithTwoFloatParameter {
    static Tensor invoke(
        const Tensor& input_tensor,
        float parameter_a,
        float parameter_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

// Tensor - scalar variant with scalars of float/int32_t/uint32_t
template <UnaryOpType unary_op_type>
struct ExecuteUnaryTSVariant {
    static Tensor invoke(
        const Tensor& input_tensor,
        ScalarVariant parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct LogSigmoid {
    static Tensor invoke(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Sigmoid_accurate {
    static Tensor invoke(
        const Tensor& input,
        bool fast_and_approximate_mode = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};
struct Unary_chain {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::vector<EltwiseUnaryWithParam>& ops_chain,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Selu {
    static Tensor invoke(
        const Tensor& input_tensor,
        float scale = 1.050700987,
        float alpha = 1.673263242,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Bitcast {
    static Tensor invoke(
        const Tensor& input_tensor,
        const DataType& output_dtype,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Softplus {
    static Tensor invoke(
        const Tensor& input,
        float beta,
        float threshold,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Prelu {
    static Tensor invoke(
        const Tensor& input,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Identity {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Abs {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(const ComplexTensor& input_tensor, const MemoryConfig& output_mem_config);
};

struct Eqz {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Frac {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Ceil {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};
struct Dropout {
    static Tensor invoke(
        const Tensor& input,
        uint32_t seed,
        float probability,
        float scale,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type, typename T = int32_t>
struct ExecuteUnaryWithIntegerParameter {
    static Tensor invoke(
        const Tensor& input_tensor,
        T parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type, typename T = int32_t>
struct ExecuteUnaryWithOptionalIntegerParameter {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<T>& parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type, typename T = float>
struct SymmetricBinop {
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

struct Mish {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Tanhshrink {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        bool approx = false);
};

struct Hardmish {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Hardshrink {
    static Tensor invoke(
        const Tensor& input_tensor,
        float lambda = 0.5f,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Elu {
    static Tensor invoke(
        const Tensor& input,
        float alpha = 1.0f,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Hardtanh {
    static Tensor invoke(
        const Tensor& input_tensor,
        float min_val = -1.0f,
        float max_val = 1.0f,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Softshrink {
    static Tensor invoke(
        const Tensor& input_tensor,
        float lambda = 0.5f,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Logit {
    static Tensor invoke(
        const Tensor& input_tensor,
        std::optional<float> eps = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};
struct Deg2Rad {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Rad2Deg {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Tanh {
    static Tensor invoke(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        bool approx = false);
};

struct Clamp {
    static Tensor invoke(
        const Tensor& input_tensor,
        float min_val,
        float max_val,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        int32_t min_val,
        int32_t max_val,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Rsub {
    template <typename T>
    static Tensor invoke(
        const Tensor& input_tensor,
        T param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Rdiv {
    static Tensor invoke(
        const Tensor& input_tensor,
        float value,
        const std::optional<std::string>& rounding_mode = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Where {
    static Tensor invoke(
        const Tensor& condition,
        const ScalarVariant& value_true,
        const ScalarVariant& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Swish {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace operations::unary

// NOLINTBEGIN(bugprone-macro-parentheses)
#define REGISTER_UNARY_OPERATION(operation_name, operation_type) \
    constexpr auto operation_name = ttnn::register_operation<    \
        "ttnn::" #operation_name,                                \
        ttnn::operations::unary::ExecuteUnary<ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(operation_name, operation_type) \
    constexpr auto operation_name = ttnn::register_operation<                                   \
        "ttnn::" #operation_name,                                                               \
        ttnn::operations::unary::ExecuteUnaryWithFastAndApproximateMode<                        \
            ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_WITH_VECTOR_AND_FAST_AND_APPROXIMATE_MODE(operation_name, operation_type) \
    constexpr auto operation_name = ttnn::register_operation<                                              \
        "ttnn::" #operation_name,                                                                          \
        ttnn::operations::unary::ExecuteUnaryWithVectorAndFastAndApproximateMode<                          \
            ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(operation_name, operation_type) \
    constexpr auto operation_name = ttnn::register_operation<                         \
        "ttnn::" #operation_name,                                                     \
        ttnn::operations::unary::ExecuteUnaryWithFloatParameter<                      \
            ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_WITH_TWO_FLOAT_PARAMETER(operation_name, operation_type) \
    constexpr auto operation_name = ttnn::register_operation<                             \
        "ttnn::" #operation_name,                                                         \
        ttnn::operations::unary::ExecuteUnaryWithTwoFloatParameter<                       \
            ttnn::operations::unary::UnaryOpType::operation_type>>();

#define REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(operation_name, operation_type, data_type) \
    constexpr auto operation_name = ttnn::register_operation<                                      \
        "ttnn::" #operation_name,                                                                  \
        ttnn::operations::unary::                                                                  \
            ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::operation_type, data_type>>();

#define REGISTER_UNARY_OPERATION_WITH_OPTIONAL_INTEGER_PARAMETER(operation_name, operation_type, data_type) \
    constexpr auto operation_name = ttnn::register_operation<                                               \
        "ttnn::" #operation_name,                                                                           \
        ttnn::operations::unary::ExecuteUnaryWithOptionalIntegerParameter<                                  \
            ttnn::operations::unary::UnaryOpType::operation_type,                                           \
            data_type>>();
// NOLINTEND(bugprone-macro-parentheses)

REGISTER_UNARY_OPERATION(acos, ACOS);
REGISTER_UNARY_OPERATION(asin, ASIN);
REGISTER_UNARY_OPERATION(asinh, ASINH);
REGISTER_UNARY_OPERATION(atan, ATAN);
REGISTER_UNARY_OPERATION(atanh, ATANH);
REGISTER_UNARY_OPERATION(cos, COS);
REGISTER_UNARY_OPERATION(acosh, ACOSH);
REGISTER_UNARY_OPERATION(cosh, COSH);
REGISTER_UNARY_OPERATION(sinh, SINH);
REGISTER_UNARY_OPERATION(erfinv, ERFINV);
REGISTER_UNARY_OPERATION(exp2, EXP2);
REGISTER_UNARY_OPERATION(expm1, EXPM1);
REGISTER_UNARY_OPERATION(gez, GEZ);
REGISTER_UNARY_OPERATION(gtz, GTZ);
REGISTER_UNARY_OPERATION(i0, I0);
REGISTER_UNARY_OPERATION(i1, I1);
REGISTER_UNARY_OPERATION(isfinite, ISFINITE);
REGISTER_UNARY_OPERATION(isinf, ISINF);
REGISTER_UNARY_OPERATION(isnan, ISNAN);
REGISTER_UNARY_OPERATION(isneginf, ISNEGINF);
REGISTER_UNARY_OPERATION(isposinf, ISPOSINF);
REGISTER_UNARY_OPERATION(lez, LEZ);
REGISTER_UNARY_OPERATION(logical_not, LOGICAL_NOT_UNARY);
REGISTER_UNARY_OPERATION(ltz, LTZ);
REGISTER_UNARY_OPERATION(neg, NEG);
REGISTER_UNARY_OPERATION(nez, NEZ);
REGISTER_UNARY_OPERATION(reciprocal, RECIP);
REGISTER_UNARY_OPERATION(relu, RELU);
REGISTER_UNARY_OPERATION(relu6, RELU6);
REGISTER_UNARY_OPERATION(sign, SIGN);
REGISTER_UNARY_OPERATION(signbit, SIGNBIT);
REGISTER_UNARY_OPERATION(silu, SILU);
REGISTER_UNARY_OPERATION(sin, SIN);
REGISTER_UNARY_OPERATION(square, SQUARE);
REGISTER_UNARY_OPERATION(tan, TAN);
REGISTER_UNARY_OPERATION(tiled_prod, TILED_PROD);
REGISTER_UNARY_OPERATION(bitwise_not, BITWISE_NOT);
REGISTER_UNARY_OPERATION(alt_complex_rotate90, ALT_COMPLEX_ROTATE90);
REGISTER_UNARY_OPERATION(floor, FLOOR);
REGISTER_UNARY_OPERATION(ceil, CEIL);
REGISTER_UNARY_OPERATION(trunc, TRUNC);
REGISTER_UNARY_OPERATION(frac, FRAC);
REGISTER_UNARY_OPERATION(hardsigmoid, HARDSIGMOID);
REGISTER_UNARY_OPERATION(hardswish, HARDSWISH);
REGISTER_UNARY_OPERATION(softsign, SOFTSIGN);
REGISTER_UNARY_OPERATION(cbrt, CBRT);

// Unaries with fast_and_approximate_mode
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(exp, EXP);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erf, ERF);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erfc, ERFC);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(gelu, GELU);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log, LOG);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log10, LOG10);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log2, LOG2);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log1p, LOG1P);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt, RSQRT);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(sqrt, SQRT);

// Unaries with vector mode and fast and approximate mode
REGISTER_UNARY_OPERATION_WITH_VECTOR_AND_FAST_AND_APPROXIMATE_MODE(sigmoid, SIGMOID);

// Unaries with float parameter
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(heaviside, HEAVISIDE);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(leaky_relu, LEAKY_RELU);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_max, RELU_MAX);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_min, RELU_MIN);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(unary_remainder, REMAINDER);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(celu, CELU);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rpow, RPOW);

// Unaries with two float parameter
REGISTER_UNARY_OPERATION_WITH_TWO_FLOAT_PARAMETER(threshold, THRESHOLD);

// Unaries with integer parameter
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(power, POWER, uint32_t);

// Unaries with optional integer parameter
REGISTER_UNARY_OPERATION_WITH_OPTIONAL_INTEGER_PARAMETER(round, ROUND, int32_t);

// Other unaries
constexpr auto identity = ttnn::register_operation<"ttnn::identity", ttnn::operations::unary::Identity>();
constexpr auto abs = ttnn::register_operation<"ttnn::abs", ttnn::operations::unary::Abs>();
constexpr auto eqz = ttnn::register_operation<"ttnn::eqz", ttnn::operations::unary::Eqz>();
constexpr auto mish = ttnn::register_operation<"ttnn::mish", ttnn::operations::unary::Mish>();
constexpr auto hardmish = ttnn::register_operation<"ttnn::hardmish", ttnn::operations::unary::Hardmish>();
constexpr auto hardshrink = ttnn::register_operation<"ttnn::hardshrink", ttnn::operations::unary::Hardshrink>();
constexpr auto logit = ttnn::register_operation<"ttnn::logit", ttnn::operations::unary::Logit>();
constexpr auto elu = ttnn::register_operation<"ttnn::elu", ttnn::operations::unary::Elu>();
constexpr auto hardtanh = ttnn::register_operation<"ttnn::hardtanh", ttnn::operations::unary::Hardtanh>();
constexpr auto softshrink = ttnn::register_operation<"ttnn::softshrink", ttnn::operations::unary::Softshrink>();
constexpr auto deg2rad = ttnn::register_operation<"ttnn::deg2rad", ttnn::operations::unary::Deg2Rad>();
constexpr auto rad2deg = ttnn::register_operation<"ttnn::rad2deg", ttnn::operations::unary::Rad2Deg>();
constexpr auto clamp_tss = ttnn::register_operation<"ttnn::clamp_tss", ttnn::operations::unary::Clamp>();
constexpr auto softplus = ttnn::register_operation<"ttnn::softplus", ttnn::operations::unary::Softplus>();
constexpr auto tanh = ttnn::register_operation<"ttnn::tanh", ttnn::operations::unary::Tanh>();
constexpr auto tanhshrink = ttnn::register_operation<"ttnn::tanhshrink", ttnn::operations::unary::Tanhshrink>();
constexpr auto prelu_sfpu = ttnn::register_operation<"ttnn::prelu_sfpu", ttnn::operations::unary::Prelu>();
constexpr auto where_tss = ttnn::register_operation<"ttnn::where_tss", ttnn::operations::unary::Where>();
constexpr auto selu = ttnn::register_operation<"ttnn::selu", ttnn::operations::unary::Selu>();
constexpr auto bitcast = ttnn::register_operation<"ttnn::bitcast", ttnn::operations::unary::Bitcast>();
constexpr auto rdiv = ttnn::register_operation<"ttnn::rdiv", ttnn::operations::unary::Rdiv>();
constexpr auto swish = ttnn::register_operation<"ttnn::swish", ttnn::operations::unary::Swish>();
constexpr auto fill = ttnn::register_operation<
    "ttnn::fill",
    ttnn::operations::unary::ExecuteUnaryTSVariant<ttnn::operations::unary::UnaryOpType::FILL>>();
constexpr auto gt_unary = ttnn::register_operation<
    "ttnn::gt_unary",
    ttnn::operations::unary::ExecuteUnaryTSVariant<ttnn::operations::unary::UnaryOpType::UNARY_GT>>();
constexpr auto lt_unary = ttnn::register_operation<
    "ttnn::lt_unary",
    ttnn::operations::unary::ExecuteUnaryTSVariant<ttnn::operations::unary::UnaryOpType::UNARY_LT>>();
constexpr auto ne_unary = ttnn::register_operation<
    "ttnn::ne_unary",
    ttnn::operations::unary::ExecuteUnaryTSVariant<ttnn::operations::unary::UnaryOpType::UNARY_NE>>();
constexpr auto eq_unary = ttnn::register_operation<
    "ttnn::eq_unary",
    ttnn::operations::unary::ExecuteUnaryTSVariant<ttnn::operations::unary::UnaryOpType::UNARY_EQ>>();
constexpr auto ge_unary = ttnn::register_operation<
    "ttnn::ge_unary",
    ttnn::operations::unary::ExecuteUnaryTSVariant<ttnn::operations::unary::UnaryOpType::UNARY_GE>>();
constexpr auto le_unary = ttnn::register_operation<
    "ttnn::le_unary",
    ttnn::operations::unary::ExecuteUnaryTSVariant<ttnn::operations::unary::UnaryOpType::UNARY_LE>>();
constexpr auto sigmoid_accurate =
    ttnn::register_operation<"ttnn::sigmoid_accurate", ttnn::operations::unary::Sigmoid_accurate>();
constexpr auto log_sigmoid = ttnn::register_operation<"ttnn::log_sigmoid", ttnn::operations::unary::LogSigmoid>();
constexpr auto unary_chain = ttnn::register_operation<"ttnn::unary_chain", ttnn::operations::unary::Unary_chain>();
constexpr auto rsub_sfpu = ttnn::register_operation<"ttnn::rsub_sfpu", ttnn::operations::unary::Rsub>();
constexpr auto add_sfpu = ttnn::register_operation<
    "ttnn::add_sfpu",
    ttnn::operations::unary::SymmetricBinop<ttnn::operations::unary::UnaryOpType::ADD_UNARY_SFPU>>();
constexpr auto mul_sfpu = ttnn::register_operation<
    "ttnn::mul_sfpu",
    ttnn::operations::unary::SymmetricBinop<ttnn::operations::unary::UnaryOpType::MUL_UNARY_SFPU>>();

constexpr auto sub_sfpu = ttnn::register_operation<
    "ttnn::sub_sfpu",
    ttnn::operations::unary::AsymmetricBinop<
        ttnn::operations::unary::UnaryOpType::SUB_UNARY_SFPU,
        ttnn::operations::unary::UnaryOpType::RSUB>>();
constexpr auto div_sfpu = ttnn::register_operation<
    "ttnn::div_sfpu",
    ttnn::operations::unary::AsymmetricBinop<
        ttnn::operations::unary::UnaryOpType::DIV_UNARY_SFPU,
        ttnn::operations::unary::UnaryOpType::RDIV>>();
}  // namespace ttnn
