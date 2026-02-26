// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_composite_op.hpp"

#include <functional>
#include <optional>
#include <variant>

#include <utility>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
namespace ttnn::operations::unary {
// Note: This namespace remains as ttnn::operations::unary because it contains composite operations
// that are not primitive device operations

// TODO: In future will uplift the op once the floor and tan has supported.
// digamma support for the range of (1, inf)
Tensor _digamma(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor input = input_a.dtype() == DataType::BFLOAT8_B ? ttnn::fill_implicit_tile_padding(input_a, 1.0f) : input_a;
    Tensor t_log_out = ttnn::log(input, true, output_mem_config);  // negative log is not useful here

    // 1/2(z)
    Tensor input_recip = ttnn::reciprocal(input, output_mem_config);
    Tensor output = ttnn::multiply(input_recip, 0.5f, std::nullopt, output_mem_config);
    Tensor tmp = ttnn::square(input_recip, output_mem_config);
    Tensor val_square = tmp;
    // (1/12) * x^2
    output = ttnn::subtract(output, ttnn::multiply(tmp, 0.083333333f), std::nullopt, output_mem_config);

    // (1/120) * x^4
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output = ttnn::add(
        output,
        ttnn::multiply(tmp, 0.008333333333333333f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);

    //(1/252) * x^6
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output = ttnn::subtract(
        output,
        ttnn::multiply(tmp, 0.003968253968253968f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);

    // (1/240) *x^8
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output = ttnn::add(
        output,
        ttnn::multiply(tmp, 0.004166666666666667f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);

    //(1/132) * x^10
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output = ttnn::subtract(output, ttnn::multiply(tmp, 0.007575757575757576), std::nullopt, output_mem_config);

    //(691/32760) * x^12
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output = ttnn::add(
        output,
        ttnn::multiply(tmp, 0.021092796092796094, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);

    //(1/12) * x^14
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output = ttnn::subtract(
        output,
        ttnn::multiply(tmp, 0.08333333333333333, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);

    return ttnn::subtract(t_log_out, output, std::nullopt, output_mem_config);
}

Tensor _lgamma_fast(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    std::cout << "_lgamma_fast" << std::endl;
    // // 1. Reflection for x < 0.5
    // Tensor is_small = ttnn::lt(x, 0.5f);
    // Tensor z = ttnn::where(is_small, ttnn::rsub_sfpu(x, 1.0f), x);

    // // 2. Stirling base: (z - 0.5) * log(z) - z + log(sqrt(2*pi))
    // Tensor log_z = ttnn::log(z, true);
    // Tensor res = ttnn::multiply(ttnn::subtract(z, 0.5f), log_z);
    // res = ttnn::subtract(res, z);
    // res = ttnn::add(res, 0.9189385332046727f);

    // // 3. High-Accuracy Correction (The "Bernoulli" series)
    // // We use a minimax rational fit for 1/z.
    // // This replaces the 9 Lanczos terms with just 3 operations.
    // Tensor inv_z2 = ttnn::reciprocal(ttnn::multiply(z, z)); // 1/z^2

    // // Minimal coefficients for Float32 0-3 ULP
    // const float r0 = 0.0833333333f;    // 1/12
    // const float r1 = -0.0027777777f;   // -1/360

    // // correction = (1/z) * (r0 + r1/z^2)
    // Tensor correction = ttnn::multiply(ttnn::reciprocal(z), ttnn::add(ttnn::multiply(inv_z2, r1), r0));
    // res = ttnn::add(res, correction);

    Tensor res_positive = ttnn::lgamma_partial(x, output_mem_config);

    // 2. Calculate reflection_adj
    // (Keep this in float32 if possible)
    Tensor sin_pi_x = ttnn::sin(ttnn::multiply(x, (float)M_PI));
    Tensor is_integer = ttnn::eq(x, ttnn::floor(x));
    sin_pi_x = ttnn::where(is_integer, 0.0f, sin_pi_x);

    Tensor log_abs_sin = ttnn::log(ttnn::abs(sin_pi_x));
    Tensor reflection_adj = ttnn::rsub_sfpu(log_abs_sin, 1.1447298858f);  // ln(pi) - log|sin|

    // 3. FINAL SUBTRACTION ORDER
    // result = (ln(pi) - log|sin|) - lgamma(1-x)

    // 4. Proceed with reflection formula
    // reflection_adj - lgamma_pos(1-x)
    // If log_sin is -inf, result becomes +inf ?

    return ttnn::where(
        ttnn::lt(x, 0.5f), ttnn::subtract(reflection_adj, res_positive), res_positive, output_mem_config);
}

Tensor _lgamma_fp32(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    std::cout << "_lgamma_refined_g7" << std::endl;
    // 1. Handle Reflection for x < 0.5
    // This is vital for negative numbers and inputs close to zero.
    Tensor is_small = ttnn::lt(x, 0.5f);
    Tensor x_reflected = ttnn::rsub_sfpu(x, 1.0f);
    Tensor z = ttnn::where(is_small, x_reflected, x);

    // 2. Lanczos Approximation (g=7, n=9)
    // High-precision Godfrey coefficients
    const float log_sqrt_2pi = 0.9189385332046727f;
    const float c0 = 0.99999999999980993f;
    const float c1 = 676.5203681218851f;
    const float c2 = -1259.1392167224028f;
    const float c3 = 771.32342877765313f;
    const float c4 = -176.61502916214059f;
    const float c5 = 12.507343278686905f;
    const float c6 = -0.13857109526572012f;
    const float c7 = 9.9843695780195716e-6f;
    const float c8 = 1.5056327351493116e-7f;

    // Unrolled Series: series = c0 + c1/(z+1) + c2/(z+2) ...
    // Note: We use z directly (no z-1 shift) for better stability near 1.0
    Tensor series = ttnn::full_like(z, c0);
    series = ttnn::add(series, ttnn::multiply(ttnn::reciprocal(ttnn::add(z, 1.0f)), c1));
    series = ttnn::add(series, ttnn::multiply(ttnn::reciprocal(ttnn::add(z, 2.0f)), c2));
    series = ttnn::add(series, ttnn::multiply(ttnn::reciprocal(ttnn::add(z, 3.0f)), c3));
    series = ttnn::add(series, ttnn::multiply(ttnn::reciprocal(ttnn::add(z, 4.0f)), c4));
    series = ttnn::add(series, ttnn::multiply(ttnn::reciprocal(ttnn::add(z, 5.0f)), c5));
    series = ttnn::add(series, ttnn::multiply(ttnn::reciprocal(ttnn::add(z, 6.0f)), c6));
    series = ttnn::add(series, ttnn::multiply(ttnn::reciprocal(ttnn::add(z, 7.0f)), c7));
    series = ttnn::add(series, ttnn::multiply(ttnn::reciprocal(ttnn::add(z, 8.0f)), c8));

    // Lanczos calculation:
    // res = (z + 0.5) * log(z + 7.5) - (z + 7.5) + log(sqrt(2*pi) * series / z)
    Tensor tmp = ttnn::add(z, 7.5f, std::nullopt, output_mem_config);
    Tensor log_tmp = ttnn::log(tmp, true, output_mem_config);

    Tensor term1 = ttnn::multiply(ttnn::add(z, 0.5f), log_tmp);
    Tensor term2 = ttnn::subtract(ttnn::add(ttnn::log(series), log_sqrt_2pi), ttnn::log(z));
    Tensor res_positive = ttnn::subtract(ttnn::add(term1, term2), tmp);

    // 3. Reflection Adjustment for x < 0.5
    // reflection = log(pi) - log(abs(sin(pi * x))) - res_positive
    Tensor pi_x = ttnn::multiply(x, (float)M_PI);
    Tensor log_sin_pi_x = ttnn::log(ttnn::abs(ttnn::sin(pi_x)));
    Tensor reflection_adj = ttnn::rsub_sfpu(log_sin_pi_x, (float)std::log(M_PI));
    Tensor res_reflected = ttnn::subtract(reflection_adj, res_positive);

    // 4. Final Result Selection
    return ttnn::where(ttnn::lt(x, 0.5f), res_reflected, res_positive, output_mem_config);
}

Tensor Lgamma::invoke(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    if (x.dtype() == DataType::FLOAT32) {
        return _lgamma_fp32(x, output_mem_config);
    } else {
        return _lgamma_fast(x, output_mem_config);
    }
}

Tensor _lgamma(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    return Lgamma::invoke(x, output_mem_config);
}
// multivariate log-gamma function
// Ref : https://pytorch.org/docs/stable/special.html#torch.special.multigammaln
Tensor _multigammaln(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = _lgamma(x, output_mem_config);
    result = ttnn::add(
        result,
        _lgamma(ttnn::subtract(x, 0.5f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    result = ttnn::add(
        result,
        _lgamma(ttnn::subtract(x, 1.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    result = ttnn::add(
        result,
        _lgamma(ttnn::subtract(x, 1.5f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    result = ttnn::add(result, 3.434189657547f, std::nullopt, output_mem_config);
    return result;
}

// Function variance of whole tensor.
// Tensor variance(const Tensor& y,const Tensor& mean_y);
Tensor _variance_impl(
    const Tensor& y,
    const Tensor& /*mean_y*/,
    Tensor& y_minus_mean_y,
    const std::optional<MemoryConfig>& output_mem_config) {
    ttnn::SmallVector<int> dims = {2, 3};
    constexpr float correction = 0.0f;
    auto shape_wh = y.padded_shape();
    float scale = 1.0f / ((float)(shape_wh[3] * shape_wh[2]) - correction);
    Tensor sqr_y_minus_mean_y = ttnn::square(y_minus_mean_y, output_mem_config);
    return ttnn::sum(sqr_y_minus_mean_y, dims, true, std::nullopt, std::nullopt, scale);
}
Tensor _variance_impl(const Tensor& y, const Tensor& mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor y_minus_mean_y = ttnn::bcast(y, mean_y, ttnn::BcastOpMath::SUB, ttnn::BcastOpDim::HW);
    return _variance_impl(y, mean_y, y_minus_mean_y, output_mem_config);
}

Tensor _variance(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(y.memory_config());
    ttnn::SmallVector<int> dims = {2, 3};
    Tensor mean_y = ttnn::mean(y, dims, true);
    return _variance_impl(y, mean_y, output_memory_config);
}

// Function std
// compute standard deviation of tensor y = sqrt( E((y-<y>)^2)/ y.volume() )
//  Ref: torch.std
Tensor _std(const Tensor& y, const Tensor& mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, output_mem_config));
}

Tensor _std(
    const Tensor& y,
    const Tensor& mean_y,
    Tensor& y_minus_mean_y,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, y_minus_mean_y, output_mem_config));
}

Tensor _std_overload(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::sqrt(_variance(y, output_mem_config));
}

// Function normalize
// use transformation y = (y - mean(y))/std(y) by broadcast
Tensor _normalize(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    ttnn::SmallVector<int> dims = {2, 3};
    Tensor mean_y = ttnn::mean(y, dims, true);
    Tensor y_minus_mean_y = ttnn::bcast(y, mean_y, ttnn::BcastOpMath::SUB, ttnn::BcastOpDim::HW);
    Tensor std_y = _std(y, mean_y, y_minus_mean_y, output_mem_config);
    Tensor recip_std_y = ttnn::reciprocal(std_y, output_mem_config);
    Tensor z = ttnn::multiply(y_minus_mean_y, recip_std_y, std::nullopt, output_mem_config);
    return z;
}

// Function Clip
// use clip y = min( max( x, min_value), max_value) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
Tensor ExecuteUnaryCompositeClip::invoke(
    const Tensor& a,
    std::optional<float> min,
    std::optional<float> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    // Convert float optionals to variant optionals
    std::optional<std::variant<float, int32_t>> min_variant =
        min ? std::make_optional<std::variant<float, int32_t>>(std::in_place_type<float>, *min) : std::nullopt;
    std::optional<std::variant<float, int32_t>> max_variant =
        max ? std::make_optional<std::variant<float, int32_t>>(std::in_place_type<float>, *max) : std::nullopt;

    return ExecuteUnaryCompositeClamp::invoke(a, min_variant, max_variant, output_mem_config);
}

Tensor ExecuteUnaryCompositeClip::invoke(
    const Tensor& a,
    std::optional<Tensor> min,
    std::optional<Tensor> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ExecuteUnaryCompositeClamp::invoke(a, std::move(min), std::move(max), output_mem_config);
}

// clamp
Tensor ExecuteUnaryCompositeClamp::invoke(
    const Tensor& input_a,
    std::optional<std::variant<float, int32_t>> min,
    std::optional<std::variant<float, int32_t>> max,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(
        (max.has_value() || min.has_value()),
        "Either 'min' value or 'max' value can be None. Please provide at least one value");
    Tensor a = input_a;

    // Check if we have any int32_t scalars (both will be int32_t or null)
    bool has_int32_scalar = (min.has_value() && std::holds_alternative<int32_t>(min.value())) ||
                            (max.has_value() && std::holds_alternative<int32_t>(max.value()));

    // Convert input tensor to float32 only if input is INT32 and scalars are float (not int32)
    if (input_a.dtype() == DataType::INT32 && !has_int32_scalar) {
        a = ttnn::typecast(a, DataType::FLOAT32, output_mem_config);
    }

    if (has_int32_scalar) {
        // All scalars are int32_t (or null)
        int32_t min_val = min.has_value() ? std::get<int32_t>(min.value()) : -16775716;
        int32_t max_val =
            max.has_value() ? std::get<int32_t>(max.value())
                            : 16775716;  // max_val and min_val will be updated once unary infra supports int32 scalar.
        return ttnn::clamp_tss(a, min_val, max_val, output_mem_config, output_tensor);
    }  // All scalars are float (or null)
    float min_val = min.has_value() ? std::get<float>(min.value()) : std::numeric_limits<float>::lowest();
    float max_val = max.has_value() ? std::get<float>(max.value()) : std::numeric_limits<float>::max();
    return ttnn::clamp_tss(a, min_val, max_val, output_mem_config, output_tensor);
}

Tensor ExecuteUnaryCompositeClamp::invoke(
    const Tensor& a,
    std::optional<Tensor> min,
    std::optional<Tensor> max,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& /*output_tensor*/) {
    auto output_memory_config = output_mem_config.value_or(a.memory_config());
    TT_FATAL((max.has_value() || min.has_value()), "Only one of 'min' or 'max' can be None. Please provide one value");
    if (!max.has_value()) {
        return ttnn::where(
            ttnn::ge(a, min.value(), std::nullopt, output_memory_config), a, min.value(), output_memory_config);
    }
    if (!min.has_value()) {
        return ttnn::where(
            ttnn::le(a, max.value(), std::nullopt, output_memory_config), a, max.value(), output_memory_config);
    }
    Tensor a_max = ttnn::minimum(a, max.value(), std::nullopt, output_memory_config);
    Tensor temp = ttnn::where(
        ttnn::eq(min.value(), 0.0f, std::nullopt, output_memory_config),
        ttnn::relu(a_max, output_memory_config),
        ttnn::maximum(a_max, min.value(), std::nullopt, output_memory_config),
        output_memory_config);
    return ttnn::where(
        ttnn::gt(min.value(), max.value(), std::nullopt, output_memory_config),
        max.value(),
        temp,
        output_memory_config);
}

// Theano defines this differently...
/**
 *
 *   alpha = 1.6732632423543772848170429916717
 *    scale = 1.0507009873554804934193349852946
 *    return scale * elu(x, alpha)
 *
 */
// Function Selu - scaled exponential linear
// use transformation y = scale *(max(0,x) + min(0,alpha * (exp(X)-1))) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
Tensor _selu(
    const Tensor& x, const float scale, const float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    // term 2
    Tensor x_Exp_minus_1 = ttnn::expm1(x, output_mem_config);
    Tensor result_t2_ = ttnn::multiply_(x_Exp_minus_1, alpha);
    x_Exp_minus_1.deallocate();
    Tensor result_term2 = ttnn::minimum(result_t2_, 0.0f, std::nullopt, output_mem_config);
    result_t2_.deallocate();

    // term 1
    Tensor x_max = ttnn::maximum(x, 0.0f, std::nullopt, output_mem_config);
    Tensor sum_max_term2 = ttnn::add_(x_max, result_term2);
    x_max.deallocate();
    Tensor result_selu = ttnn::multiply_(sum_max_term2, scale);

    return result_selu;
}

std::vector<Tensor> split_tensor_for_glu(
    const Tensor& input_a, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> t_split;
    ttnn::Shape inshape(input_a.padded_shape());
    TT_FATAL(((inshape[dim] / 2) % tt::constants::TILE_WIDTH == 0), "Split tensor dimension should be in full tile");
    ttnn::SmallVector<uint32_t> s_a = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> e_a = {input_a.padded_shape()[0], inshape[1], inshape[2], inshape[3] / 2};

    ttnn::SmallVector<uint32_t> s_b = {0, 0, 0, inshape[3] / 2};
    ttnn::SmallVector<uint32_t> e_b = {inshape[0], inshape[1], inshape[2], inshape[3]};

    auto step = ttnn::SmallVector<uint32_t>({1, 1, 1, 1});
    Tensor t_a = ttnn::slice(input_a, s_a, e_a, step, output_mem_config);
    Tensor t_b = ttnn::slice(input_a, s_b, e_b, step, output_mem_config);

    t_split.emplace_back(t_a);
    t_split.emplace_back(t_b);

    return t_split;
}

// Gated Linear Unit activation: matmul(split[0],sigmoid(split[1]))
Tensor _glu(const Tensor& input_a, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim GLU only supported at this time ");
    if (dim == -1) {
        dim = 3;
    }
    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);
    Tensor sigmoid_b = ttnn::sigmoid(ab[1], (int)VecMode::RC, Sigmoid::SigmoidMode::ACCURATE, output_mem_config);
    Tensor glu_result = ttnn::multiply(ab[0], sigmoid_b, std::nullopt, output_mem_config);
    return glu_result;
}

// ReLU Gated Linear Unit activation: matmul(split[0],relu(split[1]))
Tensor _reglu(const Tensor& input_a, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim REGLU only supported at this time ");
    if (dim == -1) {
        dim = 3;
    }
    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);
    Tensor relu_b = ttnn::relu(ab[1], output_mem_config);
    Tensor reglu_result = ttnn::multiply(ab[0], relu_b, std::nullopt, output_mem_config);
    return reglu_result;
}

// Gaussian Error Gated Linear Unit activation: matmul(split[0],gelu(split[1]))
Tensor _geglu(const Tensor& input_a, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim GEGLU only supported at this time ");
    if (dim == -1) {
        dim = 3;
    }

    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);

    constexpr bool fast_appx = true;
    Tensor gelu_b = ttnn::gelu(ab[1], fast_appx, output_mem_config);
    Tensor geglu_result = ttnn::multiply(ab[0], gelu_b, std::nullopt, output_mem_config);
    return geglu_result;
}

// Swish Gated Linear Unit activation: matmul(split[0],swish(split[1]))
Tensor _swiglu(const Tensor& input_a, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim SWIGLU only supported at this time ");
    if (dim == -1) {
        dim = 3;
    }

    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);

    Tensor swish_b = ttnn::swish(ab[1], output_mem_config);
    Tensor swiglu_result = ttnn::multiply(ab[0], swish_b, std::nullopt, output_mem_config);
    return swiglu_result;
}

// tril : select lower triangular region of input matrix
Tensor _tril(const Tensor& input_a, int32_t diag, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor index_l = ttnn::index_tril<::bfloat16>(
        input_a.logical_shape(),
        input_a.padded_shape(),
        diag,
        DataType::BFLOAT16,
        Layout::TILE,
        input_a.device(),
        output_mem_config.value());
    return ttnn::multiply(input_a, index_l, std::nullopt, output_mem_config);
}

// triu : select upper triangular region of input matrix
Tensor _triu(const Tensor& input_a, int32_t diag, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor index_u = ttnn::index_triu<::bfloat16>(
        input_a.logical_shape(),
        input_a.padded_shape(),
        diag,
        DataType::BFLOAT16,
        Layout::TILE,
        input_a.device(),
        output_mem_config.value());
    return ttnn::multiply(input_a, index_u, std::nullopt, output_mem_config);
}

Tensor is_odd(const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = ttnn::multiply(input, (1.0f / 2.0f), std::nullopt, output_mem_config);
    Tensor floor_res = ttnn::floor(result, output_mem_config);
    return ttnn::ne(result, floor_res, std::nullopt, output_mem_config);
}

// polygamma support for the range of input(1, 10) and n(1, 10)
Tensor _polygamma(const Tensor& input_a, int32_t k, const std::optional<MemoryConfig>& output_mem_config) {
    float k_der = 1.0f + k;
    float fact_val = std::tgamma(k_der);
    float pos_neg = 1.0f;
    if (k == 2 || k == 4 || k == 6 || k == 8 || k == 10) {
        pos_neg = -1.0f;
    }
    Tensor temp(input_a);
    {
        Tensor z1 = ttnn::reciprocal(ttnn::power(input_a, k_der, output_mem_config), output_mem_config);
        temp = z1;
        for (int idx = 1; idx < 11; idx++) {
            z1 = ttnn::reciprocal(
                ttnn::power(ttnn::add(input_a, idx, std::nullopt, output_mem_config), k_der, output_mem_config),
                output_mem_config);
            temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);
        }
    }
    fact_val *= pos_neg;
    return ttnn::multiply(temp, fact_val, std::nullopt, output_mem_config);
}

// // tanhshrink(x) = x - tanh(x)
Tensor _logical_not_(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::logical_not(x, output_mem_config, x);
}

using HWFunctionT = std::function<Tensor(const Tensor& y, const std::optional<MemoryConfig>&)>;
Tensor _make_global_from_hw_impl(
    const HWFunctionT& fn, const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(y.padded_shape().rank() == 4, "Cannot support non-rank 4 Tensor");

    // format to HW
    Tensor y_hw = ttnn::reshape_on_device(
        y, ttnn::Shape{1, 1, y.padded_shape()[2], y.padded_shape()[3] * y.padded_shape()[1] * y.padded_shape()[0]});

    // compute @fn
    Tensor z_0 = fn(y_hw, output_mem_config);
    TT_FATAL(y_hw.padded_shape() == z_0.padded_shape(), "shape match");
    y_hw.deallocate();

    // reformat
    Tensor z_1 = ttnn::reshape_on_device(
        z_0, ttnn::Shape{y.padded_shape()[0], y.padded_shape()[1], y.padded_shape()[2], y.padded_shape()[3]});
    z_0.deallocate();

    return z_1;
}

// Global Norm
Tensor _normalize_global(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    return _make_global_from_hw_impl(_normalize, y, output_mem_config);
}

}  // namespace ttnn::operations::unary
