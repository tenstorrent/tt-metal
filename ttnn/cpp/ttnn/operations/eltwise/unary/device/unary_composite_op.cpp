// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_composite_op.hpp"

#include <array>
#include <optional>
#include <variant>

#include <utility>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/hal.hpp>
#include <cstdint>
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
namespace ttnn::detail {

// Function variance of whole tensor.
Tensor _variance_impl(
    const Tensor& y,
    const Tensor& /*mean_y*/,
    Tensor& y_minus_mean_y,
    const std::optional<MemoryConfig>& output_mem_config) {
    ttsl::SmallVector<int> dims = {2, 3};
    constexpr float correction = 0.0f;
    const auto& logical_shape = y.logical_shape();
    const auto rank = logical_shape.rank();
    TT_FATAL(rank == 4, "normalize_hw requires rank 4 tensor, got rank {}", rank);
    const uint64_t logical_elements = static_cast<uint64_t>(logical_shape[2]) * logical_shape[3];
    const float divisor = static_cast<float>(logical_elements) - correction;
    const float scale = (divisor > 0.0f) ? (1.0f / divisor) : 0.0f;
    Tensor sqr_y_minus_mean_y = ttnn::square(y_minus_mean_y, output_mem_config);
    return ttnn::sum(sqr_y_minus_mean_y, dims, true, output_mem_config, std::nullopt, scale);
}

// ttnn::var and ttnn::std over H and W run as one program, but give each (N, C) plane to a single core. The
// composite below spreads a plane over the grid in five programs (nine when H or W is not a multiple of 32, since
// each ttnn::mean then also copies the input and fills its tile padding), but it reads the input twice and writes
// and reads an input-sized intermediate. The one program is faster for planes of at most 64 tiles, and for larger
// planes from about 10 planes up, on both an 8x8 Wormhole and a 120-core Blackhole: the crossover does not move with
// the grid size. Below that the composite is faster, 3.2x for a single 1024x1024 plane on Wormhole.
bool _hw_statistic_fits_one_program(const Tensor& y) {
    const auto& shape = y.logical_shape();
    // Host tensors and ranks below 2 take the composite, whose first op reports them. Quasar has no two-pass
    // Std/Var kernel.
    if (!ttnn::is_device_tensor(y) || shape.rank() < 2 || y.device()->arch() == tt::ARCH::QUASAR) {
        return false;
    }
    const uint64_t plane_area = static_cast<uint64_t>(shape[-2]) * shape[-1];
    const uint64_t planes = plane_area == 0 ? 0 : shape.volume() / plane_area;
    const uint64_t tiles_per_plane = static_cast<uint64_t>(tt::div_up(shape[-2], tt::constants::TILE_HEIGHT)) *
                                     tt::div_up(shape[-1], tt::constants::TILE_WIDTH);
    constexpr uint64_t max_tiles_per_plane = 64;
    constexpr uint64_t min_planes = 10;
    return tiles_per_plane <= max_tiles_per_plane || planes >= min_planes;
}

// mean(y) over H and W, packed as FP32 whatever y's dtype. A bfloat16 mean is off by up to half a bfloat16 ULP, and
// subtracting it adds that error squared to every squared deviation: 4% of the variance for a mean of 100.3 and a
// std of 1. These are ttnn::mean's steps for the last two dimensions, which has no output dtype, with an FP32 output.
Tensor _mean_hw_fp32(const Tensor& y, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    const auto& shape = y.logical_shape();
    const auto rank = shape.rank();
    Tensor input = y;
    if (y.layout() == Layout::TILE &&
        (shape[-2] % tt::constants::TILE_HEIGHT != 0 || shape[-1] % tt::constants::TILE_WIDTH != 0)) {
        // fill_pad writes in place, so it fills a copy.
        input = ttnn::fill_implicit_tile_padding(ttnn::clone(input, std::nullopt, std::nullopt, std::nullopt), 0.0f);
    }
    input = rank > 4   ? operations::data_movement::squeeze_from_ND_to_4D(input)
            : rank < 4 ? operations::core::unsqueeze_to_4D(input)
                       : input;
    Tensor mean = operations::reduction::generic::detail::reduce(
        input,
        tt::tt_metal::ReduceOpMath::AVG,
        tt::tt_metal::ReduceOpDim::HW,
        1.0f / (static_cast<float>(shape[-2]) * static_cast<float>(shape[-1])),
        memory_config.value_or(input.memory_config()),
        DataType::FLOAT32);
    ttsl::SmallVector<uint32_t> mean_shape(shape.cbegin(), shape.cend());
    mean_shape[rank - 2] = 1;
    mean_shape[rank - 1] = 1;
    return ttnn::reshape(mean, ttnn::Shape(mean_shape));
}

// mean(x) over every element of x = [1, P, H, W], packed as FP32: ttnn::mean would round it to a bfloat16 or bfloat8_b
// input's dtype. The P plane means from _mean_hw_fp32 are averaged by an FP32 H reduce, which runs on the SFPU, after
// a transpose that writes zeros into the tile padding, as ttnn::mean does between its stages; ttnn::mean itself would
// first clone [1, P, 1, 1] to fill that padding. The FPU reduce of a lone bfloat16 tile still rounds its result to
// bfloat16 precision with an FP32 output, so a one-tile input is reduced from an FP32 copy.
Tensor _mean_fp32(const Tensor& x, const std::optional<MemoryConfig>& memory_config) {
    const uint32_t planes = x.logical_shape()[1];
    const bool one_tile = planes == 1 && x.padded_shape()[2] == tt::constants::TILE_HEIGHT &&
                          x.padded_shape()[3] == tt::constants::TILE_WIDTH;
    Tensor plane_means = _mean_hw_fp32(one_tile ? ttnn::typecast(x, DataType::FLOAT32) : x, memory_config);
    if (planes == 1) {
        return plane_means;
    }
    const Tensor column = ttnn::transpose(plane_means, 1, -2, memory_config, 0.0f);
    return operations::reduction::generic::detail::reduce(
        column,
        tt::tt_metal::ReduceOpMath::SUM,
        tt::tt_metal::ReduceOpDim::H,
        1.0f / static_cast<float>(planes),
        memory_config.value_or(column.memory_config()),
        DataType::FLOAT32);
}

// mean((y - mean(y))^2) over H and W. Both means divide by the logical H * W and leave out the tile padding.
// ttnn::bcast would round FP32 deviations to bfloat16 precision; ttnn::subtract keeps them. FP32 inputs subtract on
// the SFPU anyway; for bfloat16 the SFPU is requested, because the FPU would truncate the FP32 mean to TF32. The
// input-sized squared deviation stays where y is; only the result goes to output_mem_config.
Tensor _mean_squared_deviation_hw(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    ttsl::SmallVector<int> dims = {-2, -1};
    Tensor mean_y = y.dtype() == DataType::FLOAT32 ? ttnn::mean(y, dims, true) : _mean_hw_fp32(y);
    const std::array square = {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::SQUARE}};
    const std::optional<bool> fast_and_approximate_mode =
        y.dtype() == DataType::BFLOAT16 ? std::optional<bool>(false) : std::nullopt;
    Tensor squared_deviation =
        ttnn::subtract(y, mean_y, std::nullopt, std::nullopt, std::nullopt, square, {}, {}, fast_and_approximate_mode);
    return ttnn::mean(squared_deviation, dims, true, output_mem_config);
}

std::vector<Tensor> split_tensor_for_glu(
    const Tensor& input_a, std::int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> t_split;
    // The slices below always split dim 3; callers have already mapped -1 to 3.
    TT_FATAL(dim == 3, "glu-family ops only support the last dimension, got dim={}", dim);
    ttnn::Shape inshape(input_a.padded_shape());
    TT_FATAL(((inshape[3] / 2) % tt::constants::TILE_WIDTH == 0), "Split tensor dimension should be in full tile");
    ttsl::SmallVector<std::uint32_t> s_a = {0, 0, 0, 0};
    ttsl::SmallVector<std::uint32_t> e_a = {input_a.padded_shape()[0], inshape[1], inshape[2], inshape[3] / 2};

    ttsl::SmallVector<std::uint32_t> s_b = {0, 0, 0, inshape[3] / 2};
    ttsl::SmallVector<std::uint32_t> e_b = {inshape[0], inshape[1], inshape[2], inshape[3]};

    auto step = ttsl::SmallVector<std::uint32_t>({1, 1, 1, 1});
    Tensor t_a = ttnn::slice(input_a, s_a, e_a, step, output_mem_config);
    Tensor t_b = ttnn::slice(input_a, s_b, e_b, step, output_mem_config);

    t_split.emplace_back(t_a);
    t_split.emplace_back(t_b);

    return t_split;
}

}  // namespace ttnn::detail

namespace ttnn {

// Multivariate log-gamma function for p=4:
// ln(Gamma_4(a)) = 3*ln(pi) + ln(Gamma(a)) + ln(Gamma(a - 0.5)) + ln(Gamma(a - 1.0)) + ln(Gamma(a - 1.5))
// Valid domain: a > 1.5
// Ref : https://pytorch.org/docs/stable/special.html#torch.special.multigammaln
Tensor multigammaln(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = ttnn::lgamma(x, output_mem_config);
    result = ttnn::add(
        result,
        ttnn::lgamma(ttnn::subtract(x, 0.5f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    result = ttnn::add(
        result,
        ttnn::lgamma(ttnn::subtract(x, 1.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    result = ttnn::add(
        result,
        ttnn::lgamma(ttnn::subtract(x, 1.5f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    result = ttnn::add(result, 3.434189657547f, std::nullopt, output_mem_config);
    return result;
}

Tensor var_hw(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    if (detail::_hw_statistic_fits_one_program(y)) {
        return ttnn::var(y, ttsl::SmallVector<int>{-2, -1}, true, output_mem_config, std::nullopt, 1.0f, false);
    }
    return detail::_mean_squared_deviation_hw(y, output_mem_config);
}

// Function std
// compute standard deviation of tensor y over H and W = sqrt(mean((y - mean(y))^2)), dividing by the logical H * W
//  Ref: torch.std
Tensor std_hw(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    if (detail::_hw_statistic_fits_one_program(y)) {
        return ttnn::std(y, ttsl::SmallVector<int>{-2, -1}, true, output_mem_config, std::nullopt, 1.0f, false);
    }
    return ttnn::sqrt(detail::_mean_squared_deviation_hw(y, output_mem_config), false, output_mem_config);
}

// Function normalize
// use transformation y = (y - mean(y))/std(y) by broadcast
Tensor normalize_hw(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    ttsl::SmallVector<int> dims = {2, 3};
    Tensor mean_y = ttnn::mean(y, dims, true);
    Tensor y_minus_mean_y = ttnn::subtract(y, mean_y);
    // The divisor was built as reciprocal(sqrt(variance)), two dispatches for
    // what rsqrt is, and multiply applies a unary chain to its operands anyway,
    // so the rsqrt rides on the multiply that was going to read it.
    Tensor var_y = detail::_variance_impl(y, mean_y, y_minus_mean_y, output_mem_config);
    const std::array rsqrt_of_var = {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSQRT}};
    return ttnn::multiply(y_minus_mean_y, var_y, std::nullopt, output_mem_config, std::nullopt, {}, {}, rsqrt_of_var);
}

// Function Clip
// use clip y = min( max( x, min_value), max_value) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
Tensor clip(
    const Tensor& input_a,
    std::optional<float> min,
    std::optional<float> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    // Convert float optionals to variant optionals
    std::optional<std::variant<float, std::int32_t>> min_variant =
        min ? std::make_optional<std::variant<float, std::int32_t>>(std::in_place_type<float>, *min) : std::nullopt;
    std::optional<std::variant<float, std::int32_t>> max_variant =
        max ? std::make_optional<std::variant<float, std::int32_t>>(std::in_place_type<float>, *max) : std::nullopt;

    return clamp(input_a, min_variant, max_variant, output_mem_config);
}

Tensor clip(
    const Tensor& input_a,
    std::optional<Tensor> min,
    std::optional<Tensor> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    return clamp(input_a, std::move(min), std::move(max), output_mem_config);
}

// clamp
Tensor clamp(
    const Tensor& input_a,
    std::optional<std::variant<float, std::int32_t>> min,
    std::optional<std::variant<float, std::int32_t>> max,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(
        (max.has_value() || min.has_value()),
        "Either 'min' value or 'max' value can be None. Please provide at least one value");
    Tensor a = input_a;

    // Check if we have any int32_t scalars (both will be int32_t or null)
    bool has_int32_scalar = (min.has_value() && std::holds_alternative<std::int32_t>(min.value())) ||
                            (max.has_value() && std::holds_alternative<std::int32_t>(max.value()));

    // Convert input tensor to float32 only if input is INT32 and scalars are float (not int32)
    if (input_a.dtype() == DataType::INT32 && !has_int32_scalar) {
        a = ttnn::typecast(a, DataType::FLOAT32, output_mem_config);
    }

    if (has_int32_scalar) {
        // All scalars are int32_t (or null)
        std::int32_t min_val = min.has_value() ? std::get<std::int32_t>(min.value()) : -16775716;
        std::int32_t max_val =
            max.has_value() ? std::get<std::int32_t>(max.value())
                            : 16775716;  // max_val and min_val will be updated once unary infra supports int32 scalar.
        return ttnn::clamp_tss(a, min_val, max_val, output_mem_config, output_tensor);
    }  // All scalars are float (or null)
    float min_val = min.has_value() ? std::get<float>(min.value()) : std::numeric_limits<float>::lowest();
    float max_val = max.has_value() ? std::get<float>(max.value()) : std::numeric_limits<float>::max();
    return ttnn::clamp_tss(a, min_val, max_val, output_mem_config, output_tensor);
}

Tensor clamp(
    const Tensor& input_a,
    std::optional<Tensor> min,
    std::optional<Tensor> max,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    auto output_memory_config = output_mem_config.value_or(input_a.memory_config());
    TT_FATAL((max.has_value() || min.has_value()), "Only one of 'min' or 'max' can be None. Please provide one value");
    if (!max.has_value()) {
        return ttnn::where(
            ttnn::ge(input_a, min.value(), std::nullopt, output_memory_config),
            input_a,
            min.value(),
            output_memory_config,
            output_tensor);
    }
    if (!min.has_value()) {
        return ttnn::where(
            ttnn::le(input_a, max.value(), std::nullopt, output_memory_config),
            input_a,
            max.value(),
            output_memory_config,
            output_tensor);
    }
    // y = max(min(x, max), lo) with lo = min(min, max).
    // torch.clamp defines the degenerate case min > max as "every element becomes max"; clamping the
    // lower bound to the upper bound first folds that case into the plain two-op form, so the whole
    // tensor-bounds path is three device kernels instead of seven (issue #49996).
    Tensor lo = ttnn::minimum(min.value(), max.value(), std::nullopt, output_memory_config);
    Tensor a_max = ttnn::minimum(input_a, max.value(), std::nullopt, output_memory_config);
    return ttnn::maximum(a_max, lo, std::nullopt, output_memory_config, output_tensor);
}

// Gated Linear Unit activation: matmul(split[0],sigmoid(split[1]))
Tensor glu(const Tensor& input_a, std::int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    if (dim == -1) {
        dim = 3;
    }
    std::vector<Tensor> ab = detail::split_tensor_for_glu(input_a, dim, output_mem_config);
    Tensor sigmoid_b = ttnn::sigmoid(
        ab[1], (int)operations::unary::VecMode::RC, operations::unary::SigmoidMode::ACCURATE, output_mem_config);
    Tensor glu_result = ttnn::multiply(ab[0], sigmoid_b, std::nullopt, output_mem_config);
    return glu_result;
}

// ReLU Gated Linear Unit activation: matmul(split[0],relu(split[1]))
Tensor reglu(const Tensor& input_a, std::int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    if (dim == -1) {
        dim = 3;
    }
    std::vector<Tensor> ab = detail::split_tensor_for_glu(input_a, dim, output_mem_config);
    Tensor relu_b = ttnn::relu(ab[1], output_mem_config);
    Tensor reglu_result = ttnn::multiply(ab[0], relu_b, std::nullopt, output_mem_config);
    return reglu_result;
}

// Gaussian Error Gated Linear Unit activation: matmul(split[0],gelu(split[1]))
Tensor geglu(
    const Tensor& input_a,
    std::int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config,
    operations::unary::GeluVariant variant) {
    if (dim == -1) {
        dim = 3;
    }

    std::vector<Tensor> ab = detail::split_tensor_for_glu(input_a, dim, output_mem_config);

    Tensor gelu_b = ttnn::gelu(ab[1], variant, output_mem_config);
    Tensor geglu_result = ttnn::multiply(ab[0], gelu_b, std::nullopt, output_mem_config);
    return geglu_result;
}

Tensor geglu(const Tensor& input_a, std::int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    return geglu(input_a, dim, output_mem_config, operations::unary::GeluVariant::ACCURATE);
}

// Swish Gated Linear Unit activation: matmul(split[0],swish(split[1]))
Tensor swiglu(const Tensor& input_a, std::int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    if (dim == -1) {
        dim = 3;
    }

    std::vector<Tensor> ab = detail::split_tensor_for_glu(input_a, dim, output_mem_config);

    Tensor swish_b = ttnn::swish(ab[1], output_mem_config);
    Tensor swiglu_result = ttnn::multiply(ab[0], swish_b, std::nullopt, output_mem_config);
    return swiglu_result;
}

// The 0/1 mask has to be built in the input's own dtype. A bfloat16 mask against an
// integer input promotes the multiply to bfloat16, and the result comes back rounded
// to 8 mantissa bits -- tril(int32 around 2^30) was off by up to 1023.
static Tensor trilu_mask(
    const Tensor& input_a, std::int32_t diag, bool upper, const std::optional<MemoryConfig>& output_mem_config) {
    const auto mem = output_mem_config.value_or(input_a.memory_config());
    const DataType dt = input_a.dtype();
    if (dt == DataType::INT32) {
        return upper ? ttnn::index_triu<std::int32_t>(
                           input_a.logical_shape(), input_a.padded_shape(), diag, dt, Layout::TILE, input_a.device(), mem)
                     : ttnn::index_tril<std::int32_t>(
                           input_a.logical_shape(), input_a.padded_shape(), diag, dt, Layout::TILE, input_a.device(), mem);
    }
    if (dt == DataType::UINT32) {
        return upper ? ttnn::index_triu<std::uint32_t>(
                           input_a.logical_shape(), input_a.padded_shape(), diag, dt, Layout::TILE, input_a.device(), mem)
                     : ttnn::index_tril<std::uint32_t>(
                           input_a.logical_shape(), input_a.padded_shape(), diag, dt, Layout::TILE, input_a.device(), mem);
    }
    return upper ? ttnn::index_triu<::bfloat16>(
                       input_a.logical_shape(),
                       input_a.padded_shape(),
                       diag,
                       DataType::BFLOAT16,
                       Layout::TILE,
                       input_a.device(),
                       mem)
                 : ttnn::index_tril<::bfloat16>(
                       input_a.logical_shape(),
                       input_a.padded_shape(),
                       diag,
                       DataType::BFLOAT16,
                       Layout::TILE,
                       input_a.device(),
                       mem);
}

// tril : select lower triangular region of input matrix
Tensor tril(const Tensor& input_a, std::int32_t diag, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor index_l = trilu_mask(input_a, diag, /*upper=*/false, output_mem_config);
    return ttnn::multiply(input_a, index_l, std::nullopt, output_mem_config);
}

// triu : select upper triangular region of input matrix
Tensor triu(const Tensor& input_a, std::int32_t diag, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor index_u = trilu_mask(input_a, diag, /*upper=*/true, output_mem_config);
    return ttnn::multiply(input_a, index_u, std::nullopt, output_mem_config);
}

// polygamma ψ^(n)(x): implemented via a fused SFPU kernel using a finite-sum + tail approximation.
// The kernel evaluates (-1)^(n+1) * n! * Σ_{k=0}^{5} 1/(x + k)^(n+1) and applies an Euler–Maclaurin
// tail correction term to approximate the infinite remainder Σ_{k=6}^{∞} 1/(x + k)^(n+1), rather than
// performing a hard truncation at k = 5. This is a single kernel dispatch instead of 11+ composite ops.
Tensor polygamma(const Tensor& input_a, std::int32_t k, const std::optional<MemoryConfig>& output_mem_config) {
    // Range includes k=11 to support polygamma_bw which computes polygamma(input, n+1)
    TT_FATAL(k >= 1 && k <= 11, "polygamma order must be in range [1, 11], got {}", k);
    float n = static_cast<float>(k);
    float fact_val = std::tgamma(1.0f + k);       // k!
    float pos_neg = (k % 2 == 0) ? -1.0f : 1.0f;  // (-1)^(k+1)
    float scale = fact_val * pos_neg;
    return ttnn::operations::unary::detail::unary_impl(
        input_a,
        {operations::unary::UnaryWithParam(operations::unary::UnaryOpType::POLYGAMMA, {n, scale})},
        output_mem_config);
}

// // tanhshrink(x) = x - tanh(x)
Tensor logical_not_(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::logical_not(x, output_mem_config, x);
}

// Global Norm
// (y - mean(y)) / std(y) over every element of a rank 4 tensor, with the population std. [N, C, H, W] is viewed as
// [1, N * C, H, W], which only merges the outer dims and so moves no data for any H and W. Reducing dims 1, 2 and 3
// then spreads H and W over the grid plane by plane before combining the N * C plane results. Both means divide by
// the logical element count, so tile padding is not counted. A single [1, 1, N * C * H, W] plane would reduce H on
// one core, 2.4x slower for 8x8x128x128, and would move data whenever H is not a multiple of 32.
Tensor normalize_global(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    const auto& shape = y.logical_shape();
    TT_FATAL(shape.rank() == 4, "normalize_global requires a rank 4 tensor, got rank {}", shape.rank());
    const uint32_t planes = shape[0] * shape[1];
    // An empty input skips the FP32 mean and the bfloat8_b steps below: the reduce they use divides by its tile count.
    const bool empty = y.logical_volume() == 0;
    // ttnn::mean over several dims of a ROW_MAJOR FLOAT32 tensor is off by about 4e-4, so ROW_MAJOR input is tilized
    // first; the output is TILE, as it was before. to_layout cannot tilize a ROW_MAJOR shard that is not whole tiles,
    // so such an input first goes to interleaved memory of the same buffer type.
    Tensor x = y;
    if (y.layout() != Layout::TILE) {
        if (y.is_sharded() && !tt::tt_metal::can_shard_align(y.memory_config(), Layout::TILE)) {
            x = ttnn::to_memory_config(
                y, MemoryConfig{TensorMemoryLayout::INTERLEAVED, y.memory_config().buffer_type()});
        }
        x = ttnn::to_layout(x, Layout::TILE);
    }
    // The output defaults to the tilized input's memory config: the reshapes re-derive an ND shard spec, so inheriting
    // one would not give the input's back.
    const MemoryConfig output_memory_config = output_mem_config.value_or(x.memory_config());
    // bfloat8_b input runs as bfloat16, which holds its values exactly: the FPU reduce of bfloat8_b truncates its FP32
    // partial means to TF32, 0.05 off for a mean of 100.
    if (x.dtype() == DataType::BFLOAT8_B && !empty) {
        x = ttnn::typecast(x, DataType::BFLOAT16);
    }
    x = ttnn::reshape(x, ttnn::Shape{1, planes, shape[2], shape[3]});
    const ttsl::SmallVector<int> dims = planes == 1 ? ttsl::SmallVector<int>{2, 3} : ttsl::SmallVector<int>{1, 2, 3};
    // The [1, N * C, 1, 1] statistics cannot keep a sharded input's shard spec.
    const std::optional<MemoryConfig> stats_mem_config =
        x.is_sharded() ? std::optional<MemoryConfig>(ttnn::DRAM_MEMORY_CONFIG) : std::nullopt;
    Tensor y_minus_mean_y;
    if (x.dtype() == DataType::FLOAT32 || empty) {
        y_minus_mean_y = ttnn::subtract(x, ttnn::mean(x, dims, true, stats_mem_config));
    } else {
        // A bfloat16 mean of 100.3 next to a std of 0.5 put outputs about 1.0 off, so the mean stays FP32, as in
        // var_hw, and the SFPU subtracts it whole: the FPU would truncate it to TF32.
        y_minus_mean_y = ttnn::subtract(
            x,
            detail::_mean_fp32(x, stats_mem_config),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            {},
            {},
            {},
            /*fast_and_approximate_mode=*/false);
    }
    // Frees a tilized or ND-staged copy of the input; a view of y keeps y's buffer, which y still owns.
    x.deallocate();
    const Tensor var_y = ttnn::mean(ttnn::square(y_minus_mean_y), dims, true, stats_mem_config);
    // Reshaped back before the multiply, so the multiply lays out the output memory config, ND shard spec included, on
    // the output shape itself. When that reshape copies, for an ND shard spec, the [1, N * C, H, W] deviation is freed.
    Tensor centered = ttnn::reshape(y_minus_mean_y, shape);
    y_minus_mean_y.deallocate();
    // A bfloat8_b tile shares an exponent across each 16 values of a row, so the large (0 - mean) / std of the tile
    // padding would round the real outputs beside it to a few levels. Zero padding stays 0 through the multiply.
    if (y.dtype() == DataType::BFLOAT8_B && !empty &&
        (shape[2] % tt::constants::TILE_HEIGHT != 0 || shape[3] % tt::constants::TILE_WIDTH != 0)) {
        centered = ttnn::fill_implicit_tile_padding(centered, 0.0f);
    }
    const std::array rsqrt_of_var = {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSQRT}};
    const std::optional<const DataType> output_dtype =
        centered.dtype() == y.dtype() ? std::nullopt : std::optional<const DataType>(y.dtype());
    return ttnn::multiply(centered, var_y, output_dtype, output_memory_config, std::nullopt, {}, {}, rsqrt_of_var);
}

}  // namespace ttnn
