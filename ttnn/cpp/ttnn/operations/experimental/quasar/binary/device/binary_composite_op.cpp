// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>
#include <utility>
#include "ttnn/operations/experimental/quasar/binary/binary.hpp"
#include "ttnn/operations/experimental/quasar/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/experimental/quasar/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"
#include "ttnn/device.hpp"
#include <variant>
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::experimental::quasar::binary {

using namespace operations;

// nextafter
Tensor nextafter(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const float eps = tt::tt_metal::hal::get_eps();
    Tensor result(input_a);
    {
        Tensor eps_gt(input_a);
        {
            eps_gt = ttnn::where(
                gt(input_a, input_b, std::nullopt, output_mem_config),
                add(input_a, eps, std::nullopt, output_mem_config),
                input_a);
        }
        result = ttnn::where(
            lt(input_a, input_b, std::nullopt, output_mem_config),
            subtract(input_a, eps, std::nullopt, output_mem_config),
            eps_gt);
    }
    return result;
}

Tensor minimum(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& /*output_dtype*/,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_tensor_a,
        input_tensor_b,
        binary::BinaryOpType::MINIMUM,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations);
}

Tensor minimum(
    const Tensor& input_a,
    unary::ScalarVariant value,
    const std::optional<const DataType>& /*output_dtype*/,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*post_activations*/,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*lhs_activations*/,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*rhs_activations*/) {
    return std::visit(
        [&](auto input_b) {
            return ttnn::operations::unary::detail::unary_impl(
                input_a,
                {unary::EltwiseUnaryWithParam{unary::UnaryOpType::MINIMUM, (input_b)}},
                memory_config,
                optional_output_tensor);
        },
        value);
}

Tensor maximum(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& /*output_dtype*/,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_tensor_a,
        input_tensor_b,
        binary::BinaryOpType::MAXIMUM,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations);
}

Tensor maximum(
    const Tensor& input_a,
    unary::ScalarVariant value,
    const std::optional<const DataType>& /*output_dtype*/,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*post_activations*/,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*lhs_activations*/,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*rhs_activations*/) {
    return std::visit(
        [&](auto input_b) {
            return ttnn::operations::unary::detail::unary_impl(
                input_a,
                {unary::EltwiseUnaryWithParam{unary::UnaryOpType::MAXIMUM, (input_b)}},
                memory_config,
                optional_output_tensor);
        },
        value);
}

Tensor atan2(const Tensor& input_b, const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_b,
        input_a,
        binary::BinaryOpType::ATAN2,
        std::nullopt,
        output_mem_config,
        std::nullopt,
        {},
        {},
        {},
        std::nullopt);
}

Tensor div(
    const Tensor& input,
    unary::ScalarVariant value,
    bool fast_and_approximate_mode,
    const std::optional<std::string>& rounding_mode,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    const bool is_int32 = input.dtype() == DataType::INT32;

    if (is_int32) {
        TT_FATAL(
            !fast_and_approximate_mode,
            "Integer Division does not support fast_and_approximate_mode=true {}",
            fast_and_approximate_mode);
        // fast_and_approximate_mode is not supported for integer division yet.

        if (rounding_mode == "floor") {
            return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
                input,
                value,
                binary::BinaryOpType::DIV_FLOOR,
                std::nullopt,
                output_mem_config,
                output_tensor,
                post_activations,
                lhs_activations,
                rhs_activations,
                /*fast_and_approximate_mode=*/std::nullopt,
                sub_core_grids,
                sub_device_id);
        }
        if (rounding_mode == "trunc") {
            return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
                input,
                value,
                binary::BinaryOpType::DIV_TRUNC,
                std::nullopt,
                output_mem_config,
                output_tensor,
                post_activations,
                lhs_activations,
                rhs_activations,
                /*fast_and_approximate_mode=*/std::nullopt,
                sub_core_grids,
                sub_device_id);
        }
        // rounding_mode = None
        TT_FATAL(
            (!output_dtype.has_value() || output_dtype == DataType::FLOAT32),
            "Incorrect output_dtype value for Integer Division(rounding_mode=None) ; valid input values are None or "
            "ttnn.float32");
        return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
            input,
            value,
            binary::BinaryOpType::DIV,
            std::nullopt,
            output_mem_config,
            output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            std::nullopt,  // fast_and_approximate_mode
            sub_core_grids,
            sub_device_id);
    }

    // Non-int32 inputs: with rounding_mode=None, use DIV directly; with "trunc"/"floor",
    // compute the float divide then apply trunc/floor rounding.
    if (!rounding_mode.has_value()) {
        return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
            input,
            value,
            binary::BinaryOpType::DIV,
            std::nullopt,
            output_mem_config,
            output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            fast_and_approximate_mode,
            sub_core_grids,
            sub_device_id);
    }

    TT_FATAL(
        (rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");

    // Workaround for a known bfloat16 fast_and_approximate divide bug (issue #43209):
    // 0/0 returns 0 instead of NaN, and sign-of-zero is lost. The pre-legacy-removal
    // path used a float32 typecast as a safety guard; we restore the same invariant by
    // suppressing fast_and_approximate on bfloat16 inside the rounding-mode branch.
    // The rounding_mode=None case is documented via the existing test skip in
    // tests/ttnn/unit_tests/operations/eltwise/test_binary_fp32.py. Remove this
    // workaround when #43209 is fixed.
    const bool suppress_fap = fast_and_approximate_mode && input.dtype() == DataType::BFLOAT16;
    const bool effective_fap = suppress_fap ? false : fast_and_approximate_mode;

    std::optional<Tensor> divided = divide(
        input,
        value,
        std::nullopt,
        output_mem_config,
        output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        effective_fap,
        sub_core_grids,
        sub_device_id);

    if (rounding_mode == "trunc") {
        return ttnn::trunc(divided.value(), output_mem_config, output_tensor, sub_core_grids);
    }
    return ttnn::floor(divided.value(), output_mem_config, output_tensor, sub_core_grids);
}

Tensor div(
    const Tensor& input_a,
    const Tensor& input_b,
    bool fast_and_approximate_mode,
    const std::optional<std::string>& rounding_mode,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    DataType input_dtype = input_a.dtype();
    const bool is_int32 = input_dtype == DataType::INT32 && input_b.dtype() == DataType::INT32;

    if (is_int32) {
        TT_FATAL(
            !fast_and_approximate_mode,
            "Integer Division does not support fast_and_approximate_mode=true {}",
            fast_and_approximate_mode);
        // fast_and_approximate_mode is not supported for integer division yet.

        if (rounding_mode == "floor") {
            return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
                input_a,
                input_b,
                binary::BinaryOpType::DIV_FLOOR,
                std::nullopt,
                output_mem_config,
                output_tensor,
                post_activations,
                lhs_activations,
                rhs_activations,
                /*fast_and_approximate_mode=*/std::nullopt,
                sub_core_grids,
                sub_device_id);
        }
        if (rounding_mode == "trunc") {
            return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
                input_a,
                input_b,
                binary::BinaryOpType::DIV_TRUNC,
                std::nullopt,
                output_mem_config,
                output_tensor,
                post_activations,
                lhs_activations,
                rhs_activations,
                /*fast_and_approximate_mode=*/std::nullopt,
                sub_core_grids,
                sub_device_id);
        }
        // rounding_mode = None
        TT_FATAL(
            (!output_dtype.has_value() || output_dtype == DataType::FLOAT32),
            "Incorrect output_dtype value for Integer Division(rounding_mode=None) ; valid input values are None or "
            "ttnn.float32");
        return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
            input_a,
            input_b,
            binary::BinaryOpType::DIV,
            std::nullopt,
            output_mem_config,
            output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            std::nullopt,  // fast_and_approximate_mode
            sub_core_grids,
            sub_device_id);
    }

    // Non-int32 inputs: with rounding_mode=None, use DIV directly; with "trunc"/"floor",
    // compute the float divide then apply trunc/floor rounding.
    if (!rounding_mode.has_value()) {
        return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
            input_a,
            input_b,
            binary::BinaryOpType::DIV,
            std::nullopt,
            output_mem_config,
            output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            fast_and_approximate_mode,
            sub_core_grids,
            sub_device_id);
    }

    TT_FATAL(
        (rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");

    // Workaround for a known bfloat16 fast_and_approximate divide bug (issue #43209):
    // 0/0 returns 0 instead of NaN, and sign-of-zero is lost. The pre-legacy-removal
    // path used a float32 typecast as a safety guard; we restore the same invariant by
    // suppressing fast_and_approximate on bfloat16 inside the rounding-mode branch.
    // The rounding_mode=None case is documented via the existing test skip in
    // tests/ttnn/unit_tests/operations/eltwise/test_binary_fp32.py. Remove this
    // workaround when #43209 is fixed.
    const bool suppress_fap = fast_and_approximate_mode && input_dtype == DataType::BFLOAT16;
    const bool effective_fap = suppress_fap ? false : fast_and_approximate_mode;

    std::optional<Tensor> divided = divide(
        input_a,
        input_b,
        std::nullopt,
        output_mem_config,
        output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        effective_fap,
        sub_core_grids,
        sub_device_id);

    if (rounding_mode == "trunc") {
        return ttnn::trunc(divided.value(), output_mem_config, output_tensor, sub_core_grids);
    }
    return ttnn::floor(divided.value(), output_mem_config, output_tensor, sub_core_grids);
}

Tensor div_no_nan(
    const Tensor& input_a, unary::ScalarVariant value, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    float value_f = std::visit([](auto v) -> float { return static_cast<float>(v); }, value);
    if (value_f == 0) {
        return ttnn::zeros_like(input_a);
    }
    return multiply(input_a, (1.0f / value_f));
}

Tensor div_no_nan(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    if (input_a.dtype() == DataType::FLOAT32 && input_b.dtype() == DataType::FLOAT32) {
        // Not using SFPU div op here since inf/nan handling is not required
        Tensor div_result = multiply(input_a, ttnn::reciprocal(input_b), std::nullopt, output_mem_config);
        return ttnn::where(ttnn::eqz(input_b, output_mem_config), 0.0f, div_result);
    }
    Tensor div_result = divide(input_a, input_b, std::nullopt, output_mem_config);
    return ttnn::where(ttnn::eqz(input_b, output_mem_config), 0.0f, div_result);
}

Tensor prelu(
    const Tensor& input, unary::ScalarVariant weight, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    float weight_f = std::visit([](auto v) -> float { return static_cast<float>(v); }, weight);
    return ttnn::prelu_sfpu(input, weight_f);
}

Tensor prelu(
    const Tensor& input, const std::array<float, 1>& weight, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    float scalar_weight = weight[0];
    return ttnn::prelu_sfpu(input, scalar_weight);
}

Tensor prelu(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const auto& s_a = input_a.logical_shape();
    const auto volume = input_b.logical_volume();
    TT_FATAL(
        s_a[1] == volume,
        "Mismatch of parameter numbers and input channel size. Found parameter numbers = {} and channel size = {}.",
        volume,
        s_a[1]);
    Tensor b = input_b;
    if (s_a.rank() > 2) {
        ttsl::SmallVector<uint32_t> reshape(s_a.rank(), 1);
        reshape[1] = s_a[1];
        b = ttnn::operations::experimental::quasar::reshape(input_b, ttnn::Shape(reshape));
    }

    Tensor result = ttnn::where(ttnn::ltz(input_a, output_mem_config), multiply(input_a, b), input_a);
    return result;
}

// REMAINDER result = input − (other * floor(input/other))
Tensor remainder(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_a,
        input_b,
        binary::BinaryOpType::REMAINDER,
        output_dtype,
        output_mem_config,
        output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        std::nullopt,
        sub_core_grids,
        sub_device_id);
}

Tensor remainder(
    const Tensor& input,
    unary::ScalarVariant scalar,
    const std::optional<const DataType>& /*output_dtype*/,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*post_activations*/,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*lhs_activations*/,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*rhs_activations*/,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& /*sub_device_id*/) {
    float scalar_f = std::visit([](auto v) -> float { return static_cast<float>(v); }, scalar);
    return ttnn::unary_remainder(input, scalar_f, output_mem_config, output_tensor, sub_core_grids);
}

// FMOD result = input − (other * trunc(input/other))
Tensor fmod(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_a,
        input_b,
        binary::BinaryOpType::FMOD,
        std::nullopt,
        output_mem_config,
        std::nullopt,
        {},
        {},
        {},
        std::nullopt,
        sub_core_grids,
        sub_device_id);
}

Tensor fmod(
    const Tensor& input,
    unary::ScalarVariant scalar,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<CoreRangeSet>& /*sub_core_grids*/,
    const std::optional<tt::tt_metal::SubDeviceId>& /*sub_device_id*/) {
    float scalar_f = std::visit([](auto v) -> float { return static_cast<float>(v); }, scalar);
    return ttnn::unary_fmod(input, scalar_f, output_mem_config);
}

Tensor floor_div(
    const Tensor& input_a, unary::ScalarVariant value, const std::optional<MemoryConfig>& output_mem_config) {
    float value_f = std::visit([](auto v) -> float { return static_cast<float>(v); }, value);
    if (value_f == 0) {
        float t_inf = std::numeric_limits<float>::infinity();
        float t_nan = std::nanf("");
        return ttnn::where(
            ttnn::eqz(input_a, output_mem_config),
            t_nan,
            multiply(ttnn::sign(input_a, output_mem_config), t_inf, std::nullopt, output_mem_config));
    }
    Tensor temp = multiply(input_a, (1.0f / value_f), std::nullopt, output_mem_config);
    return ttnn::floor(temp);
}

Tensor floor_div(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor temp = div(input_a, input_b, false, std::nullopt, std::nullopt, output_mem_config);
    Tensor result = div(input_a, input_b, false, "floor", std::nullopt, output_mem_config);
    // floor(nan, inf, -inf) = nan, inf, -inf
    return ttnn::where(
        logical_or(
            eq(temp, std::nanf("")),
            logical_or(
                eq(temp, std::numeric_limits<float>::infinity()), eq(temp, -std::numeric_limits<float>::infinity()))),
        temp,
        result);
}

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor outer(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    const ttnn::Shape& s_a = input_a.logical_shape();
    const ttnn::Shape& s_b = input_b.logical_shape();
    auto num_ones = [](const ttnn::Shape& s) -> uint32_t {
        uint32_t num1s = 0;
        for (uint32_t idx = 0; idx < 4; idx++) {
            num1s += (uint32_t)(s[idx] == 1);
        }
        return num1s;
    };

    // check if 3 dimensions are 1
    TT_FATAL((num_ones(s_a) >= 3), "3 dimensions are required to be 1 for use with outer product");
    TT_FATAL((num_ones(s_b) >= 3), "3 dimensions are required to be 1 for use with outer product");

    const bool skip_reshape_a = (s_a[0] == 1 && s_a[1] == 1 && s_a[2] >= 1 && s_a[3] == 1);
    const bool skip_reshape_b = (s_b[0] == 1 && s_b[1] == 1 && s_b[2] == 1 && s_b[3] >= 1);

    Tensor a_slim = input_a;
    Tensor b_slim = input_b;

    if (!skip_reshape_a) {
        uint32_t a_volume = s_a[0] * s_a[1] * s_a[2] * s_a[3];
        a_slim = ttnn::operations::experimental::quasar::reshape(
            input_a, ttnn::Shape{std::array<uint32_t, 4>{1, 1, a_volume, 1}});
    }
    if (!skip_reshape_b) {
        uint32_t b_volume = s_b[0] * s_b[1] * s_b[2] * s_b[3];
        b_slim = ttnn::operations::experimental::quasar::reshape(
            input_b, ttnn::Shape{std::array<uint32_t, 4>{1, 1, 1, b_volume}});
    }
    a_slim = ttnn::operations::experimental::quasar::to_layout(a_slim, ttnn::TILE_LAYOUT);
    b_slim = ttnn::operations::experimental::quasar::to_layout(b_slim, ttnn::TILE_LAYOUT);

    auto* device = ttnn::GetDefaultDevice();
    if (device != nullptr) {
        if (a_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            a_slim = a_slim.to_device(device);
        }
        if (b_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            b_slim = b_slim.to_device(device);
        }
    }

    return ttnn::matmul(a_slim, b_slim);
}

Tensor polyval(
    const Tensor& input_a, const std::vector<float>& coeffs, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(!coeffs.empty() && "coeffs should be 1 or more coefficients");
    if (coeffs.size() == 1) {
        return ttnn::full_like(input_a, coeffs[0]);
    }
    Tensor result = multiply(input_a, coeffs[0], std::nullopt, output_mem_config);
    for (int idx = 1; idx < coeffs.size() - 1; idx++) {
        result = add(result, coeffs[idx], std::nullopt, output_mem_config);
        result = multiply(input_a, result, std::nullopt, output_mem_config);
    }
    Tensor final_tensor = add(result, coeffs.back(), std::nullopt, output_mem_config);
    return final_tensor;
}

Tensor gcd(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& /*output_dtype*/,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_tensor_a,
        input_tensor_b,
        binary::BinaryOpType::GCD,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations);
}

Tensor lcm(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& /*output_dtype*/,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_tensor_a,
        input_tensor_b,
        binary::BinaryOpType::LCM,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations);
}

// power - floating point exponent
Tensor pow(
    const Tensor& input_a,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    float exponent_floor = std::floor(exponent);
    if (static_cast<int32_t>(exponent_floor) == exponent) {
        int32_t exp = exponent;
        return pow(input_a, exp, output_mem_config, output_tensor);
    }
    return ttnn::power(input_a, exponent, output_mem_config, output_tensor);
}

// power - integer exponent
Tensor pow(
    const Tensor& input,
    int32_t exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    // For exponents 0, 1, 2, 3: use iterative approach
    if (exponent == 0 || exponent == 1 || exponent == 2 || exponent == 3) {
        uint32_t exp = exponent;
        return ttnn::power_iterative(input, exp, output_mem_config, output_tensor);
    }
    return ttnn::power(input, unary::ScalarVariant(exponent), output_mem_config, output_tensor);
}

// power - tensor exponent
Tensor pow(
    const Tensor& input,
    const Tensor& exponent,
    const std::optional<const DataType>& /*dtype*/,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input,
        exponent,
        binary::BinaryOpType::POWER,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations);
}

// power - scalar input, tensor exponent
Tensor pow(
    float input_a,
    const Tensor& exponent,
    const std::optional<const DataType>& /*dtype*/,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations) {
    // As per binary infra, first input is always a tensor but this support needed for pytorch2 tracing
    // https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/docs/operations/aten.pow.Scalar.md

    Tensor input = ttnn::full_like(exponent, input_a);
    return pow(
        input,
        exponent,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations);
}

Tensor rsub(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_tensor_a,
        input_tensor_b,
        binary::BinaryOpType::RSUB,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations);
}

Tensor rsub(
    const Tensor& input_tensor_a,
    unary::ScalarVariant input_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_tensor_a,
        input_b,
        binary::BinaryOpType::RSUB,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations);
}

Tensor bias_gelu(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    return ttnn::operations::experimental::quasar::binary::detail::invoke_binary_ng(
        input_tensor_a_arg,
        input_tensor_b_arg,
        binary::BinaryOpType::BIAS_GELU,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        /*fast_and_approximate_mode=*/std::nullopt,
        sub_core_grids,
        sub_device_id);
}

Tensor bias_gelu(
    const Tensor& input_tensor_a,
    unary::ScalarVariant bias,
    const std::optional<const DataType>& /*dtype*/,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*post_activations*/,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*lhs_activations*/,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*rhs_activations*/,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    // Resolve sub_device_id to sub_core_grids so both add and gelu use the same core restriction
    auto resolved_sub_core_grids = sub_core_grids;
    if (sub_device_id.has_value()) {
        TT_FATAL(!sub_core_grids.has_value(), "Cannot specify both sub_core_grids and sub_device_id");
        auto* device = input_tensor_a.device();
        resolved_sub_core_grids =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
    }
    return ttnn::gelu(
        add(input_tensor_a,
            bias,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            {},
            {},
            {},
            resolved_sub_core_grids),
        true,
        memory_config,
        optional_output_tensor,
        resolved_sub_core_grids);
}

}  // namespace ttnn::operations::experimental::quasar::binary
