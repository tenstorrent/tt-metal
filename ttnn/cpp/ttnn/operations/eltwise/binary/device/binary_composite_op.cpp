// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include <variant>
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn {

using namespace operations;

// nextafter
Tensor nextafter(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const float eps = tt::tt_metal::hal::get_eps();
    Tensor result(input_a);
    {
        Tensor eps_gt(input_a);
        {
            eps_gt = ttnn::where(
                ttnn::gt(input_a, input_b, std::nullopt, output_mem_config),
                ttnn::add(input_a, eps, std::nullopt, output_mem_config),
                input_a);
        }
        result = ttnn::where(
            ttnn::lt(input_a, input_b, std::nullopt, output_mem_config),
            ttnn::subtract(input_a, eps, std::nullopt, output_mem_config),
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
    return ttnn::detail::invoke_binary_ng(
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
    return ttnn::detail::invoke_binary_ng(
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
    return ttnn::detail::invoke_binary_ng(
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
    ttsl::Span<const ttnn::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::unary::EltwiseUnaryWithParam> rhs_activations,
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
            return ttnn::detail::invoke_binary_ng(
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
            return ttnn::detail::invoke_binary_ng(
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
        return ttnn::detail::invoke_binary_ng(
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
        return ttnn::detail::invoke_binary_ng(
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

    std::optional<Tensor> divided = ttnn::divide(
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
    ttsl::Span<const ttnn::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::unary::EltwiseUnaryWithParam> rhs_activations,
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
            return ttnn::detail::invoke_binary_ng(
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
            return ttnn::detail::invoke_binary_ng(
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
        return ttnn::detail::invoke_binary_ng(
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
        return ttnn::detail::invoke_binary_ng(
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

    std::optional<Tensor> divided = ttnn::divide(
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
    return ttnn::multiply(input_a, (1.0f / value_f));
}

Tensor div_no_nan(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    if (input_a.dtype() == DataType::FLOAT32 && input_b.dtype() == DataType::FLOAT32) {
        // Not using SFPU div op here since inf/nan handling is not required
        Tensor div_result = ttnn::multiply(input_a, ttnn::reciprocal(input_b), std::nullopt, output_mem_config);
        return ttnn::where(ttnn::eqz(input_b, output_mem_config), 0.0f, div_result);
    }
    Tensor div_result = ttnn::divide(input_a, input_b, std::nullopt, output_mem_config);
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
        b = ttnn::reshape(input_b, ttnn::Shape(reshape));
    }

    Tensor result = ttnn::where(ttnn::ltz(input_a, output_mem_config), ttnn::multiply(input_a, b), input_a);
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
    return ttnn::detail::invoke_binary_ng(
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
    return ttnn::detail::invoke_binary_ng(
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
            ttnn::multiply(ttnn::sign(input_a, output_mem_config), t_inf, std::nullopt, output_mem_config));
    }
    Tensor temp = ttnn::multiply(input_a, (1.0f / value_f), std::nullopt, output_mem_config);
    return ttnn::floor(temp);
}

Tensor floor_div(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor temp = ttnn::div(input_a, input_b, false, std::nullopt, std::nullopt, output_mem_config);
    Tensor result = ttnn::div(input_a, input_b, false, "floor", std::nullopt, output_mem_config);
    // floor(nan, inf, -inf) = nan, inf, -inf
    return ttnn::where(
        ttnn::logical_or(
            ttnn::eq(temp, std::nanf("")),
            ttnn::logical_or(
                ttnn::eq(temp, std::numeric_limits<float>::infinity()),
                ttnn::eq(temp, -std::numeric_limits<float>::infinity()))),
        temp,
        result);
}

// outer(a, b) treats each input's last dim as a vector and broadcasts the
// leading dims: a:[..., N], b:[..., M] -> [..., N, M], equivalent to
// a.unsqueeze(-1) * b.unsqueeze(-2).
//
// Dispatch:
//  - INT32/UINT32: broadcast-multiply (matmul does not support integer accum).
//  - FLOAT32: broadcast-multiply for precision, not speed. matmul is actually
//    faster in device time here, but its FPU truncates the FP32 inputs before
//    multiplying, whereas the eltwise multiply is FP32-native.
//  - BFLOAT16/BFLOAT8_B: matmul when the effective batch is 1 (both inputs
//    have no leading dims beyond the vector), otherwise broadcast-multiply.
//    Rationale: the [N,1]x[1,M] tile-outer-product path is the fastest kernel
//    at batch=1, but the K=1 padding tax dominates once the workload scales
//    across cores, at which point broadcast-multiply wins (~2x by batch=128).
//
// Height-sharded inputs flow through unchanged: the shard is along the
// preserved dim, so unsqueeze's reshape and the downstream op both accept
// the layout. Width-, block-, and ND-sharded inputs are materialized as
// interleaved first (preserving the source buffer_type so L1-resident
// sharded inputs stay in L1). Output sharding remains caller-controlled via
// output_mem_config.
Tensor outer(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        input_a.logical_shape().rank() >= 1 && input_b.logical_shape().rank() >= 1,
        "ttnn.outer: inputs must be at least 1D, but got shapes {} and {}",
        input_a.logical_shape(),
        input_b.logical_shape());
    // Keep this whitelist in sync with the dtype list advertised by the
    // nanobind docstring for ttnn.outer. Anything outside it would otherwise
    // fail deeper in ttnn::reshape or ttnn::multiply with a less attributable
    // error.
    auto is_supported = [](DataType dt) {
        return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::FLOAT32 ||
               dt == DataType::INT32 || dt == DataType::UINT32;
    };
    TT_FATAL(
        is_supported(input_a.dtype()) && is_supported(input_b.dtype()),
        "ttnn.outer: unsupported dtype (got {} and {}); supported dtypes are BFLOAT16, BFLOAT8_B, FLOAT32, INT32, "
        "UINT32",
        input_a.dtype(),
        input_b.dtype());
    TT_FATAL(
        input_a.dtype() == input_b.dtype(),
        "ttnn.outer: inputs must have the same dtype, but got {} and {}",
        input_a.dtype(),
        input_b.dtype());
    auto deshard_unless_height = [](const Tensor& t) {
        const auto layout = t.memory_config().memory_layout();
        const bool keep_sharded =
            layout == TensorMemoryLayout::INTERLEAVED || layout == TensorMemoryLayout::HEIGHT_SHARDED;
        if (keep_sharded) {
            return t;
        }
        // to_memory_config (not sharded_to_interleaved): the latter early-returns
        // when the legacy shard_spec is empty, silently leaving ND_SHARDED tensors
        // un-desharded. Preserve the source buffer_type so L1-resident sharded
        // inputs stay in L1.
        return ttnn::to_memory_config(
            t, MemoryConfig{TensorMemoryLayout::INTERLEAVED, t.memory_config().buffer_type()});
    };
    const auto a_unsq = ttnn::unsqueeze(deshard_unless_height(input_a), -1);
    const auto b_unsq = ttnn::unsqueeze(deshard_unless_height(input_b), -2);

    const DataType dt = input_a.dtype();
    const bool is_integer = (dt == DataType::INT32 || dt == DataType::UINT32);
    const bool is_fp32 = (dt == DataType::FLOAT32);
    // Effective batch is the product of leading dims (everything except the
    // vector dim); a scalar leading shape means batch=1. Uses logical shape so
    // padded tile geometry doesn't leak into the dispatch decision.
    auto leading_volume = [](const Tensor& t) -> uint64_t {
        const auto& shape = t.logical_shape();
        uint64_t v = 1;
        for (int i = 0; i + 1 < static_cast<int>(shape.rank()); ++i) {
            v *= static_cast<uint64_t>(shape[i]);
        }
        return v;
    };
    const uint64_t batch = std::max<uint64_t>(leading_volume(input_a), leading_volume(input_b));
    const bool use_matmul = !is_integer && !is_fp32 && batch == 1;
    if (use_matmul) {
        // matmul requires TILE inputs and, unlike the binary_ng multiply path,
        // does not tilize row-major inputs on the way in. Tilize here so the
        // documented "any layout" contract holds for the matmul dispatch.
        const auto to_tile = [](const Tensor& t) {
            return t.layout() == Layout::TILE ? t : ttnn::to_layout(t, Layout::TILE);
        };
        return ttnn::matmul(
            to_tile(a_unsq), to_tile(b_unsq), /*transpose_a=*/false, /*transpose_b=*/false, output_mem_config);
    }
    return ttnn::multiply(a_unsq, b_unsq, std::nullopt, output_mem_config);
}

Tensor polyval(
    const Tensor& input_a, const std::vector<float>& coeffs, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(!coeffs.empty() && "coeffs should be 1 or more coefficients");
    if (coeffs.size() == 1) {
        return ttnn::full_like(input_a, coeffs[0]);
    }
    Tensor result = ttnn::multiply(input_a, coeffs[0], std::nullopt, output_mem_config);
    for (int idx = 1; idx < coeffs.size() - 1; idx++) {
        result = ttnn::add(result, coeffs[idx], std::nullopt, output_mem_config);
        result = ttnn::multiply(input_a, result, std::nullopt, output_mem_config);
    }
    Tensor final_tensor = ttnn::add(result, coeffs.back(), std::nullopt, output_mem_config);
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
    return ttnn::detail::invoke_binary_ng(
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
    return ttnn::detail::invoke_binary_ng(
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
    return ttnn::detail::invoke_binary_ng(
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
    return ttnn::detail::invoke_binary_ng(
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
    return ttnn::detail::invoke_binary_ng(
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
    return ttnn::detail::invoke_binary_ng(
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
        ttnn::add(
            input_tensor_a,
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

}  // namespace ttnn
