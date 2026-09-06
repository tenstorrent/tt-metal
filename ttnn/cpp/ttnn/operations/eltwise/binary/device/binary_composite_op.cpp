// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <string_view>
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

namespace {

std::optional<CoreRangeSet> resolve_sub_device_workers(
    const Tensor& input,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    if (!sub_device_id.has_value()) {
        return sub_core_grids;
    }

    TT_FATAL(!sub_core_grids.has_value(), "Cannot specify both sub_core_grids and sub_device_id");
    return input.device()->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
}

void validate_scalar_typecast(
    Layout layout, bool is_sharded, const std::optional<CoreRangeSet>& sub_core_grids, std::string_view context) {
    TT_FATAL(layout != Layout::ROW_MAJOR || !is_sharded, "{} does not support row-major sharded tensors", context);
    TT_FATAL(
        !sub_core_grids.has_value() || (layout == Layout::TILE && !is_sharded),
        "{} on a restricted grid requires a tiled interleaved tensor",
        context);
}

struct PromotedScalarInput {
    Tensor input;
    std::optional<CoreRangeSet> sub_core_grids;
    std::optional<MemoryConfig> memory_config;
};

PromotedScalarInput promote_int32_scalar_input(
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    auto operation_sub_core_grids = resolve_sub_device_workers(input, sub_core_grids, sub_device_id);
    Tensor operation_input = input;
    if ((input.layout() == Layout::ROW_MAJOR && (input.is_sharded() || operation_sub_core_grids.has_value())) ||
        (input.is_sharded() && operation_sub_core_grids.has_value())) {
        // Use unary TYPECAST's full-tile staging for restricted row-major/sharded grids.
        // Sharded inputs use the existing tilize path, as binary_ng did before
        // promotion, so later output memory-layout changes also use tile pages.
        if (input.is_sharded() && operation_sub_core_grids.has_value()) {
            TT_FATAL(
                operation_sub_core_grids->contains(input.shard_spec()->grid),
                "INT32 scalar promotion requires the requested grid to contain all input shard cores");
        }
        const auto typecast_input =
            input.is_sharded()
                ? ttnn::to_layout(input, Layout::TILE, std::nullopt, std::nullopt, operation_sub_core_grids)
                : input;
        operation_input = unary::detail::unary_impl(
            typecast_input,
            {unary::EltwiseUnaryWithParam{
                unary::UnaryOpType::TYPECAST,
                {static_cast<float>(DataType::INT32), static_cast<float>(DataType::FLOAT32)}}},
            std::nullopt,
            std::nullopt,
            operation_sub_core_grids);
    } else {
        operation_input =
            ttnn::typecast(input, DataType::FLOAT32, std::nullopt, std::nullopt, operation_sub_core_grids);
    }
    // Keep tilized shard pages through the arithmetic; untilize performs any
    // requested output memory-layout conversion using the input shard workers.
    auto operation_mem_config =
        operation_input.layout() != input.layout() ? operation_input.memory_config() : output_mem_config;
    return {std::move(operation_input), std::move(operation_sub_core_grids), std::move(operation_mem_config)};
}

Tensor restore_scalar_output_layout(
    const Tensor& input,
    Layout operation_layout,
    const Tensor& output,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // Sharded untilize selects its workers from the shard spec, not a subgrid argument.
    const auto untilize_sub_core_grids = output.is_sharded() ? std::nullopt : sub_core_grids;
    return operation_layout == input.layout()
               ? output
               : ttnn::to_layout(output, input.layout(), std::nullopt, output_mem_config, untilize_sub_core_grids);
}

}  // namespace

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

    if (is_int32 && std::holds_alternative<float>(value)) {
        // Dispatch from the scalar's type, not its value: even 2.0 must promote
        // before division/rounding, whereas an integer 2 keeps exact INT32 division.
        // Otherwise binary_ng truncates the floating divisor to an integer.
        TT_FATAL(
            !(input.layout() == Layout::ROW_MAJOR && output_tensor.has_value()),
            "Optional output tensor with Row Major input is not supported right now for Elementwise operations");
        const auto [operation_input, operation_sub_core_grids, operation_mem_config] =
            promote_int32_scalar_input(input, output_mem_config, sub_core_grids, sub_device_id);
        const std::optional<const DataType> requested_dtype =
            output_tensor.has_value() ? std::optional<const DataType>{output_tensor->dtype()} : output_dtype;
        if (rounding_mode.has_value() && requested_dtype.has_value() &&
            !tt::tt_metal::is_floating_point(*requested_dtype)) {
            // The floating path casts integer outputs after rounding. Retain the
            // standalone typecast's layout/grid restrictions for that final step.
            validate_scalar_typecast(
                operation_input.layout(),
                output_tensor.has_value() ? output_tensor->is_sharded()
                                          : operation_mem_config.value_or(operation_input.memory_config()).is_sharded(),
                operation_sub_core_grids,
                "Division output typecast");
        }
        const auto result = ttnn::div(
            operation_input,
            value,
            fast_and_approximate_mode,
            rounding_mode,
            output_dtype,
            operation_mem_config,
            output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            operation_sub_core_grids,
            std::nullopt);
        return restore_scalar_output_layout(
            input, operation_input.layout(), result, output_mem_config, operation_sub_core_grids);
    }

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
                output_dtype,
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
                output_dtype,
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
            output_dtype,
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
            output_dtype,
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

    // A preallocated output pins the result dtype exactly like an explicit dtype does, so the two
    // have to agree before either can decide how the quotient is rounded.
    TT_FATAL(
        !output_dtype.has_value() || !output_tensor.has_value() || *output_dtype == output_tensor->dtype(),
        "If both output dtype and output tensor are provided, their dtypes should match");
    const std::optional<const DataType> requested_dtype =
        output_tensor.has_value() ? std::optional<const DataType>{output_tensor->dtype()} : output_dtype;

    // The quotient has to stay in floating point until it is rounded: narrowing it to an integer
    // dtype first truncates toward zero, which would turn floor(-3.5) into -3 instead of -4. An
    // integer destination is therefore filled by a typecast after the rounding step.
    const bool cast_after_rounding = requested_dtype.has_value() && !tt::tt_metal::is_floating_point(*requested_dtype);
    const std::optional<const DataType> quotient_dtype =
        cast_after_rounding ? std::optional<const DataType>{} : output_dtype;
    const std::optional<Tensor> quotient_output = cast_after_rounding ? std::optional<Tensor>{} : output_tensor;

    std::optional<Tensor> divided = ttnn::divide(
        input,
        value,
        quotient_dtype,
        output_mem_config,
        quotient_output,
        post_activations,
        lhs_activations,
        rhs_activations,
        effective_fap,
        sub_core_grids,
        sub_device_id);

    Tensor rounded = (rounding_mode == "trunc")
                         ? ttnn::trunc(divided.value(), output_mem_config, quotient_output, sub_core_grids)
                         : ttnn::floor(divided.value(), output_mem_config, quotient_output, sub_core_grids);
    if (!cast_after_rounding) {
        return rounded;
    }
    return ttnn::typecast(rounded, *requested_dtype, output_mem_config, output_tensor, sub_core_grids);
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
                output_dtype,
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
                output_dtype,
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
            output_dtype,
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
            output_dtype,
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

    // A preallocated output pins the result dtype exactly like an explicit dtype does, so the two
    // have to agree before either can decide how the quotient is rounded.
    TT_FATAL(
        !output_dtype.has_value() || !output_tensor.has_value() || *output_dtype == output_tensor->dtype(),
        "If both output dtype and output tensor are provided, their dtypes should match");
    const std::optional<const DataType> requested_dtype =
        output_tensor.has_value() ? std::optional<const DataType>{output_tensor->dtype()} : output_dtype;

    // The quotient has to stay in floating point until it is rounded: narrowing it to an integer
    // dtype first truncates toward zero, which would turn floor(-3.5) into -3 instead of -4. An
    // integer destination is therefore filled by a typecast after the rounding step.
    const bool cast_after_rounding = requested_dtype.has_value() && !tt::tt_metal::is_floating_point(*requested_dtype);
    const std::optional<const DataType> quotient_dtype =
        cast_after_rounding ? std::optional<const DataType>{} : output_dtype;
    const std::optional<Tensor> quotient_output = cast_after_rounding ? std::optional<Tensor>{} : output_tensor;

    std::optional<Tensor> divided = ttnn::divide(
        input_a,
        input_b,
        quotient_dtype,
        output_mem_config,
        quotient_output,
        post_activations,
        lhs_activations,
        rhs_activations,
        effective_fap,
        sub_core_grids,
        sub_device_id);

    Tensor rounded = (rounding_mode == "trunc")
                         ? ttnn::trunc(divided.value(), output_mem_config, quotient_output, sub_core_grids)
                         : ttnn::floor(divided.value(), output_mem_config, quotient_output, sub_core_grids);
    if (!cast_after_rounding) {
        return rounded;
    }
    return ttnn::typecast(rounded, *requested_dtype, output_mem_config, output_tensor, sub_core_grids);
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
        ttsl::SmallVector<std::uint32_t> reshape(s_a.rank(), 1);
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
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    Tensor operation_input = input;
    auto operation_sub_core_grids = sub_core_grids;
    auto operation_sub_device_id = sub_device_id;
    auto operation_mem_config = output_mem_config;
    if (input.dtype() == DataType::INT32 && std::holds_alternative<float>(scalar)) {
        // binary_ng did not support preallocated outputs when tilizing row-major
        // inputs. Preserve that restriction rather than writing tile data to them.
        TT_FATAL(
            !(input.layout() == Layout::ROW_MAJOR && input.is_sharded() && output_tensor.has_value()),
            "Optional output tensor with row-major sharded scalar promotion is not supported");
        auto [promoted_input, promoted_sub_core_grids, promoted_mem_config] =
            promote_int32_scalar_input(input, output_mem_config, sub_core_grids, sub_device_id);
        operation_input = std::move(promoted_input);
        operation_sub_core_grids = std::move(promoted_sub_core_grids);
        operation_mem_config = std::move(promoted_mem_config);
        operation_sub_device_id = std::nullopt;
    }

    // The unary SFPU fast path does not support INT32. Float scalars promote INT32 inputs
    // above; integral scalars must route through binary_ng.
    if (operation_input.dtype() != DataType::INT32 && !output_dtype.has_value() &&
        !operation_sub_device_id.has_value() && post_activations.empty() && lhs_activations.empty() &&
        rhs_activations.empty()) {
        // Native floating inputs already support mixed floating-point outputs in
        // unary_remainder; preserve their existing packing behavior.
        if (input.dtype() != DataType::INT32 || !output_tensor.has_value() ||
            output_tensor->dtype() == operation_input.dtype()) {
            return restore_scalar_output_layout(
                input,
                operation_input.layout(),
                ttnn::unary_remainder(
                    operation_input, scalar, operation_mem_config, output_tensor, operation_sub_core_grids),
                output_mem_config,
                operation_sub_core_grids);
        }

        // Fused unary TYPECAST supports row-major interleaved subgrids without
        // standalone typecast. Keep the existing guards for sharded outputs and
        // for the separate output-conversion path below.
        const bool fused_row_major_bf16 = output_tensor->dtype() == DataType::BFLOAT16 &&
                                          operation_input.layout() == Layout::ROW_MAJOR &&
                                          output_tensor->layout() == Layout::ROW_MAJOR &&
                                          !operation_input.is_sharded() && !output_tensor->is_sharded();
        if (!fused_row_major_bf16) {
            validate_scalar_typecast(
                operation_input.layout(),
                output_tensor->is_sharded(),
                operation_sub_core_grids,
                "Remainder output typecast");
        }

        // Fuse BF16 conversion to avoid a separate pass, but retain TYPECAST's
        // explicit rounding: direct unary packing is not bit-equivalent.
        // Leave integer outputs and BFLOAT8_B's precise packing path unchanged.
        if (output_tensor->dtype() == DataType::BFLOAT16) {
            return unary::detail::unary_impl(
                operation_input,
                {unary::EltwiseUnaryWithParam{unary::UnaryOpType::REMAINDER, std::get<float>(scalar)},
                 unary::EltwiseUnaryWithParam{
                     unary::UnaryOpType::TYPECAST,
                     {static_cast<float>(DataType::FLOAT32), static_cast<float>(DataType::BFLOAT16)}}},
                output_mem_config,
                output_tensor,
                operation_sub_core_grids);
        }

        // The intermediate inherits the input layout and the requested output's sharding.
        const Tensor operation_output = ttnn::unary_remainder(
            operation_input, scalar, output_tensor->memory_config(), std::nullopt, operation_sub_core_grids);
        return ttnn::typecast(
            operation_output, output_tensor->dtype(), std::nullopt, output_tensor, operation_sub_core_grids);
    }
    auto result = ttnn::detail::invoke_binary_ng(
        operation_input,
        scalar,
        binary::BinaryOpType::REMAINDER,
        output_dtype,
        operation_mem_config,
        output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        std::nullopt,
        operation_sub_core_grids,
        operation_sub_device_id);
    return restore_scalar_output_layout(
        input, operation_input.layout(), result, output_mem_config, operation_sub_core_grids);
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
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    if (input.dtype() == DataType::INT32 && std::holds_alternative<float>(scalar)) {
        const auto [operation_input, operation_sub_core_grids, operation_mem_config] =
            promote_int32_scalar_input(input, output_mem_config, sub_core_grids, sub_device_id);
        const float scalar_f = std::get<float>(scalar);
        return restore_scalar_output_layout(
            input,
            operation_input.layout(),
            ttnn::unary_fmod(operation_input, scalar_f, operation_mem_config, std::nullopt, operation_sub_core_grids),
            output_mem_config,
            operation_sub_core_grids);
    }
    if (input.dtype() == DataType::INT32 || sub_device_id.has_value()) {
        return ttnn::detail::invoke_binary_ng(
            input,
            scalar,
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
    const float scalar_f = std::visit([](auto value) -> float { return static_cast<float>(value); }, scalar);
    return ttnn::unary_fmod(input, scalar_f, output_mem_config, std::nullopt, sub_core_grids);
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
    // floor(inf, -inf) = inf, -inf. isinf tests both in a single SFPU pass,
    // replacing two eq's and a logical_or. The dropped eq(temp, nan) term was
    // always false under IEEE, so NaN selects the floored value here exactly as
    // it did before; isinf (rather than !isfinite) keeps that branch identical
    // without relying on floor propagating NaN.
    return ttnn::where(ttnn::isinf(temp, output_mem_config), temp, result);
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
    auto leading_volume = [](const Tensor& t) -> std::uint64_t {
        const auto& shape = t.logical_shape();
        std::uint64_t v = 1;
        for (int i = 0; i + 1 < static_cast<int>(shape.rank()); ++i) {
            v *= static_cast<std::uint64_t>(shape[i]);
        }
        return v;
    };
    const std::uint64_t batch = std::max<std::uint64_t>(leading_volume(input_a), leading_volume(input_b));
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
    if (static_cast<std::int32_t>(exponent_floor) == exponent) {
        std::int32_t exp = exponent;
        return pow(input_a, exp, output_mem_config, output_tensor);
    }
    return ttnn::power(input_a, exponent, output_mem_config, output_tensor);
}

// power - integer exponent
Tensor pow(
    const Tensor& input,
    std::int32_t exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    // For exponents 0, 1, 2, 3: use iterative approach
    if (exponent == 0 || exponent == 1 || exponent == 2 || exponent == 3) {
        std::uint32_t exp = exponent;
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
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& fast_and_approximate_mode) {
    return ttnn::detail::invoke_binary_ng(
        input_tensor_a,
        input_tensor_b,
        binary::BinaryOpType::RSUB,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        ttnn::detail::resolve_fast_and_approximate_mode(fast_and_approximate_mode));
}

Tensor rsub(
    const Tensor& input_tensor_a,
    unary::ScalarVariant input_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& fast_and_approximate_mode) {
    return ttnn::detail::invoke_binary_ng(
        input_tensor_a,
        input_b,
        binary::BinaryOpType::RSUB,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        ttnn::detail::resolve_fast_and_approximate_mode(fast_and_approximate_mode));
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
            /*fast_and_approximate_mode*/ std::nullopt,
            resolved_sub_core_grids),
        true,
        memory_config,
        optional_output_tensor,
        resolved_sub_core_grids);
}

// At/below this width the intermediates are worth keeping in L1: it skips the DRAM round-trip
// between the composed ops. 3072 is the K3 routed-expert moe_intermediate_size.
constexpr uint32_t SITU_GLU_L1_MAX_HIDDEN = 3072;

// Width alone does not bound the intermediates -- their size is the whole volume. Three are
// live at the peak (softcap(gate) and sigmoid(gate) are still alive when their multiply
// allocates situ_a), and an interleaved-L1 buffer that does not fit is a hard allocator
// failure rather than a DRAM fallback, so the token count has to be checked too.
constexpr uint64_t SITU_GLU_L1_PEAK_INTERMEDIATES = 3;
// Fraction of total L1 the intermediates may claim, leaving room for the ops' CBs.
constexpr uint64_t SITU_GLU_L1_BUDGET_NUM = 3;
constexpr uint64_t SITU_GLU_L1_BUDGET_DEN = 4;

static bool situ_glu_intermediates_fit_l1(const Tensor& gate) {
    const auto& allocator = gate.device()->allocator();
    const uint64_t l1_total = static_cast<uint64_t>(allocator->get_bank_size(tt::tt_metal::BufferType::L1)) *
                              allocator->get_num_banks(tt::tt_metal::BufferType::L1);
    const uint64_t peak = SITU_GLU_L1_PEAK_INTERMEDIATES * gate.buffer()->size();
    return peak * SITU_GLU_L1_BUDGET_DEN <= l1_total * SITU_GLU_L1_BUDGET_NUM;
}

Tensor situ_glu(
    const Tensor& gate,
    const Tensor& up,
    float beta1,
    float beta2,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    using namespace operations::unary;

    // softcap precomputes 1/beta, so zero would emit inf.
    TT_FATAL(beta1 != 0.0f && beta2 != 0.0f, "situ_glu: beta1 and beta2 must be non-zero");

    // The composed unaries take sub_core_grids but no sub_device_id, so resolve here and let every
    // step share one core restriction.
    auto cores = sub_core_grids;
    if (sub_device_id.has_value()) {
        TT_FATAL(!sub_core_grids.has_value(), "Cannot specify both sub_core_grids and sub_device_id");
        cores = gate.device()->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
    }
    if (!cores.has_value()) {
        // Unrestricted, the composed ops fall back to the worker cores of get_sub_device_ids().front().
        // That is the whole grid only while the device is unpartitioned: the default manager holds one
        // sub-device spanning it. Once a custom manager is loaded, front() is sub-device 0 -- some
        // arbitrary strip -- and silently landing there is never what a caller means.
        const auto& loaded_sub_devices = gate.device()->get_sub_device_ids();
        TT_FATAL(
            loaded_sub_devices.size() == 1,
            "situ_glu: {} sub-devices are loaded, so leaving the cores unrestricted would run on "
            "sub-device 0 rather than the full grid. Pass sub_core_grids or sub_device_id.",
            loaded_sub_devices.size());
    }

    // A core restriction means another op is running concurrently on the complementary cores, and an
    // interleaved-L1 buffer comes from the global allocator: it takes L1 on every worker core, the
    // other op's included, growing down toward that op's circular buffers. A program only re-checks
    // its CB region against live L1 buffers when it is enqueued, and the concurrent op is already in
    // flight by then, so an overlap is silent corruption rather than a throw.
    //
    // Declining the L1 fast path below is not enough to rule that out: the intermediates then follow
    // the output placement, which is interleaved L1 whenever the caller asks for it or hands in an
    // interleaved-L1 gate with no output_mem_config. Sharded L1 stays safe -- its shard spec confines
    // it to named cores -- so only the interleaved case is rejected.
    const MemoryConfig effective_out = output_mem_config.value_or(gate.memory_config());
    TT_FATAL(
        !(cores.has_value() && effective_out.is_l1() && !effective_out.is_sharded()),
        "situ_glu: a core restriction cannot be combined with an interleaved-L1 output, which would "
        "take L1 on the cores restricted away. Use DRAM or a sharded L1 memory config.");

    // Sharded inputs keep the ops' own placement: interleaved-L1 intermediates against a sharded
    // input would add an unshard/reshard round-trip, which is the opposite of the point here.
    const bool use_l1 = !gate.is_sharded() && !cores.has_value() &&
                        gate.logical_shape()[-1] <= SITU_GLU_L1_MAX_HIDDEN && situ_glu_intermediates_fit_l1(gate);
    const std::optional<MemoryConfig> interm_mem =
        use_l1 ? std::optional<MemoryConfig>(ttnn::L1_MEMORY_CONFIG) : output_mem_config;

    Tensor situ_a = ttnn::softcap(gate, beta1, interm_mem, std::nullopt, cores);
    {
        Tensor gate_sigmoid =
            ttnn::sigmoid(gate, static_cast<int>(VecMode::RC), SigmoidMode::ACCURATE, interm_mem, std::nullopt, cores);
        ttnn::multiply_(situ_a, gate_sigmoid, {}, {}, {}, std::nullopt, cores);
    }
    Tensor up_half = ttnn::softcap(up, beta2, interm_mem, std::nullopt, cores);

    // Without L1 the intermediates already sit at the output placement, so the last multiply can
    // accumulate in place -- one buffer fewer to allocate and free, which matters when this runs
    // overlapped with an op that is handed whatever DRAM is freed here.
    if (!use_l1) {
        ttnn::multiply_(situ_a, up_half, {}, {}, {}, std::nullopt, cores);
        return situ_a;
    }
    // Pin the output placement, or multiply would inherit situ_a's L1 config and make placement
    // depend on the hidden dim.
    return ttnn::multiply(situ_a, up_half, std::nullopt, effective_out);
}

}  // namespace ttnn
