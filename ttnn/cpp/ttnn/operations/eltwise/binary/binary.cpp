
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary.hpp"

#include "device/binary_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"

// Implementation macros for binary operations (must match declarations in binary.hpp)
#define TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(NAME, OP_TYPE)                             \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        const Tensor& rhs,                                                           \
        const std::optional<const DataType>& output_dtype,                           \
        const std::optional<MemoryConfig>& memory_config,                            \
        const std::optional<Tensor>& output,                                         \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        const std::optional<bool>& use_legacy,                                       \
        const std::optional<CoreRangeSet>& sub_core_grids) {                         \
        return ttnn::detail::invoke_binary_ng(                                       \
            lhs,                                                                     \
            rhs,                                                                     \
            operations::binary::BinaryOpType::OP_TYPE,                               \
            output_dtype,                                                            \
            memory_config,                                                           \
            output,                                                                  \
            post_activations,                                                        \
            lhs_activations,                                                         \
            rhs_activations,                                                         \
            use_legacy,                                                              \
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids);                                                         \
    }

#define TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(NAME, OP_TYPE)                              \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        float rhs,                                                                   \
        const std::optional<const DataType>& output_dtype,                           \
        const std::optional<MemoryConfig>& memory_config,                            \
        const std::optional<Tensor>& output,                                         \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        const std::optional<bool>& use_legacy,                                       \
        const std::optional<CoreRangeSet>& sub_core_grids) {                         \
        return ttnn::detail::invoke_binary_ng(                                       \
            lhs,                                                                     \
            rhs,                                                                     \
            operations::binary::BinaryOpType::OP_TYPE,                               \
            output_dtype,                                                            \
            memory_config,                                                           \
            output,                                                                  \
            post_activations,                                                        \
            lhs_activations,                                                         \
            rhs_activations,                                                         \
            use_legacy,                                                              \
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids);                                                         \
    }

#define TTNN_BINARY_OP_INPLACE_IMPL(NAME, OP_TYPE)                                   \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        const Tensor& rhs,                                                           \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        std::optional<bool> use_legacy,                                              \
        const std::optional<CoreRangeSet>& sub_core_grids) {                         \
        return ttnn::detail::invoke_binary_ng(                                       \
            lhs,                                                                     \
            rhs,                                                                     \
            operations::binary::BinaryOpType::OP_TYPE,                               \
            std::nullopt,                                                            \
            std::nullopt,                                                            \
            lhs,                                                                     \
            post_activations,                                                        \
            lhs_activations,                                                         \
            rhs_activations,                                                         \
            use_legacy,                                                              \
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids);                                                         \
    }                                                                                \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        float rhs,                                                                   \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        std::optional<bool> use_legacy,                                              \
        const std::optional<CoreRangeSet>& sub_core_grids) {                         \
        return ttnn::detail::invoke_binary_ng(                                       \
            lhs,                                                                     \
            rhs,                                                                     \
            operations::binary::BinaryOpType::OP_TYPE,                               \
            std::nullopt,                                                            \
            std::nullopt,                                                            \
            lhs,                                                                     \
            post_activations,                                                        \
            lhs_activations,                                                         \
            rhs_activations,                                                         \
            use_legacy,                                                              \
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids);                                                         \
    }

#define TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL(NAME, OP_TYPE)                                            \
    Tensor NAME(                                                                                         \
        const Tensor& lhs,                                                                               \
        const Tensor& rhs,                                                                               \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,                     \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,                      \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,                      \
        std::optional<bool> use_legacy,                                                                  \
        const std::optional<CoreRangeSet>& sub_core_grids) {                                             \
        return operations::binary::inplace_relational_binary<operations::binary::BinaryOpType::OP_TYPE>( \
            lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);   \
    }                                                                                                    \
    Tensor NAME(                                                                                         \
        const Tensor& lhs,                                                                               \
        float rhs,                                                                                       \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,                     \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,                      \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,                      \
        std::optional<bool> use_legacy,                                                                  \
        const std::optional<CoreRangeSet>& sub_core_grids) {                                             \
        return operations::binary::inplace_relational_binary<operations::binary::BinaryOpType::OP_TYPE>( \
            lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);   \
    }

#define TTNN_BINARY_OP_INPLACE_INVOKE_IMPL(NAME, OP_TYPE)                            \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        const Tensor& rhs,                                                           \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        std::optional<bool> use_legacy,                                              \
        const std::optional<CoreRangeSet>& sub_core_grids) {                         \
        return ttnn::detail::invoke_binary_ng(                                       \
            lhs,                                                                     \
            rhs,                                                                     \
            operations::binary::BinaryOpType::OP_TYPE,                               \
            std::nullopt,                                                            \
            std::nullopt,                                                            \
            lhs,                                                                     \
            post_activations,                                                        \
            lhs_activations,                                                         \
            rhs_activations,                                                         \
            use_legacy,                                                              \
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids);                                                         \
    }                                                                                \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        float rhs,                                                                   \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        std::optional<bool> use_legacy,                                              \
        const std::optional<CoreRangeSet>& sub_core_grids) {                         \
        return ttnn::detail::invoke_binary_ng(                                       \
            lhs,                                                                     \
            rhs,                                                                     \
            operations::binary::BinaryOpType::OP_TYPE,                               \
            std::nullopt,                                                            \
            std::nullopt,                                                            \
            lhs,                                                                     \
            post_activations,                                                        \
            lhs_activations,                                                         \
            rhs_activations,                                                         \
            use_legacy,                                                              \
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids);                                                         \
    }

namespace ttnn::operations::binary {
namespace detail {

inline Tensor to_dtype(const Tensor& input, DataType dtype) {
    if (input.dtype() == dtype) {
        return input;
    }

    return ttnn::typecast(input, dtype);
}

inline float to_dtype(float input, [[maybe_unused]] DataType dtype) { return input; }

inline bool is_block_format(DataType dtype) {
    using enum DataType;
    switch (dtype) {
        case BFLOAT4_B:
        case BFLOAT8_B: return true;
        default: return false;
    }
}

inline bool is_layout_or_scalar(const Tensor& input, Layout layout) { return input.layout() == layout; }

inline bool is_layout_or_scalar([[maybe_unused]] float input, [[maybe_unused]] Layout layout) { return true; }

inline Tensor to_layout(const Tensor& input, Layout layout) {
    if (detail::is_layout_or_scalar(input, layout)) {
        return input;
    }

    return ttnn::to_layout(input, layout);
}

inline float to_layout(float input, [[maybe_unused]] Layout layout) { return input; }

inline bool needs_typecast_to_bfloat16(BinaryOpType op, const Tensor& input) {
    if (not detail::is_block_format(input.dtype())) {
        return false;
    }

    using enum BinaryOpType;

    return op != ADD and op != SUB and op != MUL;
}

inline bool needs_typecast_to_bfloat16(BinaryOpType op, const Tensor& input, [[maybe_unused]] float other) {
    return detail::needs_typecast_to_bfloat16(op, input);
}

inline bool needs_typecast_to_bfloat16(BinaryOpType op, const Tensor& input, const Tensor& other) {
    if (not detail::is_block_format(input.dtype())) {
        return false;
    }

    using enum BinaryOpType;

    if (op != ADD and op != SUB and op != MUL) {
        return true;
    }

    const auto& input_shape = input.logical_shape();
    const auto& other_shape = other.logical_shape();

    return (input_shape[-2] == 1 and other_shape[-2] > 1) or (input_shape[-1] == 1 and other_shape[-1] > 1);
}

inline bool needs_typecast_to_bfloat16(
    [[maybe_unused]] BinaryOpType op, [[maybe_unused]] float input, [[maybe_unused]] const Tensor& other) {
    return false;
}

constexpr bool is_associative(BinaryOpType op) {
    return op == BinaryOpType::ADD || op == BinaryOpType::MUL || op == BinaryOpType::EQ || op == BinaryOpType::NE ||
           op == BinaryOpType::LOGICAL_AND || op == BinaryOpType::LOGICAL_OR || op == BinaryOpType::LOGADDEXP ||
           op == BinaryOpType::LOGADDEXP2 || op == BinaryOpType::LOGICAL_XOR || op == BinaryOpType::MAXIMUM ||
           op == BinaryOpType::MINIMUM || op == BinaryOpType::GCD || op == BinaryOpType::LCM;
}

// Tensor - Scalar
inline Tensor binary_impl(
    operations::binary::BinaryOpType binary_op_type,
    const ttnn::Tensor& lhs,
    const float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    auto output_tensor = lhs;
    if (binary_op_type == BinaryOpType::GT) {
        output_tensor = ttnn::gt_unary(lhs, rhs, memory_config, output);
    } else if (binary_op_type == BinaryOpType::LT) {
        output_tensor = ttnn::lt_unary(lhs, rhs, memory_config, output);
    } else if (binary_op_type == BinaryOpType::NE) {
        output_tensor = ttnn::ne_unary(lhs, rhs, memory_config, output);
    } else if (binary_op_type == BinaryOpType::GE) {
        output_tensor = ttnn::ge_unary(lhs, rhs, memory_config, output);
    } else if (binary_op_type == BinaryOpType::LE) {
        output_tensor = ttnn::le_unary(lhs, rhs, memory_config, output);
    } else if (binary_op_type == BinaryOpType::EQ) {
        output_tensor = ttnn::eq_unary(lhs, rhs, memory_config, output);
    } else {
        TT_THROW("Unsupported operation");
    }
    if (dtype.has_value()) {
        output_tensor = ttnn::typecast(output_tensor, *dtype, std::nullopt, output);
    }
    return output_tensor;
}

// Scalar - Tensor
inline Tensor binary_impl(
    operations::binary::BinaryOpType binary_op_type,
    const float lhs,
    const ttnn::Tensor& rhs,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    if (binary_op_type == BinaryOpType::GE) {
        return ttnn::gez(ttnn::sub_sfpu(lhs, rhs, memory_config), memory_config, output);
    }
    if (binary_op_type == BinaryOpType::LE) {
        return ttnn::lez(ttnn::sub_sfpu(lhs, rhs, memory_config), memory_config, output);
    }
    if (binary_op_type == BinaryOpType::EQ) {
        return ttnn::eqz(ttnn::sub_sfpu(lhs, rhs, memory_config), memory_config, output);
    }

    TT_THROW("Unsupported operation");
}

inline auto preprocess_inputs(BinaryOpType binary_op_type, Tensor a, Tensor b) {
    // TODO: #7731 (Remove calls to repeat)
    constexpr auto repeat_smaller = [](const Tensor& first, Tensor& second) {
        const auto& first_shape = first.logical_shape();
        const auto& second_shape = second.logical_shape();
        // repeats second if it is smaller
        if (first_shape.rank() == 4 and second_shape.rank() == 4 and first_shape[0] > second_shape[0]) {
            TT_FATAL(second_shape[0] == 1, "Dimension trying to broadcast is not equal to 1");
            Shape repeats(std::array<uint32_t, 4>{first_shape[0], 1, 1, 1});
            second = ttnn::repeat(second, repeats);
        }
        // repeats second if it is smaller
        if (first_shape.rank() >= 3 and second_shape.rank() >= 3 and first_shape[-3] > second_shape[-3]) {
            TT_FATAL(second_shape[-3] == 1, "Dimension trying to broadcast is not equal to 1");
            int rank_a = first_shape.rank();
            std::vector<uint32_t> repeat_dim(rank_a, 1);
            repeat_dim[rank_a - 3] = first_shape[rank_a - 3];
            Shape repeats(repeat_dim);
            second = ttnn::repeat(second, repeats);
        }
    };

    repeat_smaller(a, b);
    repeat_smaller(b, a);

    // Swap tensors if a needs to be broadcasted to b
    if (detail::is_associative(binary_op_type) and a.logical_volume() < b.logical_volume()) {
        return std::make_tuple(b, a);
    }

    return std::make_tuple(a, b);
}

inline auto any_sharded_block_format(const Tensor& a, const auto& b) {
    if (a.is_sharded() and is_block_format(a.dtype())) {
        return true;
    }

    if constexpr (requires { b.is_sharded(); }) {
        if (b.is_sharded() and is_block_format(b.dtype())) {
            return true;
        }
    }

    return false;
}

inline auto any_subtile_broadcasted_block_format(const Tensor& a, const auto& b) {
    if constexpr (requires { b.logical_shape(); }) {
        const auto& a_shape = a.logical_shape();
        const auto& b_shape = b.logical_shape();

        if (is_block_format(a.dtype()) &&
            ((a_shape[-2] == 1 && b_shape[-2] > 1) || (a_shape[-1] == 1 && b_shape[-1] > 1))) {
            return true;
        }

        if (is_block_format(b.dtype()) &&
            ((b_shape[-2] == 1 && a_shape[-2] > 1) || (b_shape[-1] == 1 && a_shape[-1] > 1))) {
            return true;
        }
    }

    return false;
}

inline auto is_binary_ng_only(const Tensor& a, const auto& b) {
    if constexpr (requires {
                      b.dtype();
                      b.is_sharded();
                      b.logical_shape();
                  }) {
        if (a.dtype() == DataType::INT32 or b.dtype() == DataType::INT32 or a.dtype() == DataType::UINT32 or
            b.dtype() == DataType::UINT32 or a.dtype() == DataType::UINT16 or b.dtype() == DataType::UINT16 or
            a.dtype() == DataType::UINT8 or b.dtype() == DataType::UINT8) {
            return true;
        }

        if (a.logical_shape().rank() > 4 or b.logical_shape().rank() > 4) {
            return true;
        }

        if (a.logical_shape()[-2] == 1 && b.logical_shape()[-2] > 1 && a.logical_shape()[-1] > 1 &&
            b.logical_shape()[-1] == 1) {
            return true;
        }
        if (b.logical_shape()[-2] == 1 && a.logical_shape()[-2] > 1 && b.logical_shape()[-1] > 1 &&
            a.logical_shape()[-1] == 1) {
            return true;
        }
    }
    return false;
}

}  // namespace detail

bool is_legacy_only(
    const Tensor& lhs,
    const auto& rhs,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations) {
    const auto& output_mem_cfg = memory_config.value_or(output ? output->memory_config() : MemoryConfig{});

    if (detail::any_sharded_block_format(lhs, rhs) or detail::any_subtile_broadcasted_block_format(lhs, rhs)) {
        TT_FATAL(
            lhs_activations.size() <= 1,
            "lhs_activations support maximum of 1 for legacy-only configuration; Override with use_legacy=False "
            "but note there may be issues");
        TT_FATAL(
            rhs_activations.empty(),
            "rhs_activations not supported for legacy-only configuration; Override with use_legacy=False but note "
            "there may be issues");
        return true;
    }

    return false;
}

template bool is_legacy_only<Tensor>(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations);

template bool is_legacy_only<float>(
    const Tensor& lhs,
    const float& rhs,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations);

template bool is_legacy_only<int32_t>(
    const Tensor& lhs,
    const int32_t& rhs,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations);

}  // namespace ttnn::operations::binary

namespace ttnn::detail {

inline auto invoke_binary_ng_impl(
    const Tensor& lhs,
    const auto& rhs,
    operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (use_legacy
            ? *use_legacy
            : operations::binary::is_legacy_only(lhs, rhs, memory_config, output, lhs_activations, rhs_activations) and
                  (not operations::binary::detail::is_binary_ng_only(lhs, rhs))) {
        const std::vector activations(post_activations.begin(), post_activations.end());
        const std::optional lhs_activation =
            lhs_activations.empty() ? std::nullopt : std::optional{lhs_activations.front()};

        if constexpr (requires { operations::binary::detail::preprocess_inputs(binary_op_type, lhs, rhs); }) {
            auto [a, b] = operations::binary::detail::preprocess_inputs(binary_op_type, lhs, rhs);

            return ttnn::prim::binary(a, b, binary_op_type, dtype, memory_config, output, activations, lhs_activation);
        } else {
            return ttnn::prim::binary(
                lhs, rhs, binary_op_type, dtype, memory_config, output, activations, lhs_activation);
        }
    }

    const auto a_dtype = lhs.dtype();
    const DataType b_dtype = [&] {
        if constexpr (requires { rhs.dtype(); }) {
            return rhs.dtype();
        } else {
            return a_dtype;
        }
    }();
    const auto output_preallocated = output.has_value();
    const auto is_integer_division = (binary_op_type == operations::binary::BinaryOpType::DIV) &&
                                     (a_dtype == DataType::INT32) && (b_dtype == DataType::INT32);
    if (is_integer_division) {
        // For integer division, output dtype should be float32
        if (dtype.has_value() || output_preallocated) {
            auto temp_dtype = output_preallocated ? output->dtype() : *dtype;
            TT_FATAL(temp_dtype == DataType::FLOAT32, "For integer division, supported output dtype is FLOAT32");
        }
    }
    const auto out_dtype = output_preallocated ? output->dtype() : dtype.value_or(a_dtype);

    const auto mem_config = output_preallocated ? output->memory_config() : memory_config.value_or(lhs.memory_config());

    if (dtype.has_value() && output_preallocated) {
        TT_FATAL(*dtype == out_dtype, "If both output dtype and output tensor are provided, their dtypes should match");
    }

    const auto typecast_a = operations::binary::detail::needs_typecast_to_bfloat16(binary_op_type, lhs, rhs);
    const auto typecast_b = operations::binary::detail::needs_typecast_to_bfloat16(binary_op_type, rhs, lhs);
    const auto typecast_out = operations::binary::detail::is_block_format(out_dtype);

    // RM is never BFLOAT8 or BFLOAT4 so we can assume it goes in here.
    if (not typecast_a and not typecast_b) {
        const auto input_a_rm = operations::binary::detail::is_layout_or_scalar(lhs, Layout::ROW_MAJOR);
        const auto input_b_rm = operations::binary::detail::is_layout_or_scalar(rhs, Layout::ROW_MAJOR);
        const auto input_a = operations::binary::detail::to_layout(lhs, Layout::TILE);
        const auto input_b = operations::binary::detail::to_layout(rhs, Layout::TILE);

        if (input_a_rm and input_b_rm) {
            // we don't support to_layout with optional output tensor
            TT_FATAL(
                !output_preallocated,
                "Optional output tensor with Row Major input is not supported right now for Elementwise "
                "operations");
        }

        auto result = ttnn::prim::binary_ng(
            input_a,
            input_b,
            binary_op_type,
            out_dtype,
            memory_config,
            output,
            fast_and_approximate_mode,
            lhs_activations,
            rhs_activations,
            post_activations,
            std::nullopt,
            sub_core_grids);

        // if both inputs are in row major, convert the output to row major
        // since there's no consensus here, avoiding the conversion if we have an excuse to is likely the best option
        // since it leads to better perf
        if (input_a_rm and input_b_rm) {
            return operations::binary::detail::to_layout(result, Layout::ROW_MAJOR);
        }

        return result;
    }
    const auto input_a = operations::binary::detail::to_dtype(lhs, DataType::BFLOAT16);
    const auto input_b = operations::binary::detail::to_dtype(rhs, DataType::BFLOAT16);
    const auto output_tensor =
        output_preallocated and typecast_out ? ttnn::typecast(*output, DataType::BFLOAT16) : output;

    Tensor result = ttnn::prim::binary_ng(
        input_a,
        input_b,
        binary_op_type,
        input_a.dtype(),
        memory_config,
        output_tensor,
        fast_and_approximate_mode,
        lhs_activations,
        rhs_activations,
        post_activations,
        std::nullopt,
        sub_core_grids);
    return typecast_out ? ttnn::typecast(result, out_dtype, mem_config, output) : result;
}

Tensor invoke_binary_ng(
    const Tensor& lhs,
    const Tensor& rhs,
    operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return invoke_binary_ng_impl(
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}

Tensor invoke_binary_ng(
    const Tensor& lhs,
    float rhs,
    operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return invoke_binary_ng_impl(
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}

Tensor invoke_binary_ng(
    const Tensor& lhs,
    int32_t rhs,
    operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return invoke_binary_ng_impl(
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}

}  // namespace ttnn::detail

namespace ttnn::operations::binary {

template <BinaryOpType binary_op_type>
Tensor relational_binary(
    const ttnn::Tensor& lhs,
    const float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (use_legacy
            ? *use_legacy
            : operations::binary::is_legacy_only(lhs, rhs, memory_config, output, lhs_activations, rhs_activations) and
                  (not operations::binary::detail::is_binary_ng_only(lhs, rhs))) {
        return detail::binary_impl(binary_op_type, lhs, rhs, dtype, memory_config, output);
    }

    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        /*fast_and_approximate_mode*/ false,
        sub_core_grids);
}

// scalar - tensor combination not available on Pytorch for this op
template <BinaryOpType binary_op_type>
Tensor relational_binary(
    const float lhs,
    const ttnn::Tensor& rhs,
    const std::optional<const DataType>& /*dtype*/,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return detail::binary_impl(binary_op_type, lhs, rhs, memory_config, output);
}

template <BinaryOpType binary_op_type>
Tensor inplace_relational_binary(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        binary_op_type,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        /*fast_and_approximate_mode*/ false,
        sub_core_grids);
}

template <BinaryOpType binary_op_type>
Tensor inplace_relational_binary(
    const ttnn::Tensor& lhs,
    const float rhs,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return relational_binary<binary_op_type>(
        lhs,
        rhs,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}

template <BinaryOpType binary_op_type>
Tensor inplace_mul_operation_with_fast_approx(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    bool is_block_fmt_inp = (is_block_float(lhs.dtype()) || is_block_float(rhs.dtype()));
    bool fast_and_approx = is_block_fmt_inp ? true : fast_and_approximate_mode.value_or(false);
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        binary_op_type,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approx,
        sub_core_grids);
}

template <BinaryOpType binary_op_type>
Tensor inplace_mul_operation_with_fast_approx(
    const ttnn::Tensor& lhs,
    const float rhs,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    bool is_block_fmt_inp = (is_block_float(lhs.dtype()));
    bool fast_and_approx = is_block_fmt_inp ? true : fast_and_approximate_mode.value_or(false);
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        binary_op_type,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approx,
        sub_core_grids);
}

template <BinaryOpType binary_op_type>
Tensor binary_operation_addalpha(
    const Tensor& lhs,
    const Tensor& rhs,
    float alpha,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    SmallVector<unary::EltwiseUnaryWithParam> rhs_activations{{unary::UnaryOpType::MUL_UNARY_SFPU, alpha}};
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::ADD,
        std::nullopt,
        memory_config,
        output,
        {},
        {},
        rhs_activations,
        /*use_legacy*/ false,
        /*fast_and_approximate_mode*/ false,
        std::nullopt);
}

template <BinaryOpType binary_op_type>
Tensor binary_operation_subalpha(
    const Tensor& lhs,
    const Tensor& rhs,
    float alpha,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    SmallVector<unary::EltwiseUnaryWithParam> rhs_activations{{unary::UnaryOpType::MUL_UNARY_SFPU, alpha}};
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::SUB,
        std::nullopt,
        memory_config,
        output,
        {},
        {},
        rhs_activations,
        /*use_legacy*/ false,
        /*fast_and_approximate_mode*/ false,
        std::nullopt);
}

template <BinaryOpType binary_op_type>
Tensor where_operation_with_scalar(
    const Tensor& condition,
    const Tensor& true_false_tensor,
    unary::ScalarVariant scalar_value,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    constexpr ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> none{};
    return ttnn::prim::binary_ng(
        condition,
        true_false_tensor,
        binary_op_type,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        false,            // fast_and_approximate_mode
        none,             // lhs_activations
        none,             // rhs_activations
        none,             // post_activations
        scalar_value,     // scalar
        sub_core_grids);  // sub_core_grids
}

template Tensor where_operation_with_scalar<BinaryOpType::WHERE_TST>(
    const Tensor&,
    const Tensor&,
    unary::ScalarVariant,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&);
template Tensor where_operation_with_scalar<BinaryOpType::WHERE_TTS>(
    const Tensor&,
    const Tensor&,
    unary::ScalarVariant,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&);

}  // namespace ttnn::operations::binary

namespace ttnn {

TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(add, ADD)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(add, ADD)
TTNN_BINARY_OP_INPLACE_IMPL(add_, ADD)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(subtract, SUB)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(subtract, SUB)
TTNN_BINARY_OP_INPLACE_IMPL(subtract_, SUB)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(eq, EQ)
Tensor eq(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::EQ>(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
Tensor eq(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::EQ>(
        lhs, rhs, dtype, memory_config, output);
}
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(ne, NE)
Tensor ne(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::NE>(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
Tensor ne(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::NE>(
        lhs, rhs, dtype, memory_config, output);
}
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(ge, GE)
Tensor ge(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::GE>(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
Tensor ge(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::GE>(
        lhs, rhs, dtype, memory_config, output);
}
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(gt, GT)
Tensor gt(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::GT>(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
Tensor gt(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::GT>(
        lhs, rhs, dtype, memory_config, output);
}
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(le, LE)
Tensor le(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::LE>(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
Tensor le(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::LE>(
        lhs, rhs, dtype, memory_config, output);
}
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(lt, LT)
Tensor lt(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::LT>(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
Tensor lt(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return operations::binary::relational_binary<operations::binary::BinaryOpType::LT>(
        lhs, rhs, dtype, memory_config, output);
}
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logical_and, LOGICAL_AND)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(logical_and, LOGICAL_AND)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logical_or, LOGICAL_OR)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(logical_or, LOGICAL_OR)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logical_xor, LOGICAL_XOR)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(logical_xor, LOGICAL_XOR)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(ldexp, LDEXP)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(ldexp, LDEXP)
TTNN_BINARY_OP_INPLACE_IMPL(ldexp_, LDEXP)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logaddexp, LOGADDEXP)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(logaddexp, LOGADDEXP)
TTNN_BINARY_OP_INPLACE_IMPL(logaddexp_, LOGADDEXP)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logaddexp2, LOGADDEXP2)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(logaddexp2, LOGADDEXP2)
TTNN_BINARY_OP_INPLACE_IMPL(logaddexp2_, LOGADDEXP2)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(squared_difference, SQUARED_DIFFERENCE)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(squared_difference, SQUARED_DIFFERENCE)
TTNN_BINARY_OP_INPLACE_IMPL(squared_difference_, SQUARED_DIFFERENCE)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logical_right_shift, LOGICAL_RIGHT_SHIFT)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(logical_right_shift, LOGICAL_RIGHT_SHIFT)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(xlogy, XLOGY)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(xlogy, XLOGY)

Tensor divide(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::DIV,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}
Tensor divide(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::DIV,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}
Tensor divide_(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::DIV,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}
Tensor divide_(
    const Tensor& lhs,
    float rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::DIV,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}
Tensor multiply(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    bool is_block_fmt_inp = (is_block_float(lhs.dtype()) || is_block_float(rhs.dtype()));
    bool fast_and_approx = is_block_fmt_inp ? true : fast_and_approximate_mode.value_or(false);
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::MUL,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approx,
        sub_core_grids);
}
Tensor multiply(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    bool is_block_fmt_inp = (is_block_float(lhs.dtype()));
    bool fast_and_approx = is_block_fmt_inp ? true : fast_and_approximate_mode.value_or(false);
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::MUL,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approx,
        sub_core_grids);
}
Tensor multiply(const Tensor& lhs, const Tensor& rhs, bool fast_and_approximate_mode) {
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::MUL,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        {},
        {},
        {},
        std::nullopt,
        fast_and_approximate_mode,
        std::nullopt);
}
Tensor multiply(const Tensor& lhs, float rhs, bool fast_and_approximate_mode) {
    return ttnn::detail::invoke_binary_ng(
        lhs,
        rhs,
        operations::binary::BinaryOpType::MUL,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        {},
        {},
        {},
        std::nullopt,
        fast_and_approximate_mode,
        std::nullopt);
}
Tensor multiply_(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::inplace_mul_operation_with_fast_approx<operations::binary::BinaryOpType::MUL>(
        lhs,
        rhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}
Tensor multiply_(
    const Tensor& lhs,
    float rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::inplace_mul_operation_with_fast_approx<operations::binary::BinaryOpType::MUL>(
        lhs,
        rhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}
TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL(gt_, GT)
TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL(ge_, GE)
TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL(le_, LE)
TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL(lt_, LT)
TTNN_BINARY_OP_INPLACE_INVOKE_IMPL(logical_and_, LOGICAL_AND)
TTNN_BINARY_OP_INPLACE_INVOKE_IMPL(logical_or_, LOGICAL_OR)
TTNN_BINARY_OP_INPLACE_INVOKE_IMPL(logical_xor_, LOGICAL_XOR)
TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL(eq_, EQ)
TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL(ne_, NE)
TTNN_BINARY_OP_INPLACE_INVOKE_IMPL(rsub_, RSUB)
TTNN_BINARY_OP_INPLACE_INVOKE_IMPL(bias_gelu_, BIAS_GELU)
#undef TTNN_BINARY_OP_TENSOR_TENSOR_IMPL
#undef TTNN_BINARY_OP_TENSOR_FLOAT_IMPL
#undef TTNN_BINARY_OP_INPLACE_IMPL
#undef TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL
#undef TTNN_BINARY_OP_INPLACE_INVOKE_IMPL
Tensor addalpha(
    const Tensor& lhs,
    const Tensor& rhs,
    float alpha,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return operations::binary::binary_operation_addalpha<operations::binary::BinaryOpType::ADDALPHA>(
        lhs, rhs, alpha, memory_config, output);
}
Tensor subalpha(
    const Tensor& lhs,
    const Tensor& rhs,
    float alpha,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return operations::binary::binary_operation_subalpha<operations::binary::BinaryOpType::SUBALPHA>(
        lhs, rhs, alpha, memory_config, output);
}
Tensor hypot(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::invoke_binary_ng(
        input_tensor_a,
        input_tensor_b,
        operations::binary::BinaryOpType::HYPOT,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        /*use_legacy*/ false,
        /*fast_and_approximate_mode*/ false,
        std::nullopt);
}

}  // namespace ttnn
