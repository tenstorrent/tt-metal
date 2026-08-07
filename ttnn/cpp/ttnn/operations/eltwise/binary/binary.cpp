
// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-logger/tt-logger.hpp>

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
        const std::optional<CoreRangeSet>& sub_core_grids,                           \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {             \
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
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids,                                                          \
            sub_device_id);                                                          \
    }

#define TTNN_BINARY_OP_TENSOR_TENSOR_UINT8_IMPL(NAME, OP_TYPE)                                                \
    Tensor NAME(                                                                                              \
        const Tensor& lhs,                                                                                    \
        const Tensor& rhs,                                                                                    \
        const std::optional<const DataType>& output_dtype,                                                    \
        const std::optional<MemoryConfig>& memory_config,                                                     \
        const std::optional<Tensor>& output,                                                                  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,                          \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,                           \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,                           \
        const std::optional<CoreRangeSet>& sub_core_grids,                                                    \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {                                      \
        const std::optional<Tensor> lhs_cast =                                                                \
            lhs.dtype() == DataType::UINT8 ? ttnn::typecast(lhs, DataType::UINT16) : std::optional<Tensor>{}; \
        const Tensor& a = lhs_cast.has_value() ? *lhs_cast : lhs;                                             \
        const std::optional<Tensor> rhs_cast =                                                                \
            rhs.dtype() == DataType::UINT8 ? ttnn::typecast(rhs, DataType::UINT16) : std::optional<Tensor>{}; \
        const Tensor& b = rhs_cast.has_value() ? *rhs_cast : rhs;                                             \
        return ttnn::detail::invoke_binary_ng(                                                                \
            a,                                                                                                \
            b,                                                                                                \
            operations::binary::BinaryOpType::OP_TYPE,                                                        \
            output_dtype,                                                                                     \
            memory_config,                                                                                    \
            output,                                                                                           \
            post_activations,                                                                                 \
            lhs_activations,                                                                                  \
            rhs_activations,                                                                                  \
            /*fast_and_approximate_mode*/ false,                                                              \
            sub_core_grids,                                                                                   \
            sub_device_id);                                                                                   \
    }

#define TTNN_BINARY_OP_TENSOR_SCALAR_UINT8_IMPL(NAME, OP_TYPE)                                                \
    Tensor NAME(                                                                                              \
        const Tensor& lhs,                                                                                    \
        operations::unary::ScalarVariant rhs,                                                                 \
        const std::optional<const DataType>& output_dtype,                                                    \
        const std::optional<MemoryConfig>& memory_config,                                                     \
        const std::optional<Tensor>& output,                                                                  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,                          \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,                           \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,                           \
        const std::optional<CoreRangeSet>& sub_core_grids,                                                    \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {                                      \
        const std::optional<Tensor> lhs_cast =                                                                \
            lhs.dtype() == DataType::UINT8 ? ttnn::typecast(lhs, DataType::UINT16) : std::optional<Tensor>{}; \
        const Tensor& a = lhs_cast.has_value() ? *lhs_cast : lhs;                                             \
        return ttnn::detail::invoke_binary_ng(                                                                \
            a,                                                                                                \
            rhs,                                                                                              \
            operations::binary::BinaryOpType::OP_TYPE,                                                        \
            output_dtype,                                                                                     \
            memory_config,                                                                                    \
            output,                                                                                           \
            post_activations,                                                                                 \
            lhs_activations,                                                                                  \
            rhs_activations,                                                                                  \
            /*fast_and_approximate_mode*/ false,                                                              \
            sub_core_grids,                                                                                   \
            sub_device_id);                                                                                   \
    }

#define TTNN_BINARY_OP_FLOAT_TENSOR_UINT8_IMPL(NAME, OP_TYPE)                                                 \
    Tensor NAME(                                                                                              \
        float lhs,                                                                                            \
        const Tensor& rhs,                                                                                    \
        const std::optional<const DataType>& dtype,                                                           \
        const std::optional<MemoryConfig>& memory_config,                                                     \
        const std::optional<Tensor>& output) {                                                                \
        const std::optional<Tensor> rhs_cast =                                                                \
            rhs.dtype() == DataType::UINT8 ? ttnn::typecast(rhs, DataType::UINT16) : std::optional<Tensor>{}; \
        const Tensor& b = rhs_cast.has_value() ? *rhs_cast : rhs;                                             \
        return operations::binary::relational_binary<operations::binary::BinaryOpType::OP_TYPE>(              \
            lhs, b, dtype, memory_config, output);                                                            \
    }

#define TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(NAME, OP_TYPE)                             \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        operations::unary::ScalarVariant rhs,                                        \
        const std::optional<const DataType>& output_dtype,                           \
        const std::optional<MemoryConfig>& memory_config,                            \
        const std::optional<Tensor>& output,                                         \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        const std::optional<CoreRangeSet>& sub_core_grids,                           \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {             \
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
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids,                                                          \
            sub_device_id);                                                          \
    }

#define TTNN_BINARY_OP_INPLACE_IMPL(NAME, OP_TYPE)                                   \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        const Tensor& rhs,                                                           \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        const std::optional<CoreRangeSet>& sub_core_grids,                           \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {             \
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
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids,                                                          \
            sub_device_id);                                                          \
    }                                                                                \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        operations::unary::ScalarVariant rhs,                                        \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        const std::optional<CoreRangeSet>& sub_core_grids,                           \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {             \
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
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids,                                                          \
            sub_device_id);                                                          \
    }

#define TTNN_BINARY_OP_INPLACE_RELATIONAL_IMPL(NAME, OP_TYPE)                                             \
    Tensor NAME(                                                                                          \
        const Tensor& lhs,                                                                                \
        const Tensor& rhs,                                                                                \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,                      \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,                       \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,                       \
        const std::optional<CoreRangeSet>& sub_core_grids,                                                \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {                                  \
        return operations::binary::inplace_relational_binary<operations::binary::BinaryOpType::OP_TYPE>(  \
            lhs, rhs, post_activations, lhs_activations, rhs_activations, sub_core_grids, sub_device_id); \
    }                                                                                                     \
    Tensor NAME(                                                                                          \
        const Tensor& lhs,                                                                                \
        operations::unary::ScalarVariant rhs,                                                             \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,                      \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,                       \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,                       \
        const std::optional<CoreRangeSet>& sub_core_grids,                                                \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {                                  \
        return operations::binary::inplace_relational_binary<operations::binary::BinaryOpType::OP_TYPE>(  \
            lhs, rhs, post_activations, lhs_activations, rhs_activations, sub_core_grids, sub_device_id); \
    }

#define TTNN_BINARY_OP_INPLACE_INVOKE_IMPL(NAME, OP_TYPE)                            \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        const Tensor& rhs,                                                           \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        const std::optional<CoreRangeSet>& sub_core_grids,                           \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {             \
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
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids,                                                          \
            sub_device_id);                                                          \
    }                                                                                \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        operations::unary::ScalarVariant rhs,                                        \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        const std::optional<CoreRangeSet>& sub_core_grids,                           \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {             \
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
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids,                                                          \
            sub_device_id);                                                          \
    }

#define TTNN_BINARY_OP_TENSOR_TENSOR_BITWISE_IMPL(NAME, OP_TYPE)                     \
    Tensor NAME(                                                                     \
        const Tensor& lhs,                                                           \
        const Tensor& rhs,                                                           \
        const std::optional<MemoryConfig>& memory_config,                            \
        const std::optional<Tensor>& output,                                         \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,  \
        const std::optional<CoreRangeSet>& sub_core_grids,                           \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {             \
        return ttnn::detail::invoke_binary_ng(                                       \
            lhs,                                                                     \
            rhs,                                                                     \
            operations::binary::BinaryOpType::OP_TYPE,                               \
            std::nullopt,                                                            \
            memory_config,                                                           \
            output,                                                                  \
            post_activations,                                                        \
            lhs_activations,                                                         \
            rhs_activations,                                                         \
            /*fast_and_approximate_mode*/ false,                                     \
            sub_core_grids,                                                          \
            sub_device_id);                                                          \
    }

#define TTNN_BINARY_OP_TENSOR_INT32_BITWISE_IMPL(NAME, OP_TYPE)                                            \
    Tensor NAME(                                                                                           \
        const Tensor& lhs,                                                                                 \
        int32_t rhs,                                                                                       \
        const std::optional<MemoryConfig>& memory_config,                                                  \
        const std::optional<Tensor>& output,                                                               \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,                       \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,                        \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,                        \
        const std::optional<CoreRangeSet>& sub_core_grids,                                                 \
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {                                   \
        /* Fast path: unary SFPU handles scalar bitwise/shift directly with no activations. */             \
        if (!sub_device_id.has_value() && post_activations.empty() && lhs_activations.empty() &&           \
            rhs_activations.empty()) {                                                                     \
            return ttnn::unary_with_int32_param(                                                           \
                operations::unary::UnaryOpType::OP_TYPE, lhs, rhs, memory_config, output, sub_core_grids); \
        }                                                                                                  \
        /* Fallback: binary_ng tensor-scalar variant supports activations and sub_device_id. */            \
        return ttnn::detail::invoke_binary_ng(                                                             \
            lhs,                                                                                           \
            rhs,                                                                                           \
            operations::binary::BinaryOpType::OP_TYPE,                                                     \
            std::nullopt,                                                                                  \
            memory_config,                                                                                 \
            output,                                                                                        \
            post_activations,                                                                              \
            lhs_activations,                                                                               \
            rhs_activations,                                                                               \
            /*fast_and_approximate_mode*/ false,                                                           \
            sub_core_grids,                                                                                \
            sub_device_id);                                                                                \
    }

namespace ttnn::operations::binary::detail {

inline Tensor to_dtype(const Tensor& input, DataType dtype) {
    if (input.dtype() == dtype) {
        return input;
    }

    return ttnn::typecast(input, dtype);
}

inline float to_dtype(float input, [[maybe_unused]] DataType dtype) { return input; }
inline unary::ScalarVariant to_dtype(unary::ScalarVariant input, [[maybe_unused]] DataType dtype) { return input; }

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
inline bool is_layout_or_scalar([[maybe_unused]] unary::ScalarVariant input, [[maybe_unused]] Layout layout) {
    return true;
}

inline Tensor to_layout(const Tensor& input, Layout layout) {
    if (detail::is_layout_or_scalar(input, layout)) {
        return input;
    }

    return ttnn::to_layout(input, layout);
}

inline float to_layout(float input, [[maybe_unused]] Layout layout) { return input; }
inline unary::ScalarVariant to_layout(unary::ScalarVariant input, [[maybe_unused]] Layout layout) { return input; }

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

}  // namespace ttnn::operations::binary::detail

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
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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

    // Mixed float x 32-bit-integer arithmetic: binary_ng has no value conversion for a mismatched
    // integer operand, so its raw bits are reinterpreted as float when the integer CB is unpacked in
    // fp32 mode, producing inf / garbage (e.g. div(bf16, uint32) -> inf). Promote the integer operand
    // to the floating compute dtype, matching PyTorch type promotion and the existing UINT8->UINT16 and
    // integer-division handling. Scoped to DIV/MUL, the arithmetic ops where this corruption occurs.
    const auto is_32bit_int = [](DataType dt) { return dt == DataType::INT32 || dt == DataType::UINT32; };
    const auto float_promote_target = [](DataType float_dtype) {
        return float_dtype == DataType::FLOAT32 ? DataType::FLOAT32 : DataType::BFLOAT16;
    };
    std::optional<Tensor> lhs_promoted;
    std::optional<Tensor> rhs_promoted;
    if constexpr (requires { rhs.dtype(); }) {
        const bool is_float_arith = (binary_op_type == operations::binary::BinaryOpType::DIV) ||
                                    (binary_op_type == operations::binary::BinaryOpType::MUL);
        if (is_float_arith) {
            if (is_32bit_int(a_dtype) && tt::tt_metal::is_floating_point(b_dtype)) {
                const auto target = float_promote_target(b_dtype);
                log_debug(
                    tt::LogOp,
                    "Binary: typecasting lhs from integer dtype {} to {} to match floating rhs dtype {}",
                    a_dtype,
                    target,
                    b_dtype);
                lhs_promoted = ttnn::typecast(lhs, target);
            } else if (is_32bit_int(b_dtype) && tt::tt_metal::is_floating_point(a_dtype)) {
                const auto target = float_promote_target(a_dtype);
                log_debug(
                    tt::LogOp,
                    "Binary: typecasting rhs from integer dtype {} to {} to match floating lhs dtype {}",
                    b_dtype,
                    target,
                    a_dtype);
                rhs_promoted = ttnn::typecast(rhs, target);
            }
        }
    }
    const Tensor& lhs_eff = lhs_promoted.has_value() ? *lhs_promoted : lhs;
    decltype(auto) rhs_eff = [&]() -> decltype(auto) {
        if constexpr (requires { rhs.dtype(); }) {
            return (rhs_promoted.has_value() ? *rhs_promoted : rhs);
        } else {
            return (rhs);
        }
    }();

    // When an integer operand is promoted and no explicit output dtype is requested, the result should
    // follow the promoted (floating) type rather than the original integer dtype of lhs.
    const auto out_dtype = output_preallocated ? output->dtype() : dtype.value_or(lhs_eff.dtype());

    if (dtype.has_value() && output_preallocated) {
        TT_FATAL(*dtype == out_dtype, "If both output dtype and output tensor are provided, their dtypes should match");
    }

    // RM is never BFLOAT8 or BFLOAT4 so we can assume it goes in here.

    const auto input_a_rm = operations::binary::detail::is_layout_or_scalar(lhs_eff, Layout::ROW_MAJOR);
    const auto input_b_rm = operations::binary::detail::is_layout_or_scalar(rhs_eff, Layout::ROW_MAJOR);
    const auto input_a_sharded = lhs_eff.memory_config().is_sharded();
    const auto input_b_sharded = [&]() {
        if constexpr (requires { rhs_eff.memory_config(); }) {
            return rhs_eff.memory_config().is_sharded();
        } else {
            return false;
        }
    }();
    // we don't support to_layout with optional output tensor
    TT_FATAL(
        !(output_preallocated && input_a_rm && input_b_rm),
        "Optional output tensor with Row Major input is not supported right now for Elementwise operations");
    if (input_a_rm and input_b_rm and not input_a_sharded and not input_b_sharded) {
        auto result = ttnn::prim::binary_ng(
            lhs_eff,
            rhs_eff,
            binary_op_type,
            out_dtype,
            memory_config,
            output,
            fast_and_approximate_mode,
            lhs_activations,
            rhs_activations,
            post_activations,
            std::nullopt,
            sub_core_grids,
            sub_device_id);

        return result;
    }
    // Either one or both are tiles
    const auto input_a = operations::binary::detail::to_layout(lhs_eff, Layout::TILE);
    const auto input_b = operations::binary::detail::to_layout(rhs_eff, Layout::TILE);

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
        sub_core_grids,
        sub_device_id);

    // if both inputs are in row major, convert the output to row major
    // since there's no consensus here, avoiding the conversion if we have an excuse to is likely the best option
    // since it leads to better perf
    if (input_a_rm and input_b_rm) {
        return operations::binary::detail::to_layout(result, Layout::ROW_MAJOR);
    }

    return result;
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
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approximate_mode,
        sub_core_grids,
        sub_device_id);
}

Tensor invoke_binary_ng(
    const Tensor& lhs,
    operations::unary::ScalarVariant rhs,
    operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approximate_mode,
        sub_core_grids,
        sub_device_id);
}

Tensor invoke_binary_ng_isclose(
    const Tensor& lhs,
    const Tensor& rhs,
    float rtol,
    float atol,
    bool equal_nan,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    const auto fa = lhs.dtype() == DataType::INT32 ? ttnn::typecast(lhs, DataType::FLOAT32) : lhs;
    const auto fb = rhs.dtype() == DataType::INT32 ? ttnn::typecast(rhs, DataType::FLOAT32) : rhs;

    const auto input_a_rm = fa.layout() == Layout::ROW_MAJOR;
    const auto input_b_rm = fb.layout() == Layout::ROW_MAJOR;
    const auto input_a_sharded = fa.memory_config().is_sharded();
    const auto input_b_sharded = fb.memory_config().is_sharded();

    using BinaryOpType = ttnn::operations::binary_ng::BinaryOpType;

    if (input_a_rm && input_b_rm && !input_a_sharded && !input_b_sharded) {
        return ttnn::prim::binary_ng(
            fa,
            fb,
            BinaryOpType::ISCLOSE,
            std::nullopt,
            memory_config,
            output,
            std::nullopt,
            {},
            {},
            {},
            std::nullopt,
            sub_core_grids,
            std::nullopt,
            rtol,
            atol,
            equal_nan);
    }

    const auto input_a = operations::binary::detail::to_layout(fa, Layout::TILE);
    const auto input_b = operations::binary::detail::to_layout(fb, Layout::TILE);
    auto result = ttnn::prim::binary_ng(
        input_a,
        input_b,
        BinaryOpType::ISCLOSE,
        std::nullopt,
        memory_config,
        output,
        std::nullopt,
        {},
        {},
        {},
        std::nullopt,
        sub_core_grids,
        std::nullopt,
        rtol,
        atol,
        equal_nan);

    // if both inputs are in row major, convert the output to row major
    // since there's no consensus here, avoiding the conversion if we have an excuse to is likely the best option
    // since it leads to better perf
    if (input_a_rm && input_b_rm) {
        return operations::binary::detail::to_layout(result, Layout::ROW_MAJOR);
    }
    return result;
}

}  // namespace ttnn::detail

namespace ttnn::operations::binary {

template <BinaryOpType binary_op_type>
Tensor relational_binary(
    const ttnn::Tensor& lhs,
    unary::ScalarVariant rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        /*fast_and_approximate_mode*/ false,
        sub_core_grids,
        sub_device_id);
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
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        /*fast_and_approximate_mode*/ false,
        sub_core_grids,
        sub_device_id);
}

template <BinaryOpType binary_op_type>
Tensor inplace_relational_binary(
    const ttnn::Tensor& lhs,
    unary::ScalarVariant rhs,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    return relational_binary<binary_op_type>(
        lhs,
        rhs,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        sub_core_grids,
        sub_device_id);
}

template <BinaryOpType binary_op_type>
Tensor inplace_mul_operation_with_fast_approx(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approx,
        sub_core_grids,
        sub_device_id);
}

template <BinaryOpType binary_op_type>
Tensor inplace_mul_operation_with_fast_approx(
    const ttnn::Tensor& lhs,
    unary::ScalarVariant rhs,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approx,
        sub_core_grids,
        sub_device_id);
}

template <BinaryOpType binary_op_type>
Tensor binary_operation_addalpha(
    const Tensor& lhs,
    const Tensor& rhs,
    float alpha,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    ttsl::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations{{unary::UnaryOpType::MUL_UNARY_SFPU, alpha}};
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
    ttsl::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations{{unary::UnaryOpType::MUL_UNARY_SFPU, alpha}};
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
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    constexpr ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> none{};
    return ttnn::prim::binary_ng(
        condition,
        true_false_tensor,
        binary_op_type,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        false,         // fast_and_approximate_mode
        none,          // lhs_activations
        none,          // rhs_activations
        none,          // post_activations
        scalar_value,  // scalar
        sub_core_grids,
        sub_device_id);
}

template Tensor where_operation_with_scalar<BinaryOpType::WHERE_TST>(
    const Tensor&,
    const Tensor&,
    unary::ScalarVariant,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&,
    const std::optional<tt::tt_metal::SubDeviceId>&);
template Tensor where_operation_with_scalar<BinaryOpType::WHERE_TTS>(
    const Tensor&,
    const Tensor&,
    unary::ScalarVariant,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&,
    const std::optional<tt::tt_metal::SubDeviceId>&);

}  // namespace ttnn::operations::binary

namespace ttnn {

TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(add, ADD)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(add, ADD)
TTNN_BINARY_OP_INPLACE_IMPL(add_, ADD)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(subtract, SUB)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(subtract, SUB)
TTNN_BINARY_OP_INPLACE_IMPL(subtract_, SUB)
TTNN_BINARY_OP_TENSOR_TENSOR_UINT8_IMPL(eq, EQ)
TTNN_BINARY_OP_TENSOR_SCALAR_UINT8_IMPL(eq, EQ)
TTNN_BINARY_OP_FLOAT_TENSOR_UINT8_IMPL(eq, EQ)
TTNN_BINARY_OP_TENSOR_TENSOR_UINT8_IMPL(ne, NE)
TTNN_BINARY_OP_TENSOR_SCALAR_UINT8_IMPL(ne, NE)
TTNN_BINARY_OP_FLOAT_TENSOR_UINT8_IMPL(ne, NE)
TTNN_BINARY_OP_TENSOR_TENSOR_UINT8_IMPL(ge, GE)
TTNN_BINARY_OP_TENSOR_SCALAR_UINT8_IMPL(ge, GE)
TTNN_BINARY_OP_FLOAT_TENSOR_UINT8_IMPL(ge, GE)
TTNN_BINARY_OP_TENSOR_TENSOR_UINT8_IMPL(gt, GT)
TTNN_BINARY_OP_TENSOR_SCALAR_UINT8_IMPL(gt, GT)
TTNN_BINARY_OP_FLOAT_TENSOR_UINT8_IMPL(gt, GT)
TTNN_BINARY_OP_TENSOR_TENSOR_UINT8_IMPL(le, LE)
TTNN_BINARY_OP_TENSOR_SCALAR_UINT8_IMPL(le, LE)
TTNN_BINARY_OP_FLOAT_TENSOR_UINT8_IMPL(le, LE)
TTNN_BINARY_OP_TENSOR_TENSOR_UINT8_IMPL(lt, LT)
TTNN_BINARY_OP_TENSOR_SCALAR_UINT8_IMPL(lt, LT)
TTNN_BINARY_OP_FLOAT_TENSOR_UINT8_IMPL(lt, LT)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logical_and, LOGICAL_AND)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(logical_and, LOGICAL_AND)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logical_or, LOGICAL_OR)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(logical_or, LOGICAL_OR)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logical_xor, LOGICAL_XOR)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(logical_xor, LOGICAL_XOR)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(ldexp, LDEXP)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(ldexp, LDEXP)
TTNN_BINARY_OP_INPLACE_IMPL(ldexp_, LDEXP)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logaddexp, LOGADDEXP)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(logaddexp, LOGADDEXP)
TTNN_BINARY_OP_INPLACE_IMPL(logaddexp_, LOGADDEXP)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logaddexp2, LOGADDEXP2)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(logaddexp2, LOGADDEXP2)
TTNN_BINARY_OP_INPLACE_IMPL(logaddexp2_, LOGADDEXP2)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(squared_difference, SQUARED_DIFFERENCE)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(squared_difference, SQUARED_DIFFERENCE)
TTNN_BINARY_OP_INPLACE_IMPL(squared_difference_, SQUARED_DIFFERENCE)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(logical_right_shift, LOGICAL_RIGHT_SHIFT)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(logical_right_shift, LOGICAL_RIGHT_SHIFT)
TTNN_BINARY_OP_TENSOR_TENSOR_BITWISE_IMPL(bitwise_and, BITWISE_AND)
TTNN_BINARY_OP_TENSOR_INT32_BITWISE_IMPL(bitwise_and, BITWISE_AND)
TTNN_BINARY_OP_TENSOR_TENSOR_BITWISE_IMPL(bitwise_or, BITWISE_OR)
TTNN_BINARY_OP_TENSOR_INT32_BITWISE_IMPL(bitwise_or, BITWISE_OR)
TTNN_BINARY_OP_TENSOR_TENSOR_BITWISE_IMPL(bitwise_xor, BITWISE_XOR)
TTNN_BINARY_OP_TENSOR_INT32_BITWISE_IMPL(bitwise_xor, BITWISE_XOR)
TTNN_BINARY_OP_TENSOR_TENSOR_BITWISE_IMPL(bitwise_left_shift, LEFT_SHIFT)
TTNN_BINARY_OP_TENSOR_INT32_BITWISE_IMPL(bitwise_left_shift, LEFT_SHIFT)
TTNN_BINARY_OP_TENSOR_TENSOR_BITWISE_IMPL(bitwise_right_shift, RIGHT_SHIFT)
TTNN_BINARY_OP_TENSOR_INT32_BITWISE_IMPL(bitwise_right_shift, RIGHT_SHIFT)
TTNN_BINARY_OP_TENSOR_TENSOR_BITWISE_IMPL(logical_left_shift, LEFT_SHIFT)
TTNN_BINARY_OP_TENSOR_INT32_BITWISE_IMPL(logical_left_shift, LEFT_SHIFT)
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(xlogy, XLOGY)
TTNN_BINARY_OP_TENSOR_SCALAR_IMPL(xlogy, XLOGY)

Tensor divide(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approximate_mode,
        sub_core_grids,
        sub_device_id);
}
Tensor divide(
    const Tensor& lhs,
    operations::unary::ScalarVariant rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approximate_mode,
        sub_core_grids,
        sub_device_id);
}
Tensor divide_(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approximate_mode,
        sub_core_grids,
        sub_device_id);
}
Tensor divide_(
    const Tensor& lhs,
    operations::unary::ScalarVariant rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approximate_mode,
        sub_core_grids,
        sub_device_id);
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
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approx,
        sub_core_grids,
        sub_device_id);
}
Tensor multiply(
    const Tensor& lhs,
    operations::unary::ScalarVariant rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        fast_and_approx,
        sub_core_grids,
        sub_device_id);
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
        fast_and_approximate_mode,
        std::nullopt);
}
Tensor multiply(const Tensor& lhs, operations::unary::ScalarVariant rhs, bool fast_and_approximate_mode) {
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
        fast_and_approximate_mode,
        std::nullopt);
}
Tensor multiply_(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    return operations::binary::inplace_mul_operation_with_fast_approx<operations::binary::BinaryOpType::MUL>(
        lhs,
        rhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        fast_and_approximate_mode,
        sub_core_grids,
        sub_device_id);
}
Tensor multiply_(
    const Tensor& lhs,
    operations::unary::ScalarVariant rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    return operations::binary::inplace_mul_operation_with_fast_approx<operations::binary::BinaryOpType::MUL>(
        lhs,
        rhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        fast_and_approximate_mode,
        sub_core_grids,
        sub_device_id);
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
#undef TTNN_BINARY_OP_FLOAT_TENSOR_UINT8_IMPL
#undef TTNN_BINARY_OP_TENSOR_SCALAR_UINT8_IMPL
#undef TTNN_BINARY_OP_TENSOR_TENSOR_UINT8_IMPL
#undef TTNN_BINARY_OP_TENSOR_SCALAR_IMPL
#undef TTNN_BINARY_OP_TENSOR_TENSOR_BITWISE_IMPL
#undef TTNN_BINARY_OP_TENSOR_INT32_BITWISE_IMPL
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
        /*fast_and_approximate_mode*/ false,
        std::nullopt);
}

Tensor isclose(
    const Tensor& input_a,
    const Tensor& input_b,
    float rtol,
    float atol,
    bool equal_nan,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::detail::invoke_binary_ng_isclose(
        input_a, input_b, rtol, atol, equal_nan, output_mem_config, std::nullopt);
}

}  // namespace ttnn
