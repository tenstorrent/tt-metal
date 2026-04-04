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
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(mish, MISH)

REGISTER_UNARY_OPERATION(selu, SELU)

// Unaries with float parameter

// -----------------------------------------------------------------------------
// Functions defined without macros (non-SFPU operations kept)
// -----------------------------------------------------------------------------

Tensor identity(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor logit(
    const Tensor& input_tensor,
    std::optional<float> eps = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor unary_chain(
    const Tensor& input_tensor,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& ops_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& input_tensor,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
