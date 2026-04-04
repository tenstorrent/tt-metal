// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::unary_ng::detail {

Tensor unary_ng_impl(
    const Tensor& input_tensor,
    const std::vector<unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::operations::unary_ng::detail

namespace ttnn {

#define DECLARE_UNARY_NG_OP(op_name)                                                   \
    Tensor op_name(                                                                    \
        const Tensor& input_tensor,                                                    \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt, \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,            \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

DECLARE_UNARY_NG_OP(cosh)

#undef DECLARE_UNARY_NG_OP

Tensor abs(const ComplexTensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn
