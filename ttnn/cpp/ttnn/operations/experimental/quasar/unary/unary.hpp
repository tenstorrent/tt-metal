// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

namespace detail {

// Quasar copy of ttnn::operations::unary::detail::unary_impl: resolves the output dtype / memory config and
// the fp32 knobs, then launches the Metal 2.0 device op (ttnn::prim::qsr::unary). BFLOAT16 / FLOAT32 only.
Tensor unary_impl(
    const Tensor& input_tensor,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace detail

// ttnn.experimental.quasar.cos: element-wise cosine (SFPU), same signature as ttnn.cos minus sub_core_grids.
Tensor cos(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
