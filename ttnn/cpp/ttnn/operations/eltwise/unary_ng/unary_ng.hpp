// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::unary_ng {

// Ops registered here use the unary_ng device operation (single compute kernel, layout flexibility, etc.).
// Add and register op by op as needed.

struct Abs {
    static Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(const ComplexTensor& input_tensor, const MemoryConfig& output_mem_config);
};

}  // namespace ttnn::operations::unary_ng

namespace ttnn {

// Registered unary_ng ops (one by one). Single API: ttnn::abs uses NG implementation.
inline constexpr auto abs = ttnn::register_operation<"ttnn::abs", ttnn::operations::unary_ng::Abs>();

}  // namespace ttnn
