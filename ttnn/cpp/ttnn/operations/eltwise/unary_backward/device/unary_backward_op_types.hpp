// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::unary_backward {

// Gradients that run as a single fused device operation rather than as a composition of
// forward ops. One entry per op migrated off the composite path in unary_backward.cpp;
// each maps to a compute kernel through the table in unary_backward_op_utils.cpp.
//
// The enum is the program-cache discriminator: it is part of operation_attributes_t, so two
// op types can never share a cache entry even though they share this device operation.
enum class UnaryBackwardOpType : uint8_t {
    SIGMOID_BW,
};

}  // namespace ttnn::operations::unary_backward
