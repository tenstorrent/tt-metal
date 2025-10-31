// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::ternary {

// Type alias for scalar or tensor values
// Supports float, int, and Tensor types
using TensorScalarVariant = std::variant<float, int, Tensor>;

// Ternary operation types
enum class TernaryOpType {
    WHERE,  // conditional selection: out = predicate ? value_true : value_false
    LERP,   // linear interpolation: out = input + weight * (end - input)
};

// Variant types for ternary operations
enum class TernaryVariant {
    TTT,  // tensor-tensor-tensor
    TTS,  // tensor-tensor-scalar
    TST,  // tensor-scalar-tensor
    TSS,  // tensor-scalar-scalar
};

// Broadcast types for ternary operations
enum class TernaryBroadcastType {
    NONE,
    OUTER_BCAST,     // bcast for outer dims -5, -4, -3, no subtile bcast.
    COL_BCAST,       // bcast for W-dim and outer dims -5, -4, -3.
    ROW_BCAST,       // Row broadcast for H-dim
    SCALAR_BCAST,    // Scalar broadcast for TTT: Either A or B or C can be (1,1)
    SCALAR_A_BCAST,  // A = (1,1) B = (H,W )
    SCALAR_B_BCAST,  // A = (H,W) B = (1,1)
    INVALID_BCAST,   // All other unsupported bcast cases go here for now
};

}  // namespace ttnn::operations::ternary
