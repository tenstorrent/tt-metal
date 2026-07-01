// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Attribute / tensor-arg types for ttnn::prim::rebank_rm.
//
// rebank_rm converts a (B_total, N) ROW_MAJOR tensor whose page is one full
// row (page_size = N * elem_bytes) into a (B_total * N/chunk_size, chunk_size)
// tensor with page_size = chunk_size * elem_bytes.  This is a pure copy
// (no arithmetic, no transposition) that avoids the L1 overflow caused by
// on-device reshape when the source page exceeds the L1 capacity.
//
// Constraint: chunk_size must be a power-of-2 divisor of N, and
//             1 ≤ chunk_size ≤ N.

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Shared inline helper — defined here so neither rebank_rm_factory.cpp nor
// rebank_rm_device_operation.cpp redefines it in their anonymous namespaces
// (Unity build concatenates all TUs; duplicate symbols → redefinition error).
inline constexpr bool rebank_is_pow2(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

struct RebankRmParams {
    uint32_t chunk_size;  // target last-dim and output page size (in elements)
};

struct RebankRmTensorArgs {
    Tensor input;
};

}  // namespace ttnn::experimental::prim
