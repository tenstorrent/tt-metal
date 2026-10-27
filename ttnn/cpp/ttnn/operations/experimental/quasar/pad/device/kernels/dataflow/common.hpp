// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Upper bound for the per-dim index/shape scratch arrays the tiled pad kernels keep in local memory.
// (ttnn tensors are at most 8-dimensional; pad preserves rank.)
constexpr uint32_t MAX_NUM_DIMS = 8;

// Takes in an id_per_dim and dims array, both of size ndims.
// Advances the id_per_dim by one, wrapping and carrying as needed.
// Templated on the pointer types so it works on both local mutable arrays (Metal 2.0 named/vararg RTAs
// are seeded into local scratch) and the legacy volatile-L1 dispatch-buffer arrays.
template <typename IdxPtr, typename DimsPtr>
static inline int advance_tensor_index(IdxPtr idx, DimsPtr dims, uint32_t ndims) {
    // increment least-significant dim first
    for (int32_t d = ndims - 1; d >= 0; d--) {
        uint32_t v = idx[d] + 1;
        if (v < dims[d]) {
            idx[d] = v;
            return 1;
        }
        idx[d] = 0;  // wrap and carry
    }
    return 0;  // overflowed most-significant dim
}
