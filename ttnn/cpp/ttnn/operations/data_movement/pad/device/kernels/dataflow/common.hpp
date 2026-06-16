// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Takes in an id_per_dim and dims array, both of size ndims
// Advances the id_per_dim by one, wrapping and carrying as needed
//
// Metal 2.0 note: the per-dim arrays now arrive as runtime varargs and are copied into local
// stack arrays kernel-side (the vararg API has no writable pointer into the dispatch buffer),
// so the parameters are plain uint32_t* rather than the legacy volatile tt_l1_ptr uint32_t*.
// The wrap/carry arithmetic is unchanged.
static inline int advance_tensor_index(uint32_t* idx, uint32_t* dims, uint32_t ndims) {
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
