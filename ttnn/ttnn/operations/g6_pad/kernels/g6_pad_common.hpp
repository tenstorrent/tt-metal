// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace g6_pad {

constexpr uint32_t TILE = 32;
constexpr uint32_t FACE = 16;

// Fills bytes [offset, offset + n_bytes) of a one-page scratch CB with c. The CB is owned by the
// calling kernel alone and never pushed, so its write pointer is a fixed L1 address.
// The zero path is a self-aimed NoC read from the hardware zero region; its barrier is a read
// barrier on Blackhole, so no other NoC write may be issued between the two calls.
template <uint32_t cb_fill_page, uint32_t es, bool fill_is_zero>
FORCE_INLINE void build_fill_page(uint32_t offset, uint32_t n_bytes, uint32_t c_bits) {
    if constexpr (fill_is_zero) {
        Noc noc;
        CircularBuffer fill_page(cb_fill_page);
        noc.async_write_zeros(fill_page, n_bytes, {.offset_bytes = offset});
        noc.write_zeros_l1_barrier();
    } else {
        dataflow_kernel_lib::fill_l1_range<es>(get_write_ptr(cb_fill_page) + offset, n_bytes, c_bits);
    }
}

// Position of a staged page in input page order: lead indices (i0, i1), tile row, tile col.
struct StagedWalk {
    uint32_t i0, i1, ht, wt;

    StagedWalk(uint32_t page, uint32_t ht_in, uint32_t wt_in, uint32_t lead1) {
        const uint32_t tiles_per_slice = ht_in * wt_in;
        const uint32_t slice = page / tiles_per_slice;
        const uint32_t in_slice = page - slice * tiles_per_slice;
        i0 = slice / lead1;
        i1 = slice - i0 * lead1;
        ht = in_slice / wt_in;
        wt = in_slice - ht * wt_in;
    }

    FORCE_INLINE void advance(uint32_t ht_in, uint32_t wt_in, uint32_t lead1) {
        if (++wt < wt_in) {
            return;
        }
        wt = 0;
        if (++ht < ht_in) {
            return;
        }
        ht = 0;
        if (++i1 < lead1) {
            return;
        }
        i1 = 0;
        ++i0;
    }
};

}  // namespace g6_pad
