// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Cheap address generation for DRAM-INTERLEAVED buffers, for the grouped
// unified_routed_expert_ffn kernels.
//
// The generic TensorAccessor computes bank = page % num_banks and offset = page /
// num_banks for every page; with a non-power-of-two bank count (7 on P100) that is a
// software division per tile, and together with the NoC command issue it costs ~150
// RISC cycles per 576-byte bf4 tile read — which is what bounds a core's DRAM read
// rate to a few GB/s. Here the per-bank NoC base addresses are computed once per
// buffer and consecutive pages are walked with a rotating bank index (no division).
namespace fast_il {

template <uint32_t NB>
struct Bases {
    uint64_t b[NB];
    // Interleaved buffers start at the same in-bank address in every bank.
    FORCE_INLINE void init(uint32_t buffer_addr, uint8_t noc) {
        for (uint32_t i = 0; i < NB; ++i) {
            b[i] = get_noc_addr_from_bank_id<true>(i, buffer_addr, noc);
        }
    }
};

// Walks consecutive pages p0, p0+1, ... of an interleaved buffer.
template <uint32_t NB>
struct Cursor {
    uint32_t bank = 0;
    uint32_t off = 0;  // in pages
    FORCE_INLINE void start(uint32_t p0) {
        bank = p0 % NB;  // one division per run start (per tile-row), not per tile
        off = p0 / NB;
    }
    FORCE_INLINE uint64_t addr(const Bases<NB>& bs, uint32_t page_bytes) const {
        return bs.b[bank] + static_cast<uint64_t>(off) * page_bytes;
    }
    FORCE_INLINE void next() {
        if (++bank == NB) {
            bank = 0;
            ++off;
        }
    }
};

}  // namespace fast_il
