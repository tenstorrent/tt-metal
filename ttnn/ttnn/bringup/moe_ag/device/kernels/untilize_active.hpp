// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Active-row untilize: the tile rows of this chip's expert outputs that hold tokens (each local expert's region,
// ceil(count / 32) tile rows) x NCH column chunks, enumerated identically by the reader and the writer; core `me` of P
// takes every P-th block.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Loads counts / regions / the local-slot map (each [NG] uint32) into l1 and calls f(tile_row, chunk) for this core's
// blocks.
template <uint32_t NG, uint32_t EPC, uint32_t NCH, typename F>
inline void for_my_tile_rows(
    uint32_t counts_addr, uint32_t regions_addr, uint32_t lmap_addr, uint32_t l1, uint32_t me, uint32_t P, F f) {
    auto row = [](uint32_t addr) {
        return get_noc_addr(0, InterleavedAddrGen<true>{.bank_base_address = addr, .page_size = NG * 4});
    };
    noc_async_read(row(counts_addr), l1, NG * 4);
    noc_async_read(row(regions_addr), l1 + NG * 4, NG * 4);
    noc_async_read(row(lmap_addr), l1 + 2 * NG * 4, NG * 4);
    noc_async_read_barrier();
    const tt_l1_ptr uint32_t* counts = reinterpret_cast<const tt_l1_ptr uint32_t*>(l1);
    const uint32_t* regions = counts + NG;
    const uint32_t* lmap = counts + 2 * NG;
    uint32_t j = 0;
    for (uint32_t gid = 0; gid < NG; ++gid) {
        if (lmap[gid] >= EPC) {
            continue;
        }
        const uint32_t t0 = regions[gid] / 32, nt = (counts[gid] + 31) / 32;
        for (uint32_t t = 0; t < nt; ++t) {
            for (uint32_t c = 0; c < NCH; ++c, ++j) {
                if (j % P == me) {
                    f(t0 + t, c);
                }
            }
        }
    }
}
