// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// L1 prefetch ride-along: a data-movement kernel of one program (the host op), once its own work is done, moves a
// block of tiles from a tensor (DRAM or L1) into an L1 region of a set of cores, for a LATER program (the consumer)
// to find there. The host op's program only ends once every write has landed, so the consumer, which starts after it,
// needs no semaphore.
//
// The region must be free while the host op runs (above its own CBs, below every live L1 buffer) and must be where the
// consumer will hold the data: typically the consumer's first operand window, at the address its CB will have. The
// host validates the first, the consumer checks the second (see matmul_reduce_scatter's weight prefetch).
//
// One job per host core, read from a per-device job table (a ROW_MAJOR uint32 tensor, one L1_PREFETCH_JOB_WORDS row
// per job), so programs stay identical across the mesh while each device moves its own block:
//   [0] first source page            [1] rows             [2] tiles per row     [3] source pages per row
//   [4] destination L1 address       [5] destination bytes per row            [6] bytes to multicast
//   [7..10] multicast rectangle (NoC-0 virtual x0, y0, x1, y1)                [11] receivers (the rectangle, but me)
//   [12] tile bytes
// The job reads rows x tiles into this core's own copy of the destination (row r, tile t at dst + r * row_bytes +
// t * tile_bytes; the gaps a short row leaves are multicast too), then multicasts the whole block to the rectangle,
// which contains this core (so no loopback).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace l1_prefetch {

constexpr uint32_t JOB_WORDS = 16;  // table row (64 B, one page)

// table / source: TensorAccessors. job: the job's row. scratch: 64 B of the (free) destination region, for the row.
template <typename Table, typename Source>
FORCE_INLINE void run_job(const Table& table, const Source& source, uint32_t job, uint32_t scratch) {
    noc_async_read(table.get_noc_addr(job, 0, noc_index), scratch, JOB_WORDS * sizeof(uint32_t), noc_index);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t page0 = w[0];
    const uint32_t rows = w[1];
    const uint32_t cols = w[2];
    const uint32_t row_pages = w[3];
    const uint32_t dst = w[4];
    const uint32_t row_bytes = w[5];
    const uint32_t bytes = w[6];
    uint32_t x0 = w[7], y0 = w[8], x1 = w[9], y1 = w[10];
    const uint32_t receivers = w[11];
    const uint32_t tile_bytes = w[12];

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t t = 0; t < cols; ++t) {
            noc_async_read(
                source.get_noc_addr(page0 + r * row_pages + t, 0, noc_index),
                dst + r * row_bytes + t * tile_bytes,
                tile_bytes,
                noc_index);
        }
    }
    noc_async_read_barrier();
    if (receivers > 0) {
        if (noc_index == 1) {  // NoC 1 multicasts from the far corner
            const uint32_t tx = x0, ty = y0;
            x0 = x1, y0 = y1, x1 = tx, y1 = ty;
        }
        noc_async_write_multicast(dst, get_noc_multicast_addr(x0, y0, x1, y1, dst, noc_index), bytes, receivers);
        noc_async_write_barrier();  // landed everywhere before this program can end
    }
}

}  // namespace l1_prefetch
