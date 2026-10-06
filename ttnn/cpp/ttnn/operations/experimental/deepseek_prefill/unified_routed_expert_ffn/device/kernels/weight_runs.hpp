// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

// Coalesced weight-tile DRAM reads, shared by the reader and the writer so the five weight read
// sites (reader gate/up/down, writer up/down) have ONE definition of the coalescing.
//
// SHARD_W > 0 — the weight tensor is DRAM ND-sharded, SHARD_W tile columns per shard. TensorAccessor
// places page (k, n) inside its shard at `(k % shard_k_rows) * SHARD_W + (n % SHARD_W)`
// (tensor_accessor.h get_bank_and_offset), so for a fixed k the pages of one shard row are
// physically contiguous in ONE bank, strided by the accessor's own page size (tensor_accessor.h
// get_noc_addr) -- which is `page_bytes` here, so `len * page_bytes` spans exactly `len` pages. A
// run therefore extends to the next SHARD_W boundary and leaves as a single NoC transaction. Shard
// HEIGHT never enters: the n-stride inside a shard is SHARD_W regardless of how many k-rows it
// holds.
//
// SHARD_W == 0 — DRAM-interleaved. Consecutive n land in different banks by construction, so no
// contiguous run exists: run() collapses to 1 and this degenerates to the per-tile read.
namespace unified_routed_expert_ffn {

template <uint32_t SHARD_W = 0>
struct WeightRuns {
    // Length of the maximal contiguous run starting at tile column j, clipped to end.
    static FORCE_INLINE uint32_t run(uint32_t j, uint32_t end) {
        if constexpr (SHARD_W > 0) {
            const uint32_t to_shard_edge = SHARD_W - (j % SHARD_W);
            const uint32_t remaining = end - j;
            return (to_shard_edge < remaining) ? to_shard_edge : remaining;
        } else {
            return 1;
        }
    }

    // Read tile columns [j0, jend) of tensor tile-row `row`, into L1 at
    // `l1_base + (j - j0) * page_bytes`. Columns are the free dim, so callers pass jend clipped to
    // the tensor's real N and the phantom columns past it stay UNWRITTEN — the guard the per-tile
    // loops used to spell out per column. Reads are ISSUED only; the caller owns the barrier.
    template <class Acc>
    static FORCE_INLINE void read(
        const Noc& noc,
        const Acc& acc,
        uint32_t row,
        uint32_t n_tiles_full,
        uint32_t j0,
        uint32_t jend,
        uint32_t l1_base,
        uint32_t page_bytes) {
        const uint32_t page_row_base = row * n_tiles_full;
        uint32_t j = j0;
        uint32_t off = 0;
        while (j < jend) {
            const uint32_t len = run(j, jend);
            // The accessor supplies the run's START page and the size covers the whole run, so
            // `len` pages leave as one transaction rather than one per page.
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(l1_base + off * page_bytes),
                len * page_bytes,
                {.page_id = page_row_base + j},
                {});
            j += len;
            off += len;
        }
    }
};

// ---------------------------------------------------------------------------------------------
// RING_WEIGHTS: the weights are the moe_compute decode (ring) tensors, read in place (one copy for decode and prefill).
//   w0_w1 (gate+up): tile page ((((c * E + e) * G + g) * R + k) * 4 + t), c = ring core = N-column of this op (NC = 8
//   columns,
//     9 N tiles each), g = group of 2 N tiles, k = K tile row (R = K padded to 7), t = [gate n0, up n0, gate n1, up
//     n1].
//   w2 (down): page ((((c * E + e) * GD + g) * RD + r) * 4 + t): c = hidden column (20 tiles each), g = group of 4
//   hidden tiles, r = stored
//     K row (RD = N padded to 7), t = the 4 hidden tiles. K rows are stored per 9-row chunk, rotated per core (see
//     moe_compute_utils.py prepare_w2_tensor_for_moe_compute): logical chunk ch of core c sits at stored chunk q = (c -
//     ch) mod NC.
// The accessor is the decode tensor's own (HEIGHT_SHARDED over the DRAM banks, tile pages), so a page id is all that is
// needed.
template <uint32_t E, uint32_t G, uint32_t R, uint32_t GD, uint32_t RD, uint32_t NC, uint32_t CH>
struct RingWeights {
    // gate (is_up = 0) or up (is_up = 1) tiles of K row `row`, N columns [0, ncols) of column c, into l1_base + nl *
    // page_bytes
    template <class Acc>
    static FORCE_INLINE void read_gu(
        const Noc& noc,
        const Acc& acc,
        uint32_t is_up,
        uint32_t e,
        uint32_t c,
        uint32_t row,
        uint32_t ncols,
        uint32_t l1_base,
        uint32_t page_bytes) {
        const uint32_t base = (c * E + e) * G;
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t page0 = ((base + g) * R + row) * 4 + is_up;
#pragma GCC unroll 2
            for (uint32_t i = 0; i < 2; ++i) {
                const uint32_t nl = 2 * g + i;
                if (nl < ncols) {
                    noc.async_read(
                        acc,
                        CoreLocalMem<uint32_t>(l1_base + nl * page_bytes),
                        page_bytes,
                        {.page_id = page0 + 2 * i},
                        {});
                }
            }
        }
    }

    // down tiles of logical K row `row`, hidden columns [20 c, 20 c + 4 GD) of column c, into l1_base + n * page_bytes
    template <class Acc>
    static FORCE_INLINE void read_d(
        const Noc& noc, const Acc& acc, uint32_t e, uint32_t c, uint32_t row, uint32_t l1_base, uint32_t page_bytes) {
        const uint32_t ch = row / CH;
        const uint32_t j = row - ch * CH;
        const uint32_t q = (c + NC - ch) % NC;
        const uint32_t r = q * CH + j;
        const uint32_t base = (c * E + e) * GD;
        for (uint32_t g = 0; g < GD; ++g) {
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(l1_base + 4 * g * page_bytes),
                4 * page_bytes,
                {.page_id = ((base + g) * RD + r) * 4},
                {});
        }
    }
};

}  // namespace unified_routed_expert_ffn
