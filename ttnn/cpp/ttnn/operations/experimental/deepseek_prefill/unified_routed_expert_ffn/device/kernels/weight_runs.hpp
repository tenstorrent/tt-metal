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
template <uint32_t E, uint32_t G, uint32_t R, uint32_t GD, uint32_t RD, uint32_t NC, uint32_t CH, uint32_t G_ODD = 0>
struct RingWeights {
    // G = ceil(per-core gate/up columns / 2). PER_CORE_N_GU is not a template argument: it follows from G and HALF.
    static constexpr bool HALF = (G_ODD != 0);
    static constexpr uint32_t FULL_G = HALF ? G - 1 : G;
    // compact layout: odd column count, or fewer than 4 columns (moe_ring_common.h w0_w1_compact_for_shape)
    static constexpr bool COMPACT = HALF || (G * 2 < 4);
    static constexpr uint32_t HALF_ROWS = ((R / 7) * 7 == R ? ((R / 7 + 1) / 2) * 7 : 0);  // stored rows of the half column
    static constexpr uint32_t PAGES_PER_CE = COMPACT ? (FULL_G * R + (HALF ? HALF_ROWS : 0)) * 4 : G * R * 4;
    // gate (is_up = 0) or up (is_up = 1) tiles of K row `row`, N columns [0, ncols) of column c, into l1_base + nl *
    // page_bytes.
    // Layout of the w0_w1 tensor (moe_compute_utils.py, compact layout of the upstream op): one DRAM bank per ring core c,
    // per expert e a contiguous slice of PAGES_PER_CE pages: FULL_G block-columns (4 tiles wide, R K rows each, as before),
    // and, for an odd per-core column count, one HALF block-column (2 tiles wide: gate/up of one N column) stored
    // 14 K rows per block, two consecutive K rows per stored row: page = ((row / 2) * 4) + 2 * (row % 2) + is_up.
    // An even count of at least 4 keeps the old uniform layout (G groups of R rows).
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
        const uint32_t base = (c * E + e) * PAGES_PER_CE;
#pragma GCC unroll 2
        for (uint32_t g = 0; g < FULL_G; ++g) {
            const uint32_t page0 = base + (g * R + row) * 4 + is_up;
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
        if constexpr (HALF) {
            const uint32_t nl = 2 * FULL_G;
            if (nl < ncols) {
                noc.async_read(
                    acc,
                    CoreLocalMem<uint32_t>(l1_base + nl * page_bytes),
                    page_bytes,
                    {.page_id = base + FULL_G * R * 4 + (row >> 1) * 4 + 2 * (row & 1) + is_up},
                    {});
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

// ---------------------------------------------------------------------------------------------
// RING_WEIGHTS with a decode ring of NR = GPB * GRID_X cores (MOE_COMPUTE_RING=24 / 40: GPB ring cores per DRAM bank,
// grouped by bank). The op keeps its GRID_X = 8 N columns; column c = bank c = ring cores GPB*c .. GPB*c + GPB - 1,
// which own exactly the gate/up columns [COLS*c, COLS*c + COLS) and hidden tiles [HCOLS*c, HCOLS*c + HCOLS)
// (static_assert: the Bresenham distribution of moe_ring_common.h is periodic per bank). Layout (moe_compute_utils.py,
// compact gate/up layout):
//   w0_w1: bank c, expert e: the sub-cores' slices back to back, each = (n/2) full block-columns (R K rows x 4 tiles:
//   gate c, up c,
//     gate c+1, up c+1) then, for an odd count n, one half block-column (K rows two per stored row: page (row/2)*4 +
//     2*(row%2) + is_up).
//   w2: ring core cr, expert e: GD groups of 4 hidden tiles (the last group padded), RD stored rows; the K rows (N
//   tiles) are stored in the
//     rotated chunk order of the ring: stored chunk q of core cr = gate/up chunk (cr - q) mod NR, a chunk = the columns
//     of one ring core.
// Everything here is constexpr tables built from the same Bresenham functions as the host (moe_ring_common.h
// shard_tiles / w2_shard_tiles).
namespace ring_n {
constexpr uint32_t shard_tiles(uint32_t n_tiles, uint32_t id, uint32_t n_cores) {
    const uint32_t n_big = n_tiles % n_cores;
    const uint32_t small = n_tiles / n_cores;
    return small + ((n_big > 0 && (id * n_big) % n_cores < n_big) ? 1u : 0u);
}
constexpr uint32_t w2_shard_tiles(uint32_t Ht, uint32_t id, uint32_t Nt, uint32_t n_cores) {
    const uint32_t n_big_nt = Nt % n_cores;
    const uint32_t n_big_ht = Ht % n_cores;
    const uint32_t small_ht = Ht / n_cores;
    if (n_big_nt + n_big_ht == n_cores) {
        const bool big_nt = n_big_nt > 0 && (id * n_big_nt) % n_cores < n_big_nt;
        return big_nt ? small_ht : small_ht + 1u;
    }
    return shard_tiles(Ht, id, n_cores);
}
}  // namespace ring_n

// E experts, NR ring cores, NT intermediate tiles (72), HT hidden tiles (160), GRID_X banks (8), R = HT padded to 7
// (gate/up K rows), RD = NT padded to 7 (stored w2 K rows).
template <uint32_t E, uint32_t NR, uint32_t NT, uint32_t HT, uint32_t GRID_X, uint32_t R, uint32_t RD>
struct RingWeightsN {
    static constexpr uint32_t GPB = NR / GRID_X;
    static constexpr uint32_t COLS = NT / GRID_X;
    static constexpr uint32_t HCOLS = HT / GRID_X;
    static constexpr uint32_t F_PAGES = R * 4;
    static constexpr uint32_t H_PAGES = ((HT + 13) / 14) * 14 / 2 * 4;
    static constexpr uint32_t GD = (((HT + NR - 1) / NR) + 3) / 4;

    static_assert(
        NR % GRID_X == 0 && NT % GRID_X == 0 && HT % GRID_X == 0,
        "ring weights: NR, N and hidden must divide by the bank count");
    static constexpr bool periodic() {
        for (uint32_t b = 0; b < GRID_X; ++b) {
            for (uint32_t s = 0; s < GPB; ++s) {
                if (ring_n::shard_tiles(NT, GPB * b + s, NR) != ring_n::shard_tiles(NT, s, NR)) {
                    return false;
                }
                if (ring_n::w2_shard_tiles(HT, GPB * b + s, NT, NR) != ring_n::w2_shard_tiles(HT, s, NT, NR)) {
                    return false;
                }
            }
        }
        return true;
    }
    static_assert(periodic(), "ring weights: the shard distribution must repeat per DRAM bank");

    struct GuTab {
        uint32_t off[COLS];
        uint8_t half[COLS];
        uint8_t sub[COLS];
        uint32_t cols;
        uint32_t pages;
    };
    static constexpr GuTab make_gu() {
        GuTab t{};
        uint32_t j = 0, off = 0;
        for (uint32_t s = 0; s < GPB; ++s) {
            const uint32_t n = ring_n::shard_tiles(NT, s, NR);
            for (uint32_t p = 0; p < n / 2; ++p) {
                for (uint32_t i = 0; i < 2; ++i) {
                    if (j < COLS) {
                        t.off[j] = off;
                        t.half[j] = 0;
                        t.sub[j] = static_cast<uint8_t>(i);
                    }
                    ++j;
                }
                off += F_PAGES;
            }
            if (n % 2) {
                if (j < COLS) {
                    t.off[j] = off;
                    t.half[j] = 1;
                }
                ++j;
                off += H_PAGES;
            }
        }
        t.cols = j;
        t.pages = off;
        return t;
    }
    static constexpr GuTab GU = make_gu();
    static_assert(GU.cols == COLS, "ring weights: the sub-cores of a bank must own exactly COLS gate/up columns");
    static constexpr bool compact_gu() {
        const uint32_t max_cols = (NT + NR - 1) / NR;
        const uint32_t even = max_cols + (max_cols % 2);
        const uint32_t stride = even < 4 ? 4 : even;
        return max_cols < stride;
    }
    static_assert(compact_gu(), "ring weights: only the compact gate/up layout is supported");

    struct DTab {
        uint8_t tiles[GPB];
        uint8_t dst[GPB];
        uint32_t total;
    };
    static constexpr DTab make_d() {
        DTab t{};
        uint32_t d = 0;
        for (uint32_t s = 0; s < GPB; ++s) {
            const uint32_t n = ring_n::w2_shard_tiles(HT, s, NT, NR);
            t.tiles[s] = static_cast<uint8_t>(n);
            t.dst[s] = static_cast<uint8_t>(d);
            d += n;
        }
        t.total = d;
        return t;
    }
    static constexpr DTab DT = make_d();
    static_assert(DT.total == HCOLS, "ring weights: the sub-cores of a bank must own exactly HCOLS hidden tiles");

    struct ChTab {
        uint32_t cum[NR + 1];  // first N tile of gate/up chunk x (chunk x = columns of ring core x)
        uint8_t ch[NT];        // chunk of N tile n
        uint8_t off[NT];       // n - cum[ch[n]]
    };
    static constexpr ChTab make_ch() {
        ChTab t{};
        uint32_t a = 0;
        for (uint32_t x = 0; x < NR; ++x) {
            t.cum[x] = a;
            const uint32_t n = ring_n::shard_tiles(NT, x, NR);
            for (uint32_t i = 0; i < n; ++i) {
                t.ch[a + i] = static_cast<uint8_t>(x);
                t.off[a + i] = static_cast<uint8_t>(i);
            }
            a += n;
        }
        t.cum[NR] = a;
        return t;
    }
    static constexpr ChTab CH = make_ch();

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
        const uint32_t base = (c * E + e) * GU.pages;
#pragma GCC unroll 9
        for (uint32_t j = 0; j < COLS; ++j) {
            if (j < ncols) {
                const uint32_t page = GU.half[j] ? base + GU.off[j] + (row >> 1) * 4 + 2 * (row & 1) + is_up
                                                 : base + GU.off[j] + row * 4 + is_up + 2 * GU.sub[j];
                noc.async_read(
                    acc, CoreLocalMem<uint32_t>(l1_base + j * page_bytes), page_bytes, {.page_id = page}, {});
            }
        }
    }

    // down tiles of logical K row `row`, hidden columns [HCOLS c, HCOLS c + HCOLS) of bank column c, into l1_base + n *
    // page_bytes
    template <class Acc>
    static FORCE_INLINE void read_d(
        const Noc& noc, const Acc& acc, uint32_t e, uint32_t c, uint32_t row, uint32_t l1_base, uint32_t page_bytes) {
        const uint32_t ch = CH.ch[row];
        const uint32_t o = CH.off[row];
#pragma GCC unroll 5
        for (uint32_t s = 0; s < GPB; ++s) {
            const uint32_t cr = GPB * c + s;
            // stored rows before the chunk: the chunks cr, cr-1, ..., ch+1 (mod NR) of the rotated order
            const uint32_t before =
                (ch <= cr) ? CH.cum[cr + 1] - CH.cum[ch + 1] : NT - (CH.cum[ch + 1] - CH.cum[cr + 1]);
            const uint32_t r = before + o;
            const uint32_t base = (cr * E + e) * GD;
            for (uint32_t g = 0; g < GD; ++g) {
                const int32_t left = static_cast<int32_t>(DT.tiles[s]) - static_cast<int32_t>(4 * g);
                if (left > 0) {
                    const uint32_t cnt = left < 4 ? static_cast<uint32_t>(left) : 4u;
                    noc.async_read(
                        acc,
                        CoreLocalMem<uint32_t>(l1_base + (DT.dst[s] + 4 * g) * page_bytes),
                        cnt * page_bytes,
                        {.page_id = ((base + g) * RD + r) * 4},
                        {});
                }
            }
        }
    }
};

}  // namespace unified_routed_expert_ffn
