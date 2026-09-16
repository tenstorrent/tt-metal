# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off (Perf tournament round 2) for the groupnorm_sc_N_1_HW_C READER's pass-1 schedule:
hide the RISC-side constant generation (reduce scaler + E^T membership block) in the DRAM-latency shadow of the
first x chunk's reads.

Program = the op's pass-1 reader geometry, nothing else: every core reads its own Ht_core x Ct_core tile block of a
DRAM-interleaved bf16 TILE tensor in the op's exact chunk order / page indexing (n*Ht*Ct + r*Ct + t), generates
cb_scaler and the E^T block per column group, and a BRISC "consumer" stub waits/pops exactly like the op's compute
(x chunk by chunk, scaler once, E^T after the last chunk of the column group, pass-2 credits at the end). In `check`
mode the consumer dumps every x tile, the scaler tile and the E^T tiles to DRAM (bit-exact gate); in `perf` mode it
writes nothing, so DEVICE KERNEL DURATION = the slowest core's "pass-1 reader done" time.

Variants (reader CT arg `variant`):
    0 baseline          the op's current schedule, verbatim: scaler helper (zero-fill + barrier + row-0 fill), then per
                        column group: chunks {reserve, issue, barrier, push}, then E^T {reserve, zero-fill, barrier,
                        lanes, push}.
    1 shadow            per column group: reserve chunk 0 -> reserve E^T (+scaler) -> ONE zero-fill of E^T (+scaler) +
                        zeros barrier -> ISSUE chunk 0's reads -> RISC fills the scaler row-0 and the E^T lanes while
                        the DRAM reads are in flight -> push scaler / E^T -> read barrier -> push chunk 0 -> remaining
                        chunks as today. (Blackhole's async_write_zeros is a NoC loopback READ and
                        write_zeros_l1_barrier is the FULL read barrier, so the zero-fill must be barriered BEFORE the
                        x reads are issued in this variant.)
    2 shadow_trid       as 1 but the zero-fill also runs inside the shadow: x reads tagged trid 2, zero loopback reads
                        tagged trid 1 (noc_async_read_set_trid), noc_async_read_barrier_with_trid(1) waits for the
                        zeros only.
    3 shadow_fastlanes  as 2 plus lane math without the per-lane `ch / Cg` divide (incremental in-group counter).
                        MEASURED REGRESSION (Cg is a compile-time constant, the divide was already a mul-shift; the
                        per-lane compare/branch costs more) — kept in the menu as the recorded null.
    4 shadow_tightlanes as 2 plus a run-based lane writer: one divide per column group, then per (tile, group) run a
                        stride-16-word pointer loop (rows c..c+run-1 of the same column gl). Same lanes.
    5 shadow_lookahead  as 2 plus one chunk of read lookahead: chunk rc+1's reads are issued (own trid) before chunk
                        rc's barrier (noc_async_read_barrier_with_trid). Resident regime only (the CB region holds the
                        whole per-core block, so chunk rc+1's L1 address is write_ptr + chunk bytes).
    6 shadow_all        2 + 4 + 5: trid-tagged zero-fill in the shadow, run-based lanes, one chunk of lookahead
                        (resident) — the graduation candidate.

Everything else is identical between variants: same CBs, same page order, same NoC assignment (the op's
DM_NOC_SPLIT alternate_y for >= 32 cores), same consumer. Pure dataflow: the data landing in L1 is bit-identical.
"""

import math
from dataclasses import dataclass

import ttnn

TILE = 32
X_DEPTH = 2  # op: streaming x ring depth, in chunks
MEMBERSHIP_DEPTH = 2  # op: membership (E) blocks buffered
NOC_SPLIT_MIN_CORES = 32  # op: DM_NOC_SPLIT_MIN_CORES (alternate_y)

CB_X_PASS1 = 0
CB_X_PASS2 = 1
CB_SCALER = 4
CB_MEMBERSHIP = 6

VARIANTS = (
    "baseline",
    "shadow",
    "shadow_trid",
    "shadow_fastlanes",
    "shadow_tightlanes",
    "shadow_lookahead",
    "shadow_all",
)
VARIANT_ID = {v: i for i, v in enumerate(VARIANTS)}


@dataclass(frozen=True)
class Case:
    """One pass-1 geometry. pr x pc cores cut the Ht x Ct tile grid of a (1, 1, HW, C) bf16 TILE input (ceil/floor
    balanced extents, exactly the op's _axis_range); cols / chunk_rows are the op's block knobs."""

    name: str
    HW: int
    C: int
    G: int
    pr: int
    pc: int
    cols: int
    chunk_rows: int
    resident: bool = True
    grid: tuple = (11, 10)

    @property
    def Ht(self):
        return self.HW // TILE

    @property
    def Ct(self):
        return self.C // TILE

    @property
    def Kg(self):
        return math.ceil(self.G / TILE)

    @property
    def p_used(self):
        return self.pr * self.pc

    @property
    def Ht_core_max(self):
        return math.ceil(self.Ht / self.pr)

    @property
    def Ct_core_max(self):
        return math.ceil(self.Ct / self.pc)

    @property
    def num_col_groups_max(self):
        return math.ceil(self.Ct_core_max / self.cols)

    @property
    def num_row_chunks_max(self):
        return math.ceil(self.Ht_core_max / self.chunk_rows)

    @property
    def chunk(self):
        return self.chunk_rows * self.cols

    @property
    def blk_max(self):
        return self.num_col_groups_max * self.num_row_chunks_max * self.chunk

    @property
    def membership_tiles(self):
        return self.cols * self.Kg

    @property
    def const_pages_per_core(self):
        return 1 + self.num_col_groups_max * self.membership_tiles


# Focus: (1,1,1024,640) G=32 on the 11x10 Blackhole grid -> pr=10, pc=11 (110 cores), Ht_core 3|4, Ct_core 1|2,
# cols=2, chunk_rows=2 (chunk=4 tiles, 2 row chunks), Kg=1, resident (blk_max = 8 tiles). Values reproduced from the
# op's _assign_images / _split_2d / _balanced_block for that shape.
FOCUS = Case("focus_1024x640_110c", HW=1024, C=640, G=32, pr=10, pc=11, cols=2, chunk_rows=2)
CASES = {
    FOCUS.name: FOCUS,
    # latency floor: one core, one tile, one chunk (G=1: every channel in one group)
    "floor_32x32_1c": Case("floor_32x32_1c", HW=32, C=32, G=1, pr=1, pc=1, cols=1, chunk_rows=1),
    # many-chunk single core: Ht_core=8, Ct_core=5, cols=5, chunk_rows=3 -> 3 chunks (3+3+2 rows) of 15 tiles
    "manychunk_256x160_1c": Case("manychunk_256x160_1c", HW=256, C=160, G=32, pr=1, pc=1, cols=5, chunk_rows=3),
    # same geometry in the streaming regime (X_DEPTH=2 chunk rings; the op's fallback when the block exceeds L1)
    "manychunk_256x160_1c_stream": Case(
        "manychunk_256x160_1c_stream", HW=256, C=160, G=32, pr=1, pc=1, cols=5, chunk_rows=3, resident=False
    ),
    # 4-core rectangle: (1,1,256,256), 2x2 cores of 4x4 tiles, cols=4, chunk_rows=2 -> 2 chunks of 8 tiles
    "rect4_256x256": Case("rect4_256x256", HW=256, C=256, G=32, pr=2, pc=2, cols=4, chunk_rows=2),
    # focus-like grid with two column groups per core (Ct_core 3|4, cols 2): E^T reserve/push of cg 1 while cg 0 in use
    "twocg_1024x1280_110c": Case("twocg_1024x1280_110c", HW=1024, C=1280, G=32, pr=10, pc=11, cols=2, chunk_rows=2),
    # three column groups on one core (Ct_core 24, cols 8): the E^T reserve of cg 2 must wait for the consumer to pop
    # cg 0 (MEMBERSHIP_DEPTH = 2) — exercises the reserve-before-chunk-0 ordering
    "threecg_64x768_1c": Case("threecg_64x768_1c", HW=64, C=768, G=32, pr=1, pc=1, cols=8, chunk_rows=2),
}


def _axis_range(extent, parts, i):
    begin = (i * extent) // parts
    end = ((i + 1) * extent) // parts
    return begin, end - begin


def _tight_rect(p_used, w, h):
    for rh in range(1, h + 1):
        if p_used % rh == 0 and p_used // rh <= w:
            return p_used // rh, rh
    best = None
    for rh in range(1, h + 1):
        rw = math.ceil(p_used / rh)
        if rw <= w and (best is None or rw * rh < best[0] * best[1]):
            best = (rw, rh)
    return best


@dataclass(frozen=True)
class CoreWork:
    x: int
    y: int
    index: int
    row_begin: int
    Ht_core: int
    col_begin: int
    Ct_core: int


def core_work(case: Case):
    Gx, Gy = case.grid
    rw, rh = _tight_rect(case.p_used, Gx, Gy)
    work = []
    for p in range(case.p_used):
        i, j = p // case.pc, p % case.pc
        row_begin, Ht_core = _axis_range(case.Ht, case.pr, i)
        col_begin, Ct_core = _axis_range(case.Ct, case.pc, j)
        work.append(CoreWork(p % rw, p // rw, p, row_begin, Ht_core, col_begin, Ct_core))
    return work


def _reader_noc_swapped(case: Case, x, y):
    """The op's DM_NOC_SPLIT 'alternate_y' (reader NoC1 / consumer NoC0 on odd rows) once >= 32 cores share an image."""
    return case.p_used >= NOC_SPLIT_MIN_CORES and (y % 2) == 1


# =============================================================================
# Kernels
# =============================================================================
_READER = r"""
// groupnorm_sc_N_1_HW_C perf_experiments/reader_pass1_shadow — pass-1 reader, five schedules (CT arg `variant`).
//
// Raw NoC / raw fill in the candidates (helpers bypassed, and why):
//   * dataflow_kernel_lib::calculate_and_prepare_reduce_scaler zero-fills its tile over the NoC and then calls
//     write_zeros_l1_barrier, which on Blackhole is the FULL noc_async_read_barrier: called while the x reads are
//     in flight it would wait for them and destroy the shadow. The candidates zero the scaler tile together with
//     the E^T pages (one batch) and write the same row-0 pattern (bf16 1.0 in row 0 of each face) by hand.
//   * noc_async_read_set_trid / noc_async_read_barrier_with_trid (variants >= 2): tag the zero loopback reads and
//     the x DRAM reads with different transaction ids so the zeros barrier waits for the zeros only.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

constexpr uint32_t ONE_F32_BITS = 0x3F800000u;
constexpr uint32_t ONE_BF16_PAIR = 0x3F803F80u;  // two bf16 1.0 (the scaler helper's Float16_b fill word)

constexpr uint32_t tile_elem_offset(uint32_t r, uint32_t c, uint32_t elem_bytes) {
    return (((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)) * elem_bytes;
}

struct Axis {
    uint32_t count;
    uint32_t last;
    FORCE_INLINE uint32_t valid(uint32_t i, uint32_t block) const { return (i + 1 == count) ? last : block; }
};

FORCE_INLINE Axis split(uint32_t extent, uint32_t block) {
    const uint32_t count = (extent + block - 1) / block;
    return Axis{count, extent - (count - 1) * block};
}

constexpr uint32_t TRID_ZERO = 1;  // zero-fill loopback reads
constexpr uint32_t TRID_X0 = 2;    // x chunk reads (even chunk index in the lookahead variant)
constexpr uint32_t TRID_X1 = 3;    // x chunk reads (odd chunk index in the lookahead variant)

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x_pass1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_pass2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_membership = get_compile_time_arg_val(3);
    constexpr bool resident = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t Kg = get_compile_time_arg_val(7);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t x_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t C = get_compile_time_arg_val(10);
    constexpr uint32_t G = get_compile_time_arg_val(11);
    constexpr uint32_t HW = get_compile_time_arg_val(12);
    constexpr uint32_t Ht = get_compile_time_arg_val(13);
    constexpr uint32_t Ct = get_compile_time_arg_val(14);
    constexpr uint32_t variant = get_compile_time_arg_val(15);
    constexpr uint32_t TA_BASE = 16;
    constexpr auto x_args = TensorAccessorArgs<TA_BASE>();

    static_assert(HW % 32 == 0, "bench: tile-aligned HW only (single full scaler tile)");
    constexpr uint32_t e_elem_size = 4;  // Float32 membership pages (fp32 DEST, the focus config)
    constexpr bool use_trid = variant >= 2;
    // 0 = the op's per-lane divide, 1 = incremental counter, 2 = run-based pointer loops
    constexpr uint32_t lane_mode = (variant == 3) ? 1 : (variant == 4 || variant == 6) ? 2 : 0;
    constexpr bool lookahead = (variant == 5 || variant == 6) && resident;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_begin = get_arg_val<uint32_t>(1);
    const uint32_t col_begin = get_arg_val<uint32_t>(2);
    const uint32_t active = get_arg_val<uint32_t>(3);
    const uint32_t Ht_core = get_arg_val<uint32_t>(4);
    const uint32_t Ct_core = get_arg_val<uint32_t>(5);
    if (active == 0) {
        return;
    }
    MaybeDeviceZoneScope("r_pass1");  // whole pass-1 reader span of this core

    const Axis row_axis = split(Ht_core, chunk_rows);
    const Axis col_axis = split(Ct_core, cols);
    const uint32_t num_row_chunks = row_axis.count;
    const uint32_t num_col_groups = col_axis.count;

    constexpr uint32_t Cg = C / G;
    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t membership_tiles = cols * Kg;
    constexpr uint32_t e_tile_bytes = get_tile_size(cb_membership);
    constexpr uint32_t scaler_tile_bytes = get_tile_size(cb_scaler);
    constexpr uint32_t n = 0;

    Noc noc;
    CircularBuffer membership_cb(cb_membership);
    CircularBuffer scaler_cb(cb_scaler);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);

    // ---- pieces shared by every schedule ----
    auto reserve_chunk = [&]() {
        MaybeDeviceZoneScope("r_x_reserve");
        cb_reserve_back(cb_x_pass1, chunk);
        if constexpr (resident) {
            cb_reserve_back(cb_x_pass2, chunk);
        }
    };
    auto push_chunk = [&]() {
        cb_push_back(cb_x_pass1, chunk);
        if constexpr (resident) {
            cb_push_back(cb_x_pass2, chunk);
        }
    };
    auto issue_chunk = [&](uint32_t cg, uint32_t rc, uint32_t l1) {
        MaybeDeviceZoneScope("r_x_issue");
        const uint32_t valid_rows = row_axis.valid(rc, chunk_rows);
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        for (uint32_t i = 0; i < valid_rows; ++i) {
            const uint32_t r = row_begin + rc * chunk_rows + i;
            for (uint32_t j = 0; j < valid_cols; ++j) {
                const uint32_t t = col_begin + cg * cols + j;
                const uint32_t page = n * Ht * Ct + r * Ct + t;
                noc_async_read(x_acc.get_noc_addr(page), l1, x_tile_bytes);
                l1 += x_tile_bytes;
            }
        }
    };
    // the op's write_membership_lanes (transposed = true), verbatim
    auto lanes_div = [&](uint32_t cg) {
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        const uint32_t base = get_write_ptr(cb_membership);
        for (uint32_t tl = 0; tl < valid_cols; ++tl) {
            const uint32_t T = col_begin + cg * cols + tl;
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t ch = T * 32 + c;
                if (ch >= C) {
                    break;
                }
                const uint32_t g = ch / Cg;
                const uint32_t kg = g >> 5;
                const uint32_t gl = g & 31;
                const uint32_t tile = tl * Kg + kg;
                const uint32_t addr = base + tile * e_tile_bytes + tile_elem_offset(c, gl, e_elem_size);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = ONE_F32_BITS;
            }
        }
    };
    // same lanes, no divide: one `ch / Cg` per column group, then an in-group counter walks the channels
    auto lanes_fast = [&](uint32_t cg) {
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        const uint32_t base = get_write_ptr(cb_membership);
        uint32_t ch = (col_begin + cg * cols) * 32;
        uint32_t g = ch / Cg;
        uint32_t k = ch - g * Cg;  // position of ch inside group g
        for (uint32_t tl = 0; tl < valid_cols; ++tl) {
            const uint32_t tile_base = base + (tl * Kg) * e_tile_bytes;
            for (uint32_t c = 0; c < 32; ++c, ++ch) {
                if (ch >= C) {
                    break;
                }
                const uint32_t kg = g >> 5;
                const uint32_t gl = g & 31;
                const uint32_t addr = tile_base + kg * e_tile_bytes + tile_elem_offset(c, gl, e_elem_size);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = ONE_F32_BITS;
                if (++k == Cg) {
                    k = 0;
                    ++g;
                }
            }
        }
    };
    // same lanes as runs: for each channel tile, walk the groups it intersects; a run of `run` channels of group g
    // writes rows c..c+run-1 of column gl of tile (tl*Kg + kg): a stride-16-word pointer loop per face row-half.
    auto lanes_tight = [&](uint32_t cg) {
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        const uint32_t base = get_write_ptr(cb_membership);
        uint32_t ch = (col_begin + cg * cols) * 32;
        uint32_t g = ch / Cg;
        uint32_t k = ch - g * Cg;
        for (uint32_t tl = 0; tl < valid_cols; ++tl) {
            const uint32_t tile_base = base + (tl * Kg) * e_tile_bytes;
            const uint32_t c_end = (ch + 32 <= C) ? 32u : (C - ch);
            uint32_t c = 0;
            while (c < c_end) {
                uint32_t run = Cg - k;
                if (run > c_end - c) {
                    run = c_end - c;
                }
                const uint32_t kg = g >> 5;
                const uint32_t gl = g & 31;
                volatile tt_l1_ptr uint32_t* p_col = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    tile_base + kg * e_tile_bytes + ((gl >> 4) * 256 + (gl & 15)) * e_elem_size);
                const uint32_t r1 = c + run;
                uint32_t r = c;
                for (; r < r1 && r < 16; ++r) {
                    p_col[r * 16] = ONE_F32_BITS;  // faces 0/1: row r
                }
                for (; r < r1; ++r) {
                    p_col[512 + (r - 16) * 16] = ONE_F32_BITS;  // faces 2/3: row r - 16
                }
                c = r1;
                ch += run;
                k += run;
                if (k == Cg) {
                    k = 0;
                    ++g;
                }
            }
        }
    };
    auto write_lanes = [&](uint32_t cg) {
        if constexpr (lane_mode == 1) {
            lanes_fast(cg);
        } else if constexpr (lane_mode == 2) {
            lanes_tight(cg);
        } else {
            lanes_div(cg);
        }
    };
    // the scaler helper's Float16_b SUM fill: row 0 of each of the 4 faces = bf16 1.0 (8 words of 2 bf16 per face)
    auto fill_scaler_row0 = [&]() {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t w = 0; w < 8; ++w) {
                p[face * 128 + w] = ONE_BF16_PAIR;
            }
        }
    };

    if constexpr (variant == 0) {
        // ======================= baseline: the op's schedule, verbatim =======================
        {
            MaybeDeviceZoneScope("r_scaler");
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL>();
        }
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                reserve_chunk();
                issue_chunk(cg, rc, get_write_ptr(cb_x_pass1));
                {
                    MaybeDeviceZoneScope("r_x_barrier");
                    noc_async_read_barrier();
                }
                push_chunk();
            }
            {
                MaybeDeviceZoneScope("r_memb_reserve");
                cb_reserve_back(cb_membership, membership_tiles);
            }
            {
                MaybeDeviceZoneScope("r_memb_fill");
                noc.async_write_zeros(membership_cb, membership_tiles * e_tile_bytes);
                noc.write_zeros_l1_barrier();
                lanes_div(cg);
            }
            cb_push_back(cb_membership, membership_tiles);
        }
    } else {
        // ======================= shadow schedules =======================
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            const bool first = (cg == 0);  // the scaler is generated once per kernel, inside the first shadow
            reserve_chunk();
            uint32_t l1 = get_write_ptr(cb_x_pass1);
            if constexpr (!use_trid) {
                // zero-fill + FULL read barrier first (nothing else in flight yet), then the reads
                {
                    MaybeDeviceZoneScope("r_memb_reserve");
                    cb_reserve_back(cb_membership, membership_tiles);
                    if (first) {
                        cb_reserve_back(cb_scaler, 1);
                    }
                }
                {
                    MaybeDeviceZoneScope("r_zero_fill");
                    noc.async_write_zeros(membership_cb, membership_tiles * e_tile_bytes);
                    if (first) {
                        noc.async_write_zeros(scaler_cb, scaler_tile_bytes);
                    }
                    noc.write_zeros_l1_barrier();
                }
                issue_chunk(cg, 0, l1);
            } else {
                // reads first (trid X0), then the zero-fill (trid ZERO) and a zeros-only barrier
                noc_async_read_set_trid(TRID_X0);
                issue_chunk(cg, 0, l1);
                {
                    MaybeDeviceZoneScope("r_memb_reserve");
                    cb_reserve_back(cb_membership, membership_tiles);
                    if (first) {
                        cb_reserve_back(cb_scaler, 1);
                    }
                }
                {
                    MaybeDeviceZoneScope("r_zero_fill");
                    noc_async_read_set_trid(TRID_ZERO);
                    noc.async_write_zeros(membership_cb, membership_tiles * e_tile_bytes);
                    if (first) {
                        noc.async_write_zeros(scaler_cb, scaler_tile_bytes);
                    }
                    noc_async_read_barrier_with_trid(TRID_ZERO);
                }
            }
            // RISC work in the shadow of chunk 0's DRAM reads
            if (first) {
                MaybeDeviceZoneScope("r_scaler");
                fill_scaler_row0();
                cb_push_back(cb_scaler, 1);
            }
            {
                MaybeDeviceZoneScope("r_memb_fill");
                write_lanes(cg);
            }
            cb_push_back(cb_membership, membership_tiles);

            if constexpr (!lookahead) {
                {
                    MaybeDeviceZoneScope("r_x_barrier");
                    noc_async_read_barrier();
                }
                push_chunk();
                for (uint32_t rc = 1; rc < num_row_chunks; ++rc) {
                    reserve_chunk();
                    l1 = get_write_ptr(cb_x_pass1);
                    if constexpr (use_trid) {
                        noc_async_read_set_trid(TRID_X0);
                    }
                    issue_chunk(cg, rc, l1);
                    {
                        MaybeDeviceZoneScope("r_x_barrier");
                        noc_async_read_barrier();
                    }
                    push_chunk();
                }
            } else {
                // one chunk of lookahead: chunk rc+1 in flight while chunk rc is barriered + pushed. Resident
                // region: the credits ring never wraps inside an image block, so chunk rc+1 lands at l1 + chunk.
                for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                    const uint32_t l1_next = l1 + chunk * x_tile_bytes;
                    if (rc + 1 < num_row_chunks) {
                        reserve_chunk();
                        noc_async_read_set_trid(((rc + 1) & 1) ? TRID_X1 : TRID_X0);
                        issue_chunk(cg, rc + 1, l1_next);
                    }
                    {
                        MaybeDeviceZoneScope("r_x_barrier");
                        noc_async_read_barrier_with_trid((rc & 1) ? TRID_X1 : TRID_X0);
                    }
                    push_chunk();
                    l1 = l1_next;
                }
            }
        }
        if constexpr (use_trid) {
            noc_async_read_barrier();  // hygiene: nothing outstanding on any trid at kernel end
            noc_async_read_set_trid(0);
        }
    }
}
"""

_CONSUMER = r"""
// groupnorm_sc_N_1_HW_C perf_experiments/reader_pass1_shadow — consumer stub (BRISC): waits / pops exactly like the
// op's compute in pass 1 (x chunk by chunk, scaler once, E^T after the last chunk of a column group, the resident
// pass-2 credits at the end). check == 1: dumps every x tile / the scaler tile / the E^T tiles to DRAM.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x_pass1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_pass2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_membership = get_compile_time_arg_val(3);
    constexpr bool resident = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t Kg = get_compile_time_arg_val(7);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t e_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t scaler_tile_bytes = get_compile_time_arg_val(10);
    constexpr bool check = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t const_pages_per_core = get_compile_time_arg_val(12);
    constexpr uint32_t Ht = get_compile_time_arg_val(13);
    constexpr uint32_t Ct = get_compile_time_arg_val(14);
    constexpr uint32_t TA_BASE = 15;
    constexpr auto out_x_args = TensorAccessorArgs<TA_BASE>();
    constexpr auto out_c_args = TensorAccessorArgs<out_x_args.next_compile_time_args_offset()>();

    const uint32_t out_x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_c_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_begin = get_arg_val<uint32_t>(2);
    const uint32_t col_begin = get_arg_val<uint32_t>(3);
    const uint32_t active = get_arg_val<uint32_t>(4);
    const uint32_t Ht_core = get_arg_val<uint32_t>(5);
    const uint32_t Ct_core = get_arg_val<uint32_t>(6);
    const uint32_t core_index = get_arg_val<uint32_t>(7);
    if (active == 0) {
        return;
    }

    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t membership_tiles = cols * Kg;
    const uint32_t num_row_chunks = (Ht_core + chunk_rows - 1) / chunk_rows;
    const uint32_t num_col_groups = (Ct_core + cols - 1) / cols;
    const uint32_t last_rows = Ht_core - (num_row_chunks - 1) * chunk_rows;
    const uint32_t last_cols = Ct_core - (num_col_groups - 1) * cols;

    const auto out_x = TensorAccessor(out_x_args, out_x_addr, x_tile_bytes);
    const auto out_c = TensorAccessor(out_c_args, out_c_addr, e_tile_bytes);
    const uint32_t const_page0 = core_index * const_pages_per_core;

    bool first = true;
    for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
        const uint32_t valid_cols = (cg + 1 == num_col_groups) ? last_cols : cols;
        for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
            const uint32_t valid_rows = (rc + 1 == num_row_chunks) ? last_rows : chunk_rows;
            cb_wait_front(cb_x_pass1, chunk);
            if (first) {
                cb_wait_front(cb_scaler, 1);
                if constexpr (check) {
                    noc_async_write(get_read_ptr(cb_scaler), out_c.get_noc_addr(const_page0), scaler_tile_bytes);
                }
                first = false;
            }
            if constexpr (check) {
                uint32_t l1 = get_read_ptr(cb_x_pass1);
                for (uint32_t i = 0; i < valid_rows; ++i) {
                    const uint32_t r = row_begin + rc * chunk_rows + i;
                    for (uint32_t j = 0; j < valid_cols; ++j) {
                        const uint32_t t = col_begin + cg * cols + j;
                        noc_async_write(l1, out_x.get_noc_addr(r * Ct + t), x_tile_bytes);
                        l1 += x_tile_bytes;
                    }
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_x_pass1, chunk);
        }
        cb_wait_front(cb_membership, membership_tiles);
        if constexpr (check) {
            uint32_t l1 = get_read_ptr(cb_membership);
            for (uint32_t k = 0; k < membership_tiles; ++k) {
                noc_async_write(l1, out_c.get_noc_addr(const_page0 + 1 + cg * membership_tiles + k), e_tile_bytes);
                l1 += e_tile_bytes;
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_membership, membership_tiles);
    }
    if constexpr (resident) {
        const uint32_t blk = num_col_groups * num_row_chunks * chunk;
        cb_wait_front(cb_x_pass2, blk);
        cb_pop_front(cb_x_pass2, blk);
    }
}
"""


# =============================================================================
# Program
# =============================================================================
def _ranges(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in cores])


def _cb(index, core_ranges, num_pages, page_size, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(x, out_x, out_const, case: Case, *, variant, check, zones):
    work = core_work(case)
    all_cores = _ranges([(w.x, w.y) for w in work])
    x_tile_bytes = ttnn.tile_size(ttnn.bfloat16)
    e_tile_bytes = ttnn.tile_size(ttnn.float32)
    scaler_tile_bytes = ttnn.tile_size(ttnn.bfloat16)

    cbs = [
        _cb(CB_SCALER, all_cores, 1, scaler_tile_bytes, ttnn.bfloat16),
        _cb(CB_MEMBERSHIP, all_cores, MEMBERSHIP_DEPTH * case.membership_tiles, e_tile_bytes, ttnn.float32),
    ]
    if case.resident:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=case.blk_max * x_tile_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_X_PASS1, data_format=ttnn.bfloat16, page_size=x_tile_bytes),
                    ttnn.CBFormatDescriptor(buffer_index=CB_X_PASS2, data_format=ttnn.bfloat16, page_size=x_tile_bytes),
                ],
            )
        )
    else:
        cbs.append(_cb(CB_X_PASS1, all_cores, X_DEPTH * case.chunk, x_tile_bytes, ttnn.bfloat16))
        cbs.append(_cb(CB_X_PASS2, all_cores, X_DEPTH * case.chunk, x_tile_bytes, ttnn.bfloat16))

    reader_ct = [
        CB_X_PASS1,
        CB_X_PASS2,
        CB_SCALER,
        CB_MEMBERSHIP,
        int(case.resident),
        case.chunk_rows,
        case.cols,
        case.Kg,
        x_tile_bytes,
        x.buffer_page_size(),
        case.C,
        case.G,
        case.HW,
        case.Ht,
        case.Ct,
        VARIANT_ID[variant],
    ]
    assert len(reader_ct) == 16
    reader_ct += list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    consumer_ct = [
        CB_X_PASS1,
        CB_X_PASS2,
        CB_SCALER,
        CB_MEMBERSHIP,
        int(case.resident),
        case.chunk_rows,
        case.cols,
        case.Kg,
        x_tile_bytes,
        e_tile_bytes,
        scaler_tile_bytes,
        int(check),
        case.const_pages_per_core,
        case.Ht,
        case.Ct,
    ]
    assert len(consumer_ct) == 15
    consumer_ct += list(ttnn.TensorAccessorArgs(out_x).get_compile_time_args())
    consumer_ct += list(ttnn.TensorAccessorArgs(out_const).get_compile_time_args())

    defines = [] if zones else [("KERNEL_LIB_PERF_ZONES_OFF", "1")]
    NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default
    kernels = []
    for swapped in (False, True):
        cores = [w for w in work if _reader_noc_swapped(case, w.x, w.y) == swapped]
        if not cores:
            continue
        rt_r, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for w in cores:
            rt_r[w.x][w.y] = [x.buffer_address(), w.row_begin, w.col_begin, 1, w.Ht_core, w.Ct_core]
            rt_c[w.x][w.y] = [
                out_x.buffer_address(),
                out_const.buffer_address(),
                w.row_begin,
                w.col_begin,
                1,
                w.Ht_core,
                w.Ct_core,
                w.index,
            ]
        ranges = _ranges([(w.x, w.y) for w in cores])
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_READER,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=reader_ct,
                defines=defines,
                runtime_args=rt_r,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=NOC1 if swapped else NOC0
                ),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_CONSUMER,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=consumer_ct,
                defines=defines,
                runtime_args=rt_c,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=NOC0 if swapped else NOC1
                ),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def dram_config():
    return ttnn.DRAM_MEMORY_CONFIG


def run_pass1(device, x, out_x, out_const, case: Case, *, variant, check, zones=False):
    """One launch of the pass-1 reader + consumer program. Returns nothing; results (check mode) are in out_x /
    out_const, which the caller allocated zero-filled (from_torch)."""
    desc = create_program_descriptor(x, out_x, out_const, case, variant=variant, check=check, zones=zones)
    ttnn.generic_op([x, out_x, out_const], desc)
