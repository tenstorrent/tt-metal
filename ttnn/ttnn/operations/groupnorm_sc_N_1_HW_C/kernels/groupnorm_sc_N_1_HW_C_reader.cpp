// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — reader (data movement).
//
// Per image owned by this core:
//   pass 1: for each column group cg: for each row chunk rc: load the chunk_rows x cols tile block of x
//           (TILE input: DRAM tiles -> cb_x_pass1 [+ cb_x_pass2 credit when resident];
//            RM input:   sticks -> cb_x_rm, compute tilizes), then generate E^T for cg -> cb_membership.
//   pass 2: for each column group cg: generate E for cg -> cb_membership and the gamma/beta row-0 tiles for cg in
//           one batch (one NoC zero-fill barrier, DRAM reads in flight while the RISC writes the E lanes, one read
//           barrier — Refinement 5), then (streaming regime only) re-stream the x chunks of cg into cb_x_pass2 /
//           cb_x_rm.
//
// Constant tiles: the reduce scaler (once per kernel; a [full, partial] REDUCE_COL pair when HW % 32 != 0, so
// the image's padded last tile-row never enters the statistics). Membership / affine-row pages are zeroed over
// the NoC (DM engine) before the few real lanes are written by the RISC.
//
// Padding independence (RM input): a stick block whose last tile-row is short (HW % 32 != 0) or whose last
// channel tile is short (C % 32 != 0) is zero-filled over the NoC before only the valid sticks / lanes are read
// into it, so the tilized pad rows / lanes are exact zeros (never stale L1, never the next image's rows).
//
// Ragged blocks (op_design.md -> Work Distribution): this core's `Ht_core x Ct_core` extents are RT args; the
// last row chunk / column group may be short. Every CB quantum stays nominal (`chunk`, `cols`, `cols*Kg`); the
// valid tiles are laid out densely (row-major valid_rows x valid_cols) and only they are read from DRAM.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "groupnorm_sc_N_1_HW_C_ragged.hpp"

// Per-stage zones (MaybeDeviceZoneScope, PERMANENT — free when the profiler is off; perf_instrumentation.hpp):
// CB reserves (back-pressure from compute), NoC issue loops (RISC-serial transaction cost) and NoC barriers
// (fabric / DRAM latency) are zoned separately — a barrier of ~0 only says the transfers had landed by then.
//
// Ablation switches (perf measurement only, host env GROUPNORM_SC_N_1_HW_C_KERNEL_DEFINES; output wrong by design):
//   GN_ABLATE_READ_X        skip the x tile / stick NoC reads (address math, barriers and CB traffic kept)
//   GN_ABLATE_MEMBERSHIP    skip the E zero-fill + lane writes (reserve / push kept)
//   GN_ABLATE_AFFINE_READS  skip the gamma / beta row reads + fix-ups (zero-fill, barrier, reserve / push kept)
#ifdef GN_ABLATE_READ_X
constexpr bool ablate_read_x = true;
#else
constexpr bool ablate_read_x = false;
#endif
#ifdef GN_ABLATE_MEMBERSHIP
constexpr bool ablate_membership = true;
#else
constexpr bool ablate_membership = false;
#endif
#ifdef GN_ABLATE_AFFINE_READS
constexpr bool ablate_affine_reads = true;
#else
constexpr bool ablate_affine_reads = false;
#endif

namespace {

constexpr uint32_t ONE_F32_BITS = 0x3F800000u;
constexpr uint16_t ONE_BF16_BITS = 0x3F80u;

// Byte offset of element (r, c) inside a 32x32 tile made of four 16x16 faces of `elem_bytes` elements.
constexpr uint32_t tile_elem_offset(uint32_t r, uint32_t c, uint32_t elem_bytes) {
    return (((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)) * elem_bytes;
}

}  // namespace

void kernel_main() {
    // ---------------- compile-time args ----------------
    constexpr uint32_t cb_x_pass1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_pass2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_membership = get_compile_time_arg_val(4);
    constexpr uint32_t cb_gamma_row = get_compile_time_arg_val(5);
    constexpr uint32_t cb_beta_row = get_compile_time_arg_val(6);
    constexpr bool is_rm = get_compile_time_arg_val(7) != 0;
    constexpr bool resident = get_compile_time_arg_val(8) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(9) != 0;
    constexpr bool has_beta = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(11);
    constexpr uint32_t cols = get_compile_time_arg_val(12);
    constexpr uint32_t Kg = get_compile_time_arg_val(13);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t x_elem_size = get_compile_time_arg_val(15);
    constexpr uint32_t x_page_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t g_elem_size = get_compile_time_arg_val(17);
    constexpr bool g_is_tile = get_compile_time_arg_val(18) != 0;
    constexpr bool g_is_bfp = get_compile_time_arg_val(19) != 0;  // bfloat8_b affine: whole-page reads
    [[maybe_unused]] constexpr uint32_t g_page_bytes = get_compile_time_arg_val(20);
    constexpr uint32_t C = get_compile_time_arg_val(21);
    constexpr uint32_t G = get_compile_time_arg_val(22);
    constexpr uint32_t HW = get_compile_time_arg_val(23);
    constexpr uint32_t Ht = get_compile_time_arg_val(24);
    constexpr uint32_t Ct = get_compile_time_arg_val(25);
    // Membership (E) element width: 4 = Float32 pages (fp32 DEST), 2 = Float16_b pages (16-bit DEST) — the host's
    // _statistic_page_dtype; the 0/1 entries are exact in both.
    constexpr uint32_t e_elem_size = get_compile_time_arg_val(26);
    static_assert(e_elem_size == 4 || e_elem_size == 2, "membership pages are Float32 or Float16_b");
    constexpr uint32_t TA_BASE = 27;

    constexpr auto x_args = TensorAccessorArgs<TA_BASE>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // ---------------- runtime args ----------------
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t image_begin = get_arg_val<uint32_t>(3);
    const uint32_t image_count = get_arg_val<uint32_t>(4);
    const uint32_t image_stride = get_arg_val<uint32_t>(5);
    const uint32_t row_begin = get_arg_val<uint32_t>(6);
    const uint32_t col_begin = get_arg_val<uint32_t>(7);
    const uint32_t active = get_arg_val<uint32_t>(8);
    const uint32_t Ht_core = get_arg_val<uint32_t>(9);
    const uint32_t Ct_core = get_arg_val<uint32_t>(10);

    if (active == 0) {
        return;  // idle core inside an image rectangle: no reads, no records
    }

    // Ragged accounting: blocks along each axis of this core's extent and the valid units of the last one.
    const auto row_axis = groupnorm_ragged::split(Ht_core, chunk_rows);
    const auto col_axis = groupnorm_ragged::split(Ct_core, cols);
    const uint32_t num_row_chunks = row_axis.count;
    const uint32_t num_col_groups = col_axis.count;

    constexpr uint32_t Cg = C / G;
    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t membership_tiles = cols * Kg;
    constexpr uint32_t e_tile_bytes = get_tile_size(cb_membership);

    Noc noc;
    CircularBuffer membership_cb(cb_membership);

    // ---------------- constants ----------------
    constexpr uint32_t hw_tail = HW % 32;  // valid rows of the image's last tile-row (0 = tile-aligned)
    constexpr uint32_t c_tail = C % 32;    // valid lanes of the last channel tile (0 = tile-aligned)
    {
        MaybeDeviceZoneScope("r_scaler");
        if constexpr (hw_tail != 0) {
            // [full, partial] pair: compute selects tile 1 for the last row tile of the chunk holding row Ht - 1
            dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<
                cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                hw_tail>();
        } else {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL>();
        }
    }
    const bool owns_last_row = (row_begin + Ht_core == Ht);  // this core's rows end at the image's last tile-row

    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);

    // E^T (pass 1, transposed = true) or E (pass 2) for column group cg: cols x Kg 0/1 tiles (Float32 or Float16_b
    // per e_elem_size). E_T[g', c] = 1 iff channel 32T + c belongs to group 32kg + g'. Tile order = the consuming
    // matmul_block's row-major K x N in1 block (index k * N + n, matmul_block_helpers.inl):
    //   pass 1  in1 = E^T, K = cols channel tiles, N = Kg   -> index tl * Kg + kg; tiles tl >= valid_cols stay zero
    //           (they are the K pad of the aggregation matmul);
    //   pass 2  in1 = E,   K = Kg, N = cols (Perf 1: ONE expansion matmul over the NOMINAL column group) -> index
    //           kg * cols + tl; the zero pad tiles tl >= valid_cols yield a_T = b_T = 0, never read by the apply.
    // (Identical orders when Kg == 1, i.e. every shape with num_groups <= 32.)
    // The RISC part: the 0/1 lanes into the (already zeroed, reserved) membership pages.
    auto write_membership_lanes = [&](uint32_t cg, bool transposed) {
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        const uint32_t base = get_write_ptr(cb_membership);
        for (uint32_t tl = 0; tl < valid_cols; ++tl) {
            const uint32_t T = col_begin + cg * cols + tl;
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t ch = T * 32 + c;
                if (ch >= C) {
                    break;  // padded lanes of a ragged last channel tile stay zero
                }
                const uint32_t g = ch / Cg;
                const uint32_t kg = g >> 5;
                const uint32_t gl = g & 31;
                const uint32_t r = transposed ? c : gl;
                const uint32_t cc = transposed ? gl : c;
                const uint32_t tile = transposed ? (tl * Kg + kg) : (kg * cols + tl);
                const uint32_t addr = base + tile * e_tile_bytes + tile_elem_offset(r, cc, e_elem_size);
                if constexpr (e_elem_size == 4) {
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = ONE_F32_BITS;
                } else {
                    *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr) = ONE_BF16_BITS;
                }
            }
        }
    };
    auto fill_membership = [&](uint32_t cg, bool transposed) {
        {
            MaybeDeviceZoneScope("r_memb_reserve");  // back-pressure: compute still holds the previous E block
            cb_reserve_back(cb_membership, membership_tiles);
        }
        MaybeDeviceZoneScope("r_memb_fill");  // NoC zero-fill + barrier + RISC lane writes
        if constexpr (!ablate_membership) {
            noc.async_write_zeros(membership_cb, membership_tiles * e_tile_bytes);
            noc.write_zeros_l1_barrier();
            write_membership_lanes(cg, transposed);
        }
        cb_push_back(cb_membership, membership_tiles);
    };

    // TILE input: the valid_rows x valid_cols tiles of image n, block (cg, rc), dense row-major from the block's
    // first page; the push stays the nominal `chunk`.
    auto load_x_chunk_tiles = [&](uint32_t n, uint32_t cg, uint32_t rc, bool pass2) {
        const uint32_t cb_id = pass2 ? cb_x_pass2 : cb_x_pass1;
        const uint32_t valid_rows = row_axis.valid(rc, chunk_rows);
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        {
            MaybeDeviceZoneScope("r_x_reserve");  // back-pressure from compute (streaming ring) / none (resident)
            cb_reserve_back(cb_id, chunk);
            if constexpr (resident) {
                cb_reserve_back(cb_x_pass2, chunk);  // aliased region: pass-2 credits move in lockstep
            }
        }
        uint32_t l1 = get_write_ptr(cb_id);
        {
            MaybeDeviceZoneScope("r_x_issue");  // RISC-serial: one TensorAccessor address + NoC command per tile
            for (uint32_t i = 0; i < valid_rows; ++i) {
                const uint32_t r = row_begin + rc * chunk_rows + i;
                for (uint32_t j = 0; j < valid_cols; ++j) {
                    const uint32_t t = col_begin + cg * cols + j;
                    const uint32_t page = n * Ht * Ct + r * Ct + t;
                    const uint64_t src = x_acc.get_noc_addr(page);
                    if constexpr (ablate_read_x) {
                        asm volatile("" ::"r"(static_cast<uint32_t>(src)), "r"(l1));  // keep the address math
                    } else {
                        noc_async_read(src, l1, x_tile_bytes);
                    }
                    l1 += x_tile_bytes;
                }
            }
        }
        {
            MaybeDeviceZoneScope("r_x_barrier");  // DRAM / fabric latency of the chunk's reads
            noc_async_read_barrier();
        }
        cb_push_back(cb_id, chunk);
        if constexpr (resident) {
            cb_push_back(cb_x_pass2, chunk);
        }
    };

    // RM input: the sticks of block (cg, rc) into cb_x_rm (valid_cols tile pages per tile-row). A ragged column
    // group keeps the tile-row quantum at `cols` pages: each tile-row's valid_cols data pages are followed by a
    // data-less pad push, and compute tilizes that group one tile-row at a time, popping the pad after each
    // (keeps the 2*cols ring aligned to tile-row starts).
    //
    // Padding independence: a tile-row is `lane_ragged` when this group holds the last channel tile of a
    // c_non_aligned tensor (only C - 32*T_first lanes exist in the stick) and `row_ragged` when it is the image's
    // last tile-row of an hw_non_aligned tensor (only hw_tail sticks exist; rows beyond belong to the next image or
    // lie past the buffer). Either way the block is zero-filled over the NoC first and only the valid sticks x
    // valid bytes are read over it, so the tilized pad rows / lanes are exact zeros. Pass 2 only feeds the apply,
    // whose pad rows / lanes are sliced off the output, so it skips the zero-fill but never reads past the image.
    CircularBuffer x_rm_cb(cb_x_rm);
    auto load_x_chunk_sticks = [&](uint32_t n, uint32_t cg, uint32_t rc, bool pass2) {
        if constexpr (is_rm) {
            MaybeDeviceZoneScope("r_x_sticks");  // RM: reserve + per-stick reads + barrier of one block
            const uint32_t valid_rows = row_axis.valid(rc, chunk_rows);
            const uint32_t valid_cols = col_axis.valid(cg, cols);
            const uint32_t T_first = col_begin + cg * cols;
            const uint32_t tile_row0 = row_begin + rc * chunk_rows;
            const uint32_t col_off = T_first * 32 * x_elem_size;
            const uint32_t lane_end = (T_first + valid_cols) * 32;
            const bool lane_ragged = (c_tail != 0) && (lane_end > C);
            const uint32_t row_bytes = ((lane_ragged ? C : lane_end) - T_first * 32) * x_elem_size;
            const bool row_ragged = (hw_tail != 0) && owns_last_row && (rc + 1 == num_row_chunks);
            if (valid_cols == cols && !lane_ragged && !row_ragged) {
                if constexpr (ablate_read_x) {
                    // sync only: the helper reserves / pushes valid_cols pages per tile-row
                    for (uint32_t i = 0; i < valid_rows; ++i) {
                        cb_reserve_back(cb_x_rm, valid_cols);
                        cb_push_back(cb_x_rm, valid_cols);
                    }
                } else {
                    dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm>(
                        x_acc, 32 * valid_rows, row_bytes, n * HW + 32 * tile_row0, col_off);
                }
                return;
            }
            for (uint32_t i = 0; i < valid_rows; ++i) {
                const bool last_tile_row = row_ragged && (i + 1 == valid_rows);
                const uint32_t sticks = last_tile_row ? hw_tail : 32u;
                if ((lane_ragged || last_tile_row) && !pass2) {
                    // reserve is idempotent: the helper below reserves the same pages, then reads over the zeros
                    cb_reserve_back(cb_x_rm, valid_cols);
                    noc.async_write_zeros(x_rm_cb, valid_cols * x_tile_bytes);
                    noc.write_zeros_l1_barrier();
                }
                dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm>(
                    x_acc, sticks, row_bytes, n * HW + 32 * (tile_row0 + i), col_off);
                groupnorm_ragged::pad_push(cb_x_rm, cols - valid_cols, cols - valid_cols);
            }
        }
    };

    auto load_x_chunk = [&](uint32_t n, uint32_t cg, uint32_t rc, bool pass2) {
        if constexpr (is_rm) {
            load_x_chunk_sticks(n, cg, rc, pass2);
        } else {
            load_x_chunk_tiles(n, cg, rc, pass2);
        }
    };

    // gamma / beta: row-0-valid tiles for the cols channel tiles of column group cg. Rows 1..31 zero.
    // The 32 values of channel tile T sit contiguously in the RM stick (64 B for bf16) but must land as
    // lanes 0..15 -> face 0 row 0 and lanes 16..31 -> face 1 row 0 (+face_bytes). A NoC read must keep the
    // source and destination residues modulo the DRAM alignment (64 B on Blackhole) equal, so the second
    // half cannot be read straight into face 1 (+32 B source vs +face_bytes destination). Read the whole
    // aligned run into face 0 rows 0..1, then move row 1 to face 1 row 0 with a few word stores and re-zero it.
    // Tiles tl >= valid_cols of a ragged last group stay zero and are never consumed (compute pad-pops them).
    //
    // Split into "issue the reads" and "fix up after the read barrier" so fill_pass2_constants can overlap the DRAM
    // latency of the gamma / beta reads with the RISC's membership-lane writes (Refinement 5: at the latency floor the
    // reader's pass-2 constants gated the affine build, 1.4 us of zero-fill / read barriers in series).
    constexpr uint32_t g_half_row_bytes = 16 * g_elem_size;
    constexpr uint32_t g_face_bytes = 256 * g_elem_size;
    auto issue_affine_reads = [&](uint32_t cb_row, const auto& acc, uint32_t cg) {
        const uint32_t g_tile_bytes = get_tile_size(cb_row);
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        uint32_t l1 = get_write_ptr(cb_row);
        for (uint32_t tl = 0; tl < valid_cols; ++tl) {
            const uint32_t T = col_begin + cg * cols + tl;
            if constexpr (g_is_tile && g_is_bfp) {
                // bfloat8_b TILE affine: a block format has no addressable row 0 (shared-exponent header +
                // packed mantissas), so fetch the whole tile page; rows 1..31 are the tensor's own zero padding.
                noc_async_read(acc.get_noc_addr(T, 0), l1, g_tile_bytes);
            } else if constexpr (g_is_tile) {
                // TILE affine: row 0 of faces 0 and 1 of page T are already at the right offsets.
                noc_async_read(acc.get_noc_addr(T, 0), l1, g_half_row_bytes);
                noc_async_read(acc.get_noc_addr(T, g_face_bytes), l1 + g_face_bytes, g_half_row_bytes);
            } else {
                // RM affine: one aligned read of all 32 values into face 0 rows 0..1 (fixed up below).
                noc_async_read(acc.get_noc_addr(0, T * 32 * g_elem_size), l1, 2 * g_half_row_bytes);
            }
            l1 += g_tile_bytes;
        }
    };
    // After the read barrier: RM row-1 -> face-1 move, and the c_non_aligned pad-lane zeroing.
    auto fixup_affine_rows = [&](uint32_t cb_row, uint32_t cg) {
        const uint32_t g_tile_bytes = get_tile_size(cb_row);
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        const uint32_t l1_base = get_write_ptr(cb_row);
        if constexpr (!g_is_tile) {
            uint32_t l1 = l1_base;
            for (uint32_t tl = 0; tl < valid_cols; ++tl) {
                volatile tt_l1_ptr uint32_t* row1 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + g_half_row_bytes);
                volatile tt_l1_ptr uint32_t* face1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + g_face_bytes);
                for (uint32_t w = 0; w < g_half_row_bytes / 4; ++w) {
                    face1[w] = row1[w];
                    row1[w] = 0u;
                }
                l1 += g_tile_bytes;
            }
        }
        // c_non_aligned: the last channel tile's lanes >= C carry whatever followed the stick in DRAM (RM
        // affine reads a full 32-lane run). They must be ZERO, not merely masked: pass 2 writes
        // y = 0 * a_T + b_T = beta into those output lanes, and a bfloat8_b output tile shares one exponent per
        // 16-lane block, so garbage there destroys the valid neighbouring channels (seen as PCC 0.96 on
        // (1,1,64,50) bf8b). TILE affine pages are zero-padded by the tensor itself; bf8b pages are read whole.
        if constexpr (!g_is_bfp) {
            if (c_tail != 0) {
                const uint32_t T_last = Ct - 1;
                const uint32_t T_first = col_begin + cg * cols;
                if (T_last >= T_first && T_last < T_first + valid_cols) {
                    const uint32_t tile_l1 = l1_base + (T_last - T_first) * g_tile_bytes;
                    for (uint32_t lane = c_tail; lane < 32; ++lane) {
                        const uint32_t off =
                            (lane < 16) ? lane * g_elem_size : g_face_bytes + (lane - 16) * g_elem_size;
                        volatile tt_l1_ptr uint8_t* q = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile_l1 + off);
                        for (uint32_t b = 0; b < g_elem_size; ++b) {
                            q[b] = 0u;
                        }
                    }
                }
            }
        }
    };

    // Pass-2 constants of column group cg in ONE batch: reserve E / gamma rows / beta rows, zero them all over the NoC
    // behind a single barrier, issue the gamma / beta DRAM reads, write the E lanes on the RISC while those reads are
    // in flight, then one read barrier + the fix-ups. (Was: three serial zero-fill barriers and two read barriers.)
    [[maybe_unused]] CircularBuffer gamma_row_cb(cb_gamma_row);
    [[maybe_unused]] CircularBuffer beta_row_cb(cb_beta_row);
    auto fill_pass2_constants = [&](uint32_t cg) {
        {
            MaybeDeviceZoneScope("r_p2_reserve");  // back-pressure: pass-1 E / previous group's rows still in use
            cb_reserve_back(cb_membership, membership_tiles);
            if constexpr (has_gamma) {
                cb_reserve_back(cb_gamma_row, cols);
            }
            if constexpr (has_beta) {
                cb_reserve_back(cb_beta_row, cols);
            }
        }
        MaybeDeviceZoneScope("r_p2_fill");  // zero-fill barrier + affine DRAM reads + E lanes + read barrier + fix-ups
        if constexpr (!ablate_membership) {
            noc.async_write_zeros(membership_cb, membership_tiles * e_tile_bytes);
        }
        if constexpr (has_gamma) {
            noc.async_write_zeros(gamma_row_cb, cols * get_tile_size(cb_gamma_row));
        }
        if constexpr (has_beta) {
            noc.async_write_zeros(beta_row_cb, cols * get_tile_size(cb_beta_row));
        }
        noc.write_zeros_l1_barrier();  // zeros landed before any read data lands on the same pages
        if constexpr (has_gamma && !ablate_affine_reads) {
            const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, g_page_bytes);
            issue_affine_reads(cb_gamma_row, gamma_acc, cg);
        }
        if constexpr (has_beta && !ablate_affine_reads) {
            const auto beta_acc = TensorAccessor(beta_args, beta_addr, g_page_bytes);
            issue_affine_reads(cb_beta_row, beta_acc, cg);
        }
        if constexpr (!ablate_membership) {
            write_membership_lanes(cg, /*transposed=*/false);
        }
        cb_push_back(cb_membership, membership_tiles);
        if constexpr (has_gamma || has_beta) {
            noc_async_read_barrier();
        }
        if constexpr (has_gamma) {
            if constexpr (!ablate_affine_reads) {
                fixup_affine_rows(cb_gamma_row, cg);
            }
            cb_push_back(cb_gamma_row, cols);
        }
        if constexpr (has_beta) {
            if constexpr (!ablate_affine_reads) {
                fixup_affine_rows(cb_beta_row, cg);
            }
            cb_push_back(cb_beta_row, cols);
        }
    };

    for (uint32_t img = 0; img < image_count; ++img) {
        const uint32_t n = image_begin + img * image_stride;

        // ---------------- pass 1: statistics ----------------
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                load_x_chunk(n, cg, rc, /*pass2=*/false);
            }
            fill_membership(cg, /*transposed=*/true);
        }

        // ---------------- pass 2: apply ----------------
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            fill_pass2_constants(cg);
            if constexpr (!resident) {
                for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                    load_x_chunk(n, cg, rc, /*pass2=*/true);
                }
            }
        }
    }
}
