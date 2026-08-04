// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for rms_norm (NCRISC, NoC0).
//
// Per core, per row-block, per width chunk it stages:
//   * TILE build      : whole x tiles         -> cb_input_tiles
//   * ROW_MAJOR build : padded x sticks       -> cb_input_sticks (compute tilizes)
// plus, once at boot, the reduce scaler; plus gamma (once per core in the
// RESIDENT regime, once per width chunk in STREAM).
//
// Loop nest mirrors the compute kernel exactly (op_design.md section 7):
//     for blk in blocks: for pass in {A} or {A,B}: for c in chunks
// Pass B is re-read ONLY in the STREAM regime (X_RESIDENT == false); in the
// RESIDENT regime cb_input_tiles is held across both passes, so x is read once.
//
// Helper-usage notes
// ------------------
// * scaler CB          -> dataflow_kernel_lib::prepare_[partial_]reduce_scalers
//                         (ReduceTile datapath) or prepare_reduce_mask
//                         (AccumulateViaAdd datapath), pool-type-aware
//                         overloads (PoolType::SUM, ReduceDim::REDUCE_ROW).
// * ROW_MAJOR staging  -> dataflow_kernel_lib::read_sticks_for_tilize at TILE
//                         granularity, which is exactly the contract of
//                         compute_kernel_lib::tilize<WT_CHUNK>(rows).
// * TILE staging + gamma reads are TensorAccessor + noc_async_read_tile: the
//   dataflow tilize helper covers neither whole-tile interleaved reads nor the
//   gamma slot (op_design.md section 6.1).
//
// Refinement 2b -- the BAND scheme (ROW_MAJOR shard cutting the WIDTH axis).
// Such a shard's page is a row SEGMENT, so no accessor read can reach a row.
// Instead each core stages the band it already holds out of its OWN L1, and
// joins the unchanged cross-core combine: sum(x^2) over a row is the sum over the
// bands however the bands are cut, so a band need not align to a tile column.
// See _plan_band in rms_norm_program_descriptor.py.
//
// One raw-API addition beyond the design's table: a ONE-TIME zero of the whole
// cb_input_sticks ring at boot, via noc.async_write_zeros (the device zero
// API), gated on STAGE_ZERO.  Reason (R3): the L1 pad lanes of a staged
// ROW_MAJOR row are never written by a stick read, so whatever L1 garbage was
// there survives into the reduce.  The partial scaler multiplies pad lanes by
// zero, and inf*0 / nan*0 = NaN would poison the whole row.  Zeroing the ring
// once establishes the invariant "every pad byte is either zero or real tensor
// data" (later reads only ever overwrite with tensor values), so no per-block
// zeroing is needed.  H-tail rows need no zeroing: a padding row's reduction
// and output are confined to that row and the writer never writes it.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_gamma_sticks = 5;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(4);
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(6);
    constexpr uint32_t GAMMA_IS_RM = get_compile_time_arg_val(7);
    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(8);
    constexpr uint32_t GAMMA_ELEM_BYTES = get_compile_time_arg_val(9);
    // Total ROW_MAJOR sticks in the tensor.  The per-core stick range now comes in
    // as a runtime extent (stick_base / stick_count), which is what the BAND scheme
    // needs (its rows do not start on a tile boundary); R_RM is kept as the
    // whole-tensor figure the CT-arg contract documents.
    [[maybe_unused]] constexpr uint32_t R_RM = get_compile_time_arg_val(10);
    constexpr uint32_t W_ELEMS = get_compile_time_arg_val(11);
    constexpr uint32_t REDUCE_ACC_VIA_ADD = get_compile_time_arg_val(12);
    // Refinement 2: NATIVE_IN == 1 means cb_input_tiles is BACKED ON THE INPUT
    // SHARD (ttnn.cb_descriptor_from_sharded_tensor).  x is already resident in
    // this core's L1, so there is no NoC read for it at all -- the reader only
    // PUBLISHES the pages once so cb_wait_front can see them.
    constexpr uint32_t NATIVE_IN = get_compile_time_arg_val(13);
    constexpr uint32_t IN_SHARD_PAGES = get_compile_time_arg_val(14);
    // Refinement 2b: BAND == 1 means this core stages x out of its OWN ROW_MAJOR
    // shard -- an RM shard that cuts the WIDTH axis, whose page is a row SEGMENT,
    // so the accessor cannot reach a row.  The core's shard IS its band (every
    // stick it owns x `shard_w` elements) at x_addr + local_stick * SHARD_ROW_BYTES,
    // and the cross-core combine sums the group's per-row partials elementwise, so
    // the band need not start or end on a tile column.
    constexpr uint32_t BAND = get_compile_time_arg_val(15);
    constexpr uint32_t SHARD_ROW_BYTES = get_compile_time_arg_val(16);
    // Whether the RM staging ring must be zeroed at boot (some staged stick is
    // narrower than the ring's padded row).  On the whole-row schemes this is
    // PARTIAL_W != 0; on the BAND scheme it is a band that does not fill its tile
    // columns, and there it REPLACES the reduce mask entirely.
    constexpr uint32_t STAGE_ZERO = get_compile_time_arg_val(17);
    constexpr auto x_args = TensorAccessorArgs<18>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    constexpr bool NATIVE_X = (NATIVE_IN != 0);
    constexpr bool BAND_X = (BAND != 0);
    constexpr bool RM = (IS_TILE == 0);
    static_assert(!BAND_X || IS_TILE == 0, "rms_norm: the BAND scheme is ROW_MAJOR-only");
    static_assert(
        !BAND_X || PARTIAL_W == 0, "rms_norm: the BAND scheme masks pad lanes by zero-staging, not by scaler");
    constexpr bool HAS_G = (HAS_GAMMA != 0);
    constexpr bool G_RM = (GAMMA_IS_RM != 0);
    // X_RESIDENT == GAMMA_RESIDENT == (NUM_W_CHUNKS == 1): one source of truth.
    constexpr bool X_RESIDENT = (NUM_W_CHUNKS == 1);
    constexpr uint32_t NUM_PASSES = X_RESIDENT ? 1 : 2;

    // Bytes of one full width chunk of a row-major stick, and of the last one
    // (short by the tile padding when W is not tile-aligned).
    constexpr uint32_t CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * ELEM_BYTES;
    constexpr uint32_t LAST_CHUNK_ROW_BYTES = W_ELEMS * ELEM_BYTES - (NUM_W_CHUNKS - 1) * CHUNK_ROW_BYTES;
    constexpr uint32_t G_CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * GAMMA_ELEM_BYTES;
    constexpr uint32_t G_LAST_CHUNK_ROW_BYTES = W_ELEMS * GAMMA_ELEM_BYTES - (NUM_W_CHUNKS - 1) * G_CHUNK_ROW_BYTES;

    // ---- runtime work assignment -----------------------------------------
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);  // this core's first tile-row
    const uint32_t num_rows = get_arg_val<uint32_t>(3);   // tile-rows owned by this core
    // Width slice this core owns (Lamp L1/L4).  On the whole-row schemes this is
    // (0, WT_CHUNK); under a width split it is the core's shard column range.
    const uint32_t w_start = get_arg_val<uint32_t>(4);
    const uint32_t w_real = get_arg_val<uint32_t>(5);  // REAL width tiles (<= WT_CHUNK)
    // The ROW_MAJOR view of the SAME slice: sticks and width ELEMENTS.  On the
    // tile-axis schemes these are derived host-side from (row_start, w_start) so
    // the two views cannot drift (_work_tile_axis); on the BAND scheme they are
    // the primary extents and are not tile-aligned on either axis.
    const uint32_t stick_base = get_arg_val<uint32_t>(6);    // first stick this core owns
    const uint32_t stick_count = get_arg_val<uint32_t>(7);   // sticks owned
    const uint32_t w_off_elems = get_arg_val<uint32_t>(8);   // first width element owned
    const uint32_t w_real_elems = get_arg_val<uint32_t>(9);  // REAL width elements owned

    // An INACTIVE core: it joined the program only so the width combine's stat
    // multicast lands in a cb_row_final this program owns (a width shard grid need
    // not be a rectangle, so the mcast box can be larger than the grid).  It holds
    // no shard, so it must not touch a shard-backed CB at all.
    if (num_rows == 0) {
        return;
    }

    const auto x_acc = TensorAccessor(x_args, x_addr);

    // ---- boot: what cb_scaler carries, per reduce datapath ----------------
    // Value is exactly 1.0 everywhere; 1/W is applied in fp32 by the compute
    // finalize, never folded into a bf16 scaler (R4).
    //
    //   ReduceTile       aligned : [full scaler]                   -> 1 tile
    //                    partial : [full scaler, partial scaler]   -> 2 tiles
    //   AccumulateViaAdd aligned : [scaler] (unused by the datapath, but keeps
    //                              the boot SrcB format real)      -> 1 tile
    //                    partial : [0/1 mask]                      -> 1 tile
    // The tile COUNT is the descriptor's SCALER_TILES, which the compute kernel
    // pops -- this branch must agree with it (asserted host-side).
    if constexpr (REDUCE_ACC_VIA_ADD != 0) {
        if constexpr (PARTIAL_W != 0) {
            // 0/1 mask in the row-0 broadcast layout AccumulateViaAdd's masked
            // accumulating broadcast-mul consumes for the last width tile.
            dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ckernel::ReduceDim::REDUCE_ROW>(PARTIAL_W);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
        }
    } else if constexpr (PARTIAL_W != 0) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<
            cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            PARTIAL_W>(1.0f);
    } else {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            1.0f);
    }

    // ---- boot: establish the pad-lane invariant on the RM staging ring ----
    if constexpr (RM && STAGE_ZERO != 0) {
        Noc noc;
        DataflowBuffer stage_dfb(cb_input_sticks);
        noc.async_write_zeros(stage_dfb, stage_dfb.get_total_size_bytes());
        noc.write_zeros_l1_barrier();
    }

    // ---- gamma: one chunk's worth of tiles (or sticks) --------------------
    // In RESIDENT this runs once per core before the row-block loop and the
    // tiles are never popped; in STREAM it runs per pass-B chunk.
    // `w_start` shifts every gamma index/offset by this core's width slice; on the
    // whole-row schemes it is 0 and this is byte-identical to Phase 0.
    auto stage_gamma_chunk = [&](uint32_t c) {
        if constexpr (HAS_G) {
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            const uint32_t first_wt = w_start + c * WT_CHUNK;
            if constexpr (G_RM) {
                // gamma is a single stick; row 0 of the staged tile-row is the
                // only row BroadcastDim::Row reads.
                constexpr uint32_t G_TILE_COL_BYTES = TILE_DIM * GAMMA_ELEM_BYTES;
                // TILE-COLUMN aligned by construction -- which is load-bearing, not
                // incidental: a DRAM read whose source offset is not 64-byte aligned
                // is silently truncated down to the alignment (measured: bands 1..3
                // of an 8-element shard all received gamma[0..8)).  That is exactly
                // why the BAND scheme stages into the GLOBAL tile frame (see
                // band_delta_bytes) rather than at the band's own byte offset: it
                // keeps every gamma fetch on a tile column, for x's dtype AND
                // gamma's, whatever the shard granule is.
                const uint32_t off = first_wt * G_TILE_COL_BYTES;
                const uint32_t total = W_ELEMS * GAMMA_ELEM_BYTES;
                const uint32_t remaining = (off < total) ? (total - off) : 0;
                const uint32_t row_bytes = (remaining < G_CHUNK_ROW_BYTES) ? remaining : G_CHUNK_ROW_BYTES;
                uint32_t pushed = 0;
                if (row_bytes != 0) {
                    dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_sticks>(
                        g_acc,
                        /*total_num_rows=*/1,
                        row_bytes,
                        /*start_page=*/0,
                        /*byte_offset_within_page=*/off);
                    // The helper pushes ceil(row_bytes / tile-column) pages, NOT
                    // WT_CHUNK (tilize_helpers_dataflow.inl width_in_tiles).
                    pushed = (row_bytes + G_TILE_COL_BYTES - 1) / G_TILE_COL_BYTES;
                }
                if (pushed < WT_CHUNK) {
                    // A RAGGED width shard's last core owns fewer real tile columns
                    // than WT_CHUNK.  tilize<WT_CHUNK> waits for the full block, so
                    // top the push up: those pages tilize into the PAD tile columns,
                    // whose product lands in the output shard's pad region and is
                    // never written back (the writer skips wt >= WT).
                    cb_reserve_back(cb_gamma_sticks, WT_CHUNK - pushed);
                    cb_push_back(cb_gamma_sticks, WT_CHUNK - pushed);
                }
            } else {
                const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
                cb_reserve_back(cb_gamma_tiles, WT_CHUNK);
                uint32_t l1_addr = get_write_ptr(cb_gamma_tiles);
                for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                    // A RAGGED width shard ends in whole PAD tiles that have no
                    // gamma counterpart; clamp so the read stays inside the tensor
                    // (the product lands in the output shard's pad region and is
                    // never read back).
                    const uint32_t wt = first_wt + w;
                    noc_async_read_tile((wt < WT) ? wt : (WT - 1), g_acc, l1_addr);
                    l1_addr += gamma_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma_tiles, WT_CHUNK);
            }
        }
    };

    if constexpr (X_RESIDENT) {
        stage_gamma_chunk(0);
    }

    // ---- native x: publish the resident shard, once ------------------------
    // The shard IS the per-core block, so the only thing to do is make its pages
    // visible to the compute kernel.  A RAGGED width shard (Wt not a multiple of
    // the shard's tile width) ends each of its tile-rows in whole PAD tiles whose
    // L1 content is undefined; zero them once so they contribute exactly 0 to
    // sum(x^2) (the same pad-lane invariant the ROW_MAJOR staging ring gets).
    if constexpr (NATIVE_X) {
        if (w_real < WT_CHUNK) {
            const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
            const uint32_t pad_tiles = WT_CHUNK - w_real;
            Noc noc;
            DataflowBuffer x_dfb(cb_input_tiles);
            for (uint32_t r = 0; r * WT_CHUNK < IN_SHARD_PAGES; ++r) {
                noc.async_write_zeros(
                    x_dfb, pad_tiles * tile_bytes, {.offset_bytes = (r * WT_CHUNK + w_real) * tile_bytes});
            }
            noc.write_zeros_l1_barrier();
        }
        cb_reserve_back(cb_input_tiles, IN_SHARD_PAGES);
        cb_push_back(cb_input_tiles, IN_SHARD_PAGES);
    }

    // ---- one width chunk of one row-block of x ---------------------------
    // Transaction granularity is WT_CHUNK tiles (one tile-row of the chunk):
    // a single knob-derived unit that divides every CB ring by construction,
    // and >= 4 tiles per barrier whenever the block allows it.
    const uint32_t x_tile_bytes = get_tile_size(cb_input_tiles);

    // ---- BAND staging: this core's own resident RM shard -> the tilize ring --
    // The band is `band_bytes` of every stick it owns, at x_addr + local_stick *
    // SHARD_ROW_BYTES; the ring wants it at the padded tile-column stride.  When
    // the band fills its tile columns exactly AND the shard stride matches, a
    // whole tile-row moves in ONE transaction; otherwise it is one per stick --
    // the same granularity the accessor path uses, but out of local L1 rather
    // than DRAM.  Trailing lanes keep the boot zero, so they add exactly 0 to
    // sum(x^2) and no reduce mask is needed (STAGE_ZERO / PARTIAL_W == 0).
    const uint32_t band_bytes = w_real_elems * ELEM_BYTES;
    // The band is staged in the tensor's GLOBAL TILE FRAME: its first element sits
    // at lane (w_off_elems % 32) of the staged stick, so staged tile column j IS
    // global width tile (w_off_elems / 32 + j).  Two things fall out of that, and
    // both are why the frame is not the band's own byte offset:
    //   * gamma (RM or TILE) is fetched at a tile-column offset, which is a multiple
    //     of 64 bytes for every dtype -- an unaligned DRAM read is silently
    //     truncated to the alignment, so an offset of "the band's first element"
    //     would hand three cores in four the WRONG weights;
    //   * the leading lanes [0, delta) and the trailing ones are the boot zeros, so
    //     they contribute exactly 0 to sum(x^2) -- no reduce mask, either side.
    // The shard granule keeps both the L1 source and the shifted destination
    // 16-byte aligned (w_off_elems is a multiple of L1_align/elem_size).
    const uint32_t band_delta_bytes = (w_off_elems % TILE_DIM) * ELEM_BYTES;
    const bool band_contiguous =
        (band_delta_bytes == 0) && (band_bytes == CHUNK_ROW_BYTES) && (SHARD_ROW_BYTES == CHUNK_ROW_BYTES);
    auto stage_band = [&](uint32_t stick_start, uint32_t sticks) {
        for (uint32_t s = 0; s < sticks; s += TILE_DIM) {
            const uint32_t n = ((sticks - s) < TILE_DIM) ? (sticks - s) : TILE_DIM;
            cb_reserve_back(cb_input_sticks, WT_CHUNK);
            const uint32_t dst = get_write_ptr(cb_input_sticks) + band_delta_bytes;
            const uint32_t src = x_addr + (stick_start + s - stick_base) * SHARD_ROW_BYTES;
            if (band_bytes != 0) {
                if (band_contiguous) {
                    noc_async_read(get_noc_addr(src), dst, n * CHUNK_ROW_BYTES);
                } else {
                    for (uint32_t i = 0; i < n; ++i) {
                        noc_async_read(get_noc_addr(src + i * SHARD_ROW_BYTES), dst + i * CHUNK_ROW_BYTES, band_bytes);
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_sticks, WT_CHUNK);
        }
    };

    auto stage_x_chunk = [&](uint32_t r0, uint32_t rows, uint32_t c) {
        const uint32_t first_tile_row = row_start + r0;
        if constexpr (NATIVE_X) {
            return;  // already resident and published above -- no NoC read for x
        } else if constexpr (RM) {
            const uint32_t stick_start = stick_base + r0 * TILE_DIM;
            uint32_t sticks = rows * TILE_DIM;
            if (r0 * TILE_DIM + sticks > stick_count) {
                sticks = stick_count - r0 * TILE_DIM;  // short final tile-row
            }
            if constexpr (BAND_X) {
                stage_band(stick_start, sticks);
            } else {
                const uint32_t row_bytes = (c + 1 == NUM_W_CHUNKS) ? LAST_CHUNK_ROW_BYTES : CHUNK_ROW_BYTES;
                dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
                    x_acc, sticks, row_bytes, stick_start, /*byte_offset_within_page=*/c * CHUNK_ROW_BYTES);
            }
        } else {
            for (uint32_t r = 0; r < rows; ++r) {
                // + w_start: this core's width slice under a cross-core width split
                // (0 on the whole-row schemes).  The ROW_MAJOR branch above needs no
                // such term: an RM build never takes a width-split plan.
                const uint32_t tile_base = (first_tile_row + r) * WT + w_start + c * WT_CHUNK;
                cb_reserve_back(cb_input_tiles, WT_CHUNK);
                uint32_t l1_addr = get_write_ptr(cb_input_tiles);
                for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                    noc_async_read_tile(tile_base + w, x_acc, l1_addr);
                    l1_addr += x_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_tiles, WT_CHUNK);
            }
        }
    };

    // ---- row-block loop ---------------------------------------------------
    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

        for (uint32_t pass = 0; pass < NUM_PASSES; ++pass) {
            for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
                stage_x_chunk(r0, rows, c);
                // STREAM: gamma is chunked and re-read for every pass-B chunk.
                if constexpr (!X_RESIDENT) {
                    if (pass == 1) {
                        stage_gamma_chunk(c);
                    }
                }
            }
        }
    }
}
