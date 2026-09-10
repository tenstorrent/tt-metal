// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF ARTIFACT — idea `writer_bank_rotate`. NOT the real op.
//
// This is a byte-for-byte reconstruction of the real op's writer
// (`ttnn/ttnn/operations/tilize/kernels/tilize_writer.cpp`, `store_block`),
// with exactly ONE change: the ORDER in which the `block_width_tiles` writes
// of a tile-row are issued is permuted by a core-dependent rotation, selected
// at compile time by `WRITER_ROTATE_MODE`. Same bytes, same destinations,
// same transaction count/size, same single barrier per batch — only issue
// ORDER changes, which is why this reorder is "free" (see the real kernel's
// comment on why every write's L1 source address is computable from `i`
// directly rather than from an incrementing cursor).
//
// WHY THIS IS SAFE / WHY IT MIGHT WIN — see the task brief: on an interleaved
// DRAM tensor, output tile page `p` lives on bank `p % NUM_DRAM_BANKS`. On the
// focus shape (`block_width_tiles == 8`, `NUM_DRAM_BANKS == 12`,
// `gcd(8,12) == 4`), core `k`'s block starts at page `8k`, so at a FIXED
// issue step `i` (roughly lockstep across the 64 cores, since every core
// does identical work), all 64 cores write to `(8k + i) % 12`, which for
// `k = 0..63` only ever takes 3 distinct values. Rotating step `i` by a
// core-dependent amount desynchronizes which bank each core hits at any
// given instant, without touching the SET of banks a core's own 8 writes
// eventually touch (that set is fixed by `page_base` and `block_width_tiles`
// alone; only the temporal order changes).
//
// RAW-API JUSTIFICATION (inherited from the real kernel, restated here since
// this file is the raw-LLK sandbox for this idea): the same two dataflow
// helpers (`write_sticks_after_untilize`, `local_copy_helpers_dataflow`) are
// checked against this destination in `tilize_writer.cpp` and both mismatch
// concretely (ROW_MAJOR-stick addressing vs this TILE-page destination; L1-only
// destination vs this DRAM destination). Bypassed here: raw `noc_async_write`
// issued in a HOST- and CORE-dependent permuted order — no existing helper
// exposes an issue-order knob at all, so the gap is CAPABILITY, not
// ergonomics. If a mode wins, the closing helper is the same
// `write_tile_pages_for_tilize<cb>(...)` gap the real kernel already names,
// with one more parameter: an issue-order permutation (or the equivalent
// "rotate by this amount") argument.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

// Rotation modes (host picks one via `defines={"WRITER_ROTATE_MODE": "<n>"}`):
//   0 = BASELINE   — ascending i, i.e. today's op verbatim (the honest baseline).
//   1 = CHUNK      — rotate the column loop by this core's `w_chunk`.
//   2 = BLOCKID    — rotate the column loop by this core's real `block_id`
//                    (== w_chunk here since num_row_groups==1 on every sweep
//                    shape below, but the general derivation differs once a
//                    plan has num_row_groups > 1 — kept distinct on purpose).
//   3 = BANKEXACT  — derive the rotation so this core's FIRST write lands on
//                    bank `block_id % NUM_DRAM_BANKS` exactly, when reachable
//                    from the `block_width_tiles` writes available; falls
//                    back to BASELINE order for a core/row whose target bank
//                    is not in the reachable window (see derivation below).
//   4 = ROWS+COLS  — mode 2's column rotation ALSO applied to the tile-ROW
//                    order within a multi-row write batch
//                    (`write_rows_per_barrier > 1`), so a tall-narrow block
//                    (bw==1, all the "rotation" lives in the row axis) gets a
//                    rotation too.
#ifndef WRITER_ROTATE_MODE
#define WRITER_ROTATE_MODE 0
#endif
#ifndef NUM_DRAM_BANKS
#define NUM_DRAM_BANKS 12
#endif

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(2);  // R
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);   // C
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t write_rows_per_barrier = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t col_tile_offset = get_compile_time_arg_val(8);
    // split-reader compile args carried for CT-arg-index compatibility with
    // the real op's writer_ct_args list — this bench never turns them on.
    constexpr uint32_t split_reader_rows = get_compile_time_arg_val(9);
    constexpr uint32_t cb_input_rows_split = get_compile_time_arg_val(10);
    constexpr uint32_t tile_h = get_compile_time_arg_val(11);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t split_writer_share_pct = get_compile_time_arg_val(13);
    // NOT this idea's target: the split-reader block is a READ the writer
    // issues on the trailing rows of a block (Refinement 6), left byte-for-
    // byte verbatim from the real kernel so shapes that trigger it (e.g. the
    // tall-narrow sweep point, bw==1 / small row bytes) still compile and run
    // correctly. No rotation is applied to it — only `store_block`'s WRITE
    // loop below is this idea's lever.
    constexpr uint32_t col_byte_offset = col_tile_offset * (block_row_bytes / block_width_tiles);
    constexpr auto out_args = TensorAccessorArgs<14>();
    [[maybe_unused]] constexpr auto in_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t block_stride = get_arg_val<uint32_t>(3);
    const uint32_t src_addr = get_arg_val<uint32_t>(4);

    const auto out_acc = TensorAccessor(out_args, dst_addr);
    [[maybe_unused]] const auto in_acc = TensorAccessor(in_args, src_addr);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;
        const uint32_t col_base = col_tile_offset + w_chunk * block_width_tiles;

        // The rotation amount for THIS block's column loop. Modes 1/2 differ
        // only in which core-dependent quantity drives them; on every sweep
        // shape in this bench (num_row_groups == 1) block_id == w_chunk, so
        // modes 1 and 2 are numerically identical there — kept as separate
        // dials because they diverge on a plan with num_row_groups > 1
        // (untested by this bench; see the report's domain note).
#if WRITER_ROTATE_MODE == 1
        const uint32_t col_rot = block_width_tiles ? (w_chunk % block_width_tiles) : 0;
#elif WRITER_ROTATE_MODE == 2 || WRITER_ROTATE_MODE == 4
        const uint32_t col_rot = block_width_tiles ? (block_id % block_width_tiles) : 0;
#elif WRITER_ROTATE_MODE == 3
        // BANKEXACT derivation. Target: core's block lands its FIRST write on
        // bank (block_id % NUM_DRAM_BANKS). Reachable iff the required delta
        // `d` (mod NUM_DRAM_BANKS) between the target bank and col_base's own
        // bank falls inside the window of `block_width_tiles` CONSECUTIVE
        // writes this block actually issues (i = 0 .. block_width_tiles-1
        // maps to bank (col_base_page + i) % NUM_DRAM_BANKS, i.e. only
        // `block_width_tiles` of the `NUM_DRAM_BANKS` residues are ever
        // reachable by this block at all, regardless of order — reordering
        // can only pick WHEN inside that fixed reachable set the target is
        // hit, never add a bank outside it). Falls back to i=0 (baseline
        // order) when the target bank isn't in the reachable window.
        const uint32_t col_base_page0 = row_start * tensor_col_tiles + col_base;  // bank of i=0, this block's first row
        const uint32_t target_bank = block_id % NUM_DRAM_BANKS;
        const uint32_t base_bank = col_base_page0 % NUM_DRAM_BANKS;
        const uint32_t delta = (target_bank + NUM_DRAM_BANKS - base_bank) % NUM_DRAM_BANKS;
        const uint32_t col_rot = (delta < block_width_tiles) ? delta : 0;
#else
        constexpr uint32_t col_rot = 0;  // WRITER_ROTATE_MODE == 0 (baseline)
#endif

        // load_block, TRAILING HALF (Refinement 6, verbatim from the real
        // kernel) — inert (compiles to nothing) at split_reader_rows == 0,
        // which is every sweep shape except the tall-narrow one.
        if constexpr (split_reader_rows > 0) {
            uint32_t rows_writer = (block_row_extent * split_writer_share_pct) / 100;
            if (rows_writer >= block_row_extent) {
                rows_writer = block_row_extent - 1;
            }
            if (rows_writer > 0) {
                MaybeDeviceZoneScope("writer_split_read");
                const uint32_t rows_reader = block_row_extent - rows_writer;
#ifdef TILIZE_ABLATE_READS
                for (uint32_t tr = 0; tr < rows_writer; ++tr) {
                    cb_reserve_back(cb_input_rows_split, block_width_tiles);
                    noc_async_read_barrier();
                    cb_push_back(cb_input_rows_split, block_width_tiles);
                }
#else
                dataflow_kernel_lib::
                    read_sticks_for_tilize<cb_input_rows_split, dataflow_kernel_lib::TilizeGranularity::TILE>(
                        in_acc,
                        /* total_num_rows          */ rows_writer * tile_h,
                        /* row_bytes               */ block_row_bytes,
                        /* start_page              */ (row_start + rows_reader) * tile_h,
                        /* byte_offset_within_page */ col_byte_offset + w_chunk * block_row_bytes);
#endif
            }
        }

        uint32_t rows_done = 0;
        while (rows_done < block_row_extent) {
            uint32_t rows_this_batch = block_row_extent - rows_done;
            if (rows_this_batch > write_rows_per_barrier) {
                rows_this_batch = write_rows_per_barrier;
            }
            {
                const LocalCBInterface& cb = get_local_cb_interface(cb_output_tiles);
                const uint32_t contig_rows = ((cb.fifo_limit - cb.fifo_rd_ptr) / cb.fifo_page_size) / block_width_tiles;
                if (rows_this_batch > contig_rows) {
                    rows_this_batch = contig_rows;
                }
            }

            const uint32_t pages_this_batch = rows_this_batch * block_width_tiles;
            {
                MaybeDeviceZoneScope("writer_wait_out");
                cb_wait_front(cb_output_tiles, pages_this_batch);
            }

            const uint32_t l1_read_base = get_read_ptr(cb_output_tiles);
            {
                MaybeDeviceZoneScope("writer_issue");
#if WRITER_ROTATE_MODE == 4
                // ROWS+COLS: rotate the row-issue order too, by the same
                // core-dependent amount (mod rows_this_batch), so a
                // multi-row-per-barrier batch also desynchronizes ACROSS
                // cores in the row axis, not just the column axis.
                const uint32_t row_rot = rows_this_batch ? (block_id % rows_this_batch) : 0;
                for (uint32_t rs = 0; rs < rows_this_batch; ++rs) {
                    const uint32_t r = (rs + row_rot) % rows_this_batch;
                    const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;
                    const uint32_t l1_row_addr = l1_read_base + r * block_width_tiles * out_tile_bytes;
                    for (uint32_t cs = 0; cs < block_width_tiles; ++cs) {
                        const uint32_t i = (cs + col_rot) % block_width_tiles;
                        const uint32_t l1_read_addr = l1_row_addr + i * out_tile_bytes;
#ifdef TILIZE_ABLATE_WRITES
                        (void)out_acc.get_noc_addr(page_base + i);
#else
                        noc_async_write<out_tile_bytes>(
                            l1_read_addr, out_acc.get_noc_addr(page_base + i), out_tile_bytes);
#endif
                    }
                }
#else
                for (uint32_t r = 0; r < rows_this_batch; ++r) {
                    const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;
                    const uint32_t l1_row_addr = l1_read_base + r * block_width_tiles * out_tile_bytes;
                    for (uint32_t cs = 0; cs < block_width_tiles; ++cs) {
#if WRITER_ROTATE_MODE == 0
                        const uint32_t i = cs;
#else
                        const uint32_t i = (cs + col_rot) % block_width_tiles;
#endif
                        const uint32_t l1_read_addr = l1_row_addr + i * out_tile_bytes;
#ifdef TILIZE_ABLATE_WRITES
                        (void)out_acc.get_noc_addr(page_base + i);
#else
                        noc_async_write<out_tile_bytes>(
                            l1_read_addr, out_acc.get_noc_addr(page_base + i), out_tile_bytes);
#endif
                    }
                }
#endif
            }

            {
                MaybeDeviceZoneScope("writer_barrier");
                noc_async_write_barrier();
            }
            cb_pop_front(cb_output_tiles, pages_this_batch);
            rows_done += rows_this_batch;
        }
    }
}
