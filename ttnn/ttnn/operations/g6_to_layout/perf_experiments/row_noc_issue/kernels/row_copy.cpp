// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row I/O bench: one core walks units (tile-row tr, column block cb) of an interleaved ROW_MAJOR
// tensor: up to 32 rows (pages 32 tr ..), segment [cb * block_bytes, +block_bytes) of each row.
//   do_read  : one read per row into an L1 stage slot (row stride block_bytes), trid-pipelined one
//              unit ahead (two slots), as a row-major -> tile reader does.
//   do_write : one write per row from the stage slot, flushed per unit, as a tile -> row-major writer.
// Both on = check mode (copy input -> output, the variant under test on one side). One side only =
// time mode (the stage holds whatever L1 holds; the output is not checked).
// The issue loops are timed with the wall clock; per-core counters go to one 64-B stats page.
//
// write_variant ISSUE_RUN_COALESCE (5): full-row units only. The stage is run-major: the rows of
// one bank (pages p, p + NB, ...) are adjacent at the aligned page stride, so each bank's run of a
// unit is ONE write of count * aligned_page bytes (consecutive slots of that bank). A plain
// row-major stage (rows in page order) cannot be written this way.

#include "api/dataflow/dataflow_api.h"
#include "row_issue.hpp"

using namespace row_issue;

constexpr uint32_t ISSUE_RUN_COALESCE = 5;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(2);
    constexpr bool do_read = get_compile_time_arg_val(3) != 0;
    constexpr bool do_write = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t read_variant = get_compile_time_arg_val(5);
    constexpr uint32_t write_variant = get_compile_time_arg_val(6);
    constexpr uint32_t rotate = get_compile_time_arg_val(7);
    constexpr uint32_t iters = get_compile_time_arg_val(8);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(9);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(10);
    constexpr auto in_args = TensorAccessorArgs<11>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto st_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1));
    const auto st = TensorAccessor(st_args, get_common_arg_val<uint32_t>(2));
    const uint32_t unit_begin = get_arg_val<uint32_t>(0);
    const uint32_t unit_end = get_arg_val<uint32_t>(1);
    const uint32_t core = get_arg_val<uint32_t>(2);
    // rotate == ROTATE_HOST: the host's start row of this core (bank spread across cores).
    const uint32_t host_r0 = get_arg_val<uint32_t>(3);
    constexpr uint32_t ROTATE_HOST = 0xFFFF;

    constexpr uint32_t col_blocks = (row_bytes + block_bytes - 1) / block_bytes;
    constexpr uint32_t slot_bytes = 32 * block_bytes;
    constexpr bool run_major = do_write && write_variant == ISSUE_RUN_COALESCE;
    static_assert(!run_major || col_blocks == 1, "run coalescing needs full-row units");
    constexpr uint32_t NB = decltype(cursor_at(out, 0, 0))::NB;
    const uint32_t stage = get_write_ptr(cb_stage);
    const uint32_t out_aligned = out.get_aligned_page_size();

    uint32_t read_cycles = 0, write_cycles = 0, reads = 0, writes = 0, n = 0;
    const uint32_t t_start = get_timestamp_32b();
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t u = unit_begin; u < unit_end; ++u, ++n) {
            const uint32_t tr = u / col_blocks;
            const uint32_t cb = u - tr * col_blocks;
            const uint32_t first_page = tr * 32;
            const uint32_t rows = (H - first_page) < 32 ? (H - first_page) : 32;
            const uint32_t offset = cb * block_bytes;
            const uint32_t bytes = (row_bytes - offset) < block_bytes ? (row_bytes - offset) : block_bytes;
            const uint32_t slot = stage + (n & 1) * slot_bytes;
            const uint32_t r0 = rotate == ROTATE_HOST ? host_r0 % rows : rotate ? (core * rotate) % rows : 0;
            // Run-major placement: phase j (pages first_page + j + k NB) starts at run slot pos(j).
            const uint32_t per_phase = rows / NB;
            const uint32_t extra = rows % NB;

            if constexpr (do_read) {
                noc_async_read_set_trid(1 + (n & 1));
                const uint32_t t0 = get_timestamp_32b();
                if constexpr (run_major) {
                    for (uint32_t r = 0; r < rows; ++r) {
                        const uint32_t j = r % NB;
                        const uint32_t pos = j * per_phase + (j < extra ? j : extra) + r / NB;
                        noc_async_read(in.get_noc_addr(first_page + r, 0), slot + pos * out_aligned, bytes);
                    }
                } else {
                    read_rows<read_variant>(
                        in, first_page + r0, rows - r0, offset, bytes, slot + r0 * block_bytes, block_bytes);
                    if (r0) {
                        read_rows<read_variant>(in, first_page, r0, offset, bytes, slot, block_bytes);
                    }
                }
                read_cycles += get_timestamp_32b() - t0;
                reads += rows;
                if constexpr (do_write) {
                    noc_async_read_barrier_with_trid(1 + (n & 1));
                } else if (n > 0) {
                    noc_async_read_barrier_with_trid(1 + ((n - 1) & 1));
                }
            }
            if constexpr (do_write) {
                const uint32_t t0 = get_timestamp_32b();
                if constexpr (run_major) {
                    const uint32_t phases = rows < NB ? rows : NB;
                    const uint32_t j0 = rotate == ROTATE_HOST ? host_r0 % phases
                                        : rotate              ? (core * rotate) % phases
                                                              : 0;
                    for (uint32_t jj = 0; jj < phases; ++jj) {
                        const uint32_t j = (jj + j0) % phases;
                        const uint32_t count = per_phase + (j < extra ? 1 : 0);
                        const uint32_t pos = j * per_phase + (j < extra ? j : extra);
                        noc_async_write(
                            slot + pos * out_aligned, out.get_noc_addr(first_page + j, 0), count * out_aligned);
                    }
                    writes += phases;
                } else {
                    write_rows<write_variant>(
                        out, first_page + r0, rows - r0, offset, bytes, slot + r0 * block_bytes, block_bytes);
                    if (r0) {
                        write_rows<write_variant>(out, first_page, r0, offset, bytes, slot, block_bytes);
                    }
                    writes += rows;
                }
                write_cycles += get_timestamp_32b() - t0;
                noc_async_writes_flushed();
            }
        }
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
    noc_async_write_barrier();
    const uint32_t total_cycles = get_timestamp_32b() - t_start;

    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stats));
    stats[0] = unit_end - unit_begin;
    stats[1] = reads;
    stats[2] = writes;
    stats[3] = read_cycles;
    stats[4] = write_cycles;
    stats[5] = total_cycles;
    stats[6] = core;
    stats[7] = 0x600DF00D;
    for (uint32_t k = 8; k < 16; ++k) {
        stats[k] = 0;
    }
    noc_async_write(get_write_ptr(cb_stats), st.get_noc_addr(core), 64);
    noc_async_write_barrier();
}
