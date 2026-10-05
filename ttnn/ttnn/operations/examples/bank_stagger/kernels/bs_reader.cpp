// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_stagger example reader (NCRISC / NoC0).
//
// For each of this core's work units (one 32-row x chunk_wt-tile block) it reads
// the block's 32 row slices ("sticks") of an interleaved ROW_MAJOR DRAM tensor into
// the reader->compute CB, then one barrier. Row r of the tensor is DRAM page r, and
// page r lives in bank r % num_banks.
//
// `stagger_reads` (compile-time) is the ONLY difference between variants. On, the
// 32 reads are issued starting at row `row_rot` and wrapping around; off, they go
// in order 0..31 and `row_rot` is not read. Every read still lands at the L1 offset
// of ITS row, so the block compute sees is byte-identical. When many cores start
// on the same bank (shared rows, or 32 % num_banks == 0), the in-order loop makes
// them all hit one bank at each step; a distinct row_rot per core spreads that step
// across banks. Same reads, same sizes, same count, same L1 -- only the order moves.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(3);
    constexpr bool stagger_reads = get_compile_time_arg_val(4) != 0;
    constexpr auto in_args = TensorAccessorArgs<5>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const uint32_t n_w = get_arg_val<uint32_t>(3);      // chunks per tile-row
    const uint32_t row_rot = get_arg_val<uint32_t>(4);  // used only when stagger_reads

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
            const uint32_t row0 = (u / n_w) * tile_rows;
            const uint32_t byte_offset = (u % n_w) * chunk_row_bytes;

            cb_reserve_back(cb_in, chunk_wt);
            const uint32_t l1_addr = get_write_ptr(cb_in);
            for (uint32_t i = 0; i < tile_rows; ++i) {
                uint32_t r = i;
                if constexpr (stagger_reads) {
                    r = (i + row_rot) < tile_rows ? (i + row_rot) : (i + row_rot - tile_rows);
                }
                noc_async_read(
                    in_acc.get_noc_addr(row0 + r, byte_offset), l1_addr + r * chunk_row_bytes, chunk_row_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, chunk_wt);
        }
    }
}
