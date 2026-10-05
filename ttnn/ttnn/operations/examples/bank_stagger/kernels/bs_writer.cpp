// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_stagger example writer (BRISC / NoC1).
//
// Drains one tilized block (chunk_wt tiles) at a time and writes its tiles to
// interleaved DRAM. Output tile page p lives in bank p % num_banks; a block's
// chunk_wt tiles are consecutive pages, so with every core writing them in
// ascending order the cores only ever START on a few banks.
//
// `stagger_writes` (compile-time) is the ONLY difference between variants. On, the
// chunk_wt writes are issued starting at tile `col_rot` and wrapping; off, they go
// in order and `col_rot` is not read. Each tile still goes to its own page from its
// own L1 slot -- only the issue order moves.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr bool stagger_writes = get_compile_time_arg_val(3) != 0;
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const uint32_t n_w = get_arg_val<uint32_t>(3);      // chunks per tile-row
    const uint32_t wt = get_arg_val<uint32_t>(4);       // tiles per tile-row
    const uint32_t col_rot = get_arg_val<uint32_t>(5);  // used only when stagger_writes

    const auto out_acc = TensorAccessor(out_args, dst_addr, tile_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
            const uint32_t base_page = (u / n_w) * wt + (u % n_w) * chunk_wt;

            cb_wait_front(cb_out, chunk_wt);
            const uint32_t l1_addr = get_read_ptr(cb_out);
            for (uint32_t i = 0; i < chunk_wt; ++i) {
                uint32_t k = i;
                if constexpr (stagger_writes) {
                    k = (i + col_rot) < chunk_wt ? (i + col_rot) : (i + col_rot - chunk_wt);
                }
                noc_async_write(l1_addr + k * tile_bytes, out_acc.get_noc_addr(base_page + k), tile_bytes);
            }
            // The CB slot can be recycled once the data has left L1; no need to wait for acks.
            noc_async_writes_flushed();
            cb_pop_front(cb_out, chunk_wt);
        }
    }
    noc_async_write_barrier();
}
