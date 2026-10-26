// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_stagger example writer (BRISC / NoC1).
//
// Drains one tilized block (chunk_wt tiles) at a time and writes its tiles to
// interleaved DRAM. With `stagger_blocks` (compile-time) on it walks the units
// starting at `blk_rot`, the same order the reader filled the CB in.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr bool stagger_blocks = get_compile_time_arg_val(3) != 0;
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const uint32_t n_w = get_arg_val<uint32_t>(3);      // chunks per tile-row
    const uint32_t wt = get_arg_val<uint32_t>(4);       // tiles per tile-row
    const uint32_t blk_rot = get_arg_val<uint32_t>(5);  // < unit_count; used only when stagger_blocks

    const auto out_acc = TensorAccessor(out_args, dst_addr, tile_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t j = 0; j < unit_count; ++j) {
            uint32_t off = j;
            if constexpr (stagger_blocks) {
                off = (j + blk_rot) < unit_count ? (j + blk_rot) : (j + blk_rot - unit_count);
            }
            const uint32_t u = unit_start + off;
            const uint32_t base_page = (u / n_w) * wt + (u % n_w) * chunk_wt;

            cb_wait_front(cb_out, chunk_wt);
            const uint32_t l1_addr = get_read_ptr(cb_out);
            for (uint32_t k = 0; k < chunk_wt; ++k) {
                noc_async_write(l1_addr + k * tile_bytes, out_acc.get_noc_addr(base_page + k), tile_bytes);
            }
            // The CB slot can be recycled once the data has left L1; no need to wait for acks.
            noc_async_writes_flushed();
            cb_pop_front(cb_out, chunk_wt);
        }
    }
    noc_async_write_barrier();
}
