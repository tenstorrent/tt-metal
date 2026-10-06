// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — final-core BRISC: store_output_block.
//
// The summed own block's segments of this final's set go to the output (tile (row, col) at page
// row * blk_n_tiles + col; only the valid tiles of a row's last segment). One write barrier per xport_group
// segments (and at the CB wrap / the end). Then re-arm the arrival counters A / B.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_xport_sum = get_compile_time_arg_val(0);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);
    constexpr uint32_t cap_segs = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;

    size_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t segs_per_block = get_arg_val<uint32_t>(arg++);
    const uint32_t segs_per_row = get_arg_val<uint32_t>(arg++);
    const uint32_t blk_n_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t first_seg = get_arg_val<uint32_t>(arg++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(arg++);
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t expect_a = get_arg_val<uint32_t>(arg++);
    const uint32_t b_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t expect_b = get_arg_val<uint32_t>(arg++);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    uint32_t pending = 0, rpos = 0;
    for (uint32_t seg = first_seg; seg < segs_per_block; seg += seg_stride) {
        const uint32_t row = seg / segs_per_row;
        const uint32_t c0 = (seg - row * segs_per_row) * seg_tiles;
        const uint32_t valid = blk_n_tiles - c0 < seg_tiles ? blk_n_tiles - c0 : seg_tiles;
        cb_wait_front(cb_xport_sum, seg_tiles * (pending + 1));
        const uint32_t src = get_read_ptr(cb_xport_sum) + pending * seg_bytes;
        const uint32_t page0 = row * blk_n_tiles + c0;
        for (uint32_t t = 0; t < valid; ++t) {
            noc_async_write(src + t * tile_bytes, out.get_noc_addr(page0 + t), tile_bytes);
        }
        ++pending;
        if (pending == group || rpos + pending == cap_segs || seg + seg_stride >= segs_per_block) {
            noc_async_write_barrier();
            cb_pop_front(cb_xport_sum, seg_tiles * pending);
            rpos += pending;
            if (rpos == cap_segs) {
                rpos = 0;
            }
            pending = 0;
        }
    }
    if (expect_a > 0) {
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_addr);
        noc_semaphore_wait_min(c, expect_a);
        noc_semaphore_inc(get_noc_addr(a_addr), 0u - expect_a);
    }
    if (expect_b > 0) {
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);
        noc_semaphore_wait_min(c, expect_b);
        noc_semaphore_inc(get_noc_addr(b_addr), 0u - expect_b);
    }
    noc_async_atomic_barrier();
}
