// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copy core: writes the chunks its reader fetched (this chip's own shard) into this chip's own output slot.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric_all_gather_common.hpp"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t run_pages = get_compile_time_arg_val(3);
    constexpr uint32_t num_banks = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);
    constexpr bool kPrefixFromMetadata = get_compile_time_arg_val(7) != 0;
    constexpr auto out_args = TensorAccessorArgs<8>();
    constexpr auto prefix_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    // Common args: [0] output address [1..7] unused [8..15] geometry block.
    constexpr uint32_t kGeom = 8;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const Geometry geo = read_geometry<kPrefixFromMetadata>(kGeom, page_bytes, get_write_ptr(cb_meta), prefix_args);

    // Per-core args: [0] first bank [1] bank stride [2] this chip's rank (output slot).
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t stride = get_arg_val<uint32_t>(1);
    const uint32_t rank = get_arg_val<uint32_t>(2);
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    uint32_t chunks_left = fag::port_chunks(geo.num_stripes, geo.stripe_pages, num_banks, run_pages, first, stride, 0);
    constexpr uint32_t cb_chunks = 2 * group;
    uint32_t pending = 0, offset = 0;
    fag::for_each_chunk(
        geo.num_stripes,
        geo.stripe_pages,
        num_banks,
        run_pages,
        first,
        stride,
        0,
        0,
        [&](uint32_t stripe, uint32_t page, uint32_t n, uint32_t) {
            cb_wait_front(cb, run_pages * (pending + 1));
            noc_async_write(
                get_read_ptr(cb) + pending * chunk_bytes,
                out.get_noc_addr(output_page(geo, rank, stripe, page)),
                n * page_bytes);
            --chunks_left;
            const uint32_t cap = group < cb_chunks - offset ? group : cb_chunks - offset;
            if (++pending == cap || chunks_left == 0) {
                noc_async_write_barrier();
                cb_pop_front(cb, run_pages * pending);
                offset += pending;
                if (offset >= cb_chunks) {
                    offset -= cb_chunks;
                }
                pending = 0;
            }
        });
}
