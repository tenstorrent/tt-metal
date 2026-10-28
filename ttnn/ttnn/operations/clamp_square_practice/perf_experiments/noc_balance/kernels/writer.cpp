// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// noc_balance writer (BRISC). Same block/CB contract as the op's writer (wait block_tiles,
// `valid` writes, flush, pop block_tiles, final write barrier); changes WHICH NoC each write
// uses and the page ORDER (must equal the reader's order — compute is in-order).
// WR_NOC_MODE: same encoding as RD_NOC_MODE in reader.cpp.
// CT args: [cb, block_tiles, page_bytes, TensorAccessorArgs...]
// RT args: [dst_addr, start, count, stride, rot, noc_sel]

#include <cstdint>

#ifndef ABLATE_WRITE
#define ABLATE_WRITE 0  // bench ablation: stub this stream's NoC payload, keep CB sync
#endif

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef WR_NOC_MODE
#define WR_NOC_MODE 0
#endif
#ifndef NB_NUM_BANKS
#define NB_NUM_BANKS 8
#endif

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t rot = get_arg_val<uint32_t>(4);
    const uint32_t noc_sel = get_arg_val<uint32_t>(5);

    const auto acc = TensorAccessor(output_args, dst_addr, page_bytes);
    const uint8_t home = noc_index;
    const uint8_t other = 1 - noc_index;

    {
        MaybeDeviceZoneScope("writer_write");
        uint32_t j = rot;
        uint32_t blk = 0;
        for (uint32_t done = 0; done < count; done += block_tiles, ++blk) {
            const uint32_t remaining = count - done;
            const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
            cb_wait_front(cb_output_tiles, block_tiles);
            uint32_t l1 = get_read_ptr(cb_output_tiles);
            for (uint32_t i = 0; i < valid; ++i) {
                const uint32_t page = start + j * stride;
                uint8_t noc;
                if constexpr (WR_NOC_MODE == 0) {
                    noc = home;
                } else if constexpr (WR_NOC_MODE == 1) {
                    noc = (i & 1) ? other : home;
                } else if constexpr (WR_NOC_MODE == 2) {
                    noc = (blk & 1) ? other : home;
                } else if constexpr (WR_NOC_MODE == 3) {
                    noc = other;
                } else if constexpr (WR_NOC_MODE == 4) {
                    noc = noc_sel;
                } else {
                    noc = (noc_sel >> (page % NB_NUM_BANKS)) & 1;
                }
                if constexpr (!ABLATE_WRITE) {
                    noc_async_write_page(page, acc, l1, page_bytes, 0, noc);
                }
                l1 += page_bytes;
                if (++j == count) {
                    j = 0;
                }
            }
            if constexpr (WR_NOC_MODE == 0) {
                noc_async_writes_flushed(home);
            } else if constexpr (WR_NOC_MODE == 3) {
                noc_async_writes_flushed(other);
            } else if constexpr (WR_NOC_MODE == 4) {
                noc_async_writes_flushed(noc_sel);  // whole core on one NoC: one flush
            } else {
                noc_async_writes_flushed(0);
                noc_async_writes_flushed(1);
            }
            cb_pop_front(cb_output_tiles, block_tiles);
        }
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        if constexpr (WR_NOC_MODE == 0) {
            noc_async_write_barrier(home);
        } else if constexpr (WR_NOC_MODE == 3) {
            noc_async_write_barrier(other);
        } else if constexpr (WR_NOC_MODE == 4) {
            noc_async_write_barrier(noc_sel);
        } else {
            noc_async_write_barrier(0);
            noc_async_write_barrier(1);
        }
    }
}
