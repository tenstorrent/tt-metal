// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// noc_balance reader (NCRISC). Same block/CB contract as the op's reader (reserve block_tiles,
// `valid` reads, barrier, push block_tiles); what changes is WHICH NoC each read is issued on and
// in WHICH ORDER this core walks its pages.
//
// NoC choice (define RD_NOC_MODE):
//   0 home NoC (noc_index) for every page                (op baseline)
//   1 alternate NoC per page  (i even -> home, odd -> other)  — needs DM_DYNAMIC_NOC
//   2 alternate NoC per block                                   — needs DM_DYNAMIC_NOC
//   3 every page on the other NoC                               — needs DM_DYNAMIC_NOC
//   4 per-core NoC from RT arg 5 (host picks by grid position)  — needs DM_DYNAMIC_NOC
//   5 per-bank NoC from RT arg 5 bitmask, bank = page % NB_NUM_BANKS — needs DM_DYNAMIC_NOC
// Page order: page(i) = start + ((i + rot) mod count) * stride   (RT args 1..4)
//   contiguous: stride 1, rot 0 (op baseline); rotated: rot != 0; strided: start=core, stride=ncores.
// Raw dataflow API (no helper bypassed: the op itself uses raw noc_async_read_page).
// CT args: [cb, block_tiles, page_bytes, TensorAccessorArgs...]
// RT args: [src_addr, start, count, stride, rot, noc_sel]

#include <cstdint>

#ifndef ABLATE_READ
#define ABLATE_READ 0  // bench ablation: stub this stream's NoC payload, keep CB sync
#endif

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef RD_NOC_MODE
#define RD_NOC_MODE 0
#endif
#ifndef NB_NUM_BANKS
#define NB_NUM_BANKS 8
#endif

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t rot = get_arg_val<uint32_t>(4);
    const uint32_t noc_sel = get_arg_val<uint32_t>(5);

    const auto acc = TensorAccessor(input_args, src_addr, page_bytes);
    const uint8_t home = noc_index;
    const uint8_t other = 1 - noc_index;

    MaybeDeviceZoneScope("reader_read");
    uint32_t j = rot;  // logical index within [0, count), wraps
    uint32_t blk = 0;
    for (uint32_t done = 0; done < count; done += block_tiles, ++blk) {
        const uint32_t remaining = count - done;
        const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
        cb_reserve_back(cb_input_tiles, block_tiles);
        uint32_t l1 = get_write_ptr(cb_input_tiles);
        for (uint32_t i = 0; i < valid; ++i) {
            const uint32_t page = start + j * stride;
            uint8_t noc;
            if constexpr (RD_NOC_MODE == 0) {
                noc = home;
            } else if constexpr (RD_NOC_MODE == 1) {
                noc = (i & 1) ? other : home;
            } else if constexpr (RD_NOC_MODE == 2) {
                noc = (blk & 1) ? other : home;
            } else if constexpr (RD_NOC_MODE == 3) {
                noc = other;
            } else if constexpr (RD_NOC_MODE == 4) {
                noc = noc_sel;
            } else {
                noc = (noc_sel >> (page % NB_NUM_BANKS)) & 1;
            }
            if constexpr (!ABLATE_READ) {
                noc_async_read_page(page, acc, l1, 0, noc);
            }
            l1 += page_bytes;
            if (++j == count) {
                j = 0;
            }
        }
        if constexpr (RD_NOC_MODE == 0) {
            noc_async_read_barrier(home);
        } else if constexpr (RD_NOC_MODE == 3) {
            noc_async_read_barrier(other);
        } else if constexpr (RD_NOC_MODE == 4) {
            noc_async_read_barrier(noc_sel);  // whole core on one NoC: one barrier
        } else {
            noc_async_read_barrier(0);
            noc_async_read_barrier(1);
        }
        cb_push_back(cb_input_tiles, block_tiles);
    }
}
