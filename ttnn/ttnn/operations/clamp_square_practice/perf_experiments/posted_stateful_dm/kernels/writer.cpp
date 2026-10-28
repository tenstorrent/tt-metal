// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// posted_stateful_dm bench writer (BRISC / NoC1). Same block/CB structure as the op's writer
// (one wait, `valid` writes, one flush, one pop per block; final barrier).
//
// WR_MODE (define): 0 = noc_async_write_page (op's current writer), 1 = bank-address cache +
//   noc_async_write_one_packet, 2 = bank-address cache + one-packet STATEFUL write
//   (ncrisc_noc_write_set_state once: CTRL/MID/LEN; per tile: poll, rewrite NOC_RET_ADDR_COORDINATE,
//   ncrisc_noc_write_with_state<one_packet, skip_cmdbuf_chk>).
// WR_POSTED (define): 0 = all writes non-posted (acked). 1 = posted (no ack) for every tile except
//   the core's last min(NB, core_tiles) tiles, which are non-posted "fence" writes. Contiguous pages
//   rotate through all NB banks, so the tail touches every bank this core wrote; a non-posted write
//   travels the same src->bank route on the same VC (NOC_UNICAST_WRITE_VC) as the earlier posted
//   ones, so its ack (waited by the final noc_async_write_barrier) implies those landed first.
//   2 = all posted, no fence (NOT durable; only to price the fence).
// Slot retirement: noc_async_writes_flushed + noc_async_posted_writes_flushed (both "departed").
// RAW-LLK justification (mode 2): bypasses noc_async_write_page/TensorAccessor; the public
// one_packet_with_state API cannot retarget the bank, so the coordinate register is written raw.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef WR_STUB
#define WR_STUB 0  // 1 = writes stubbed (reader-only ablation), sync kept
#endif
#ifndef WR_MODE
#define WR_MODE 0
#endif
#ifndef WR_POSTED
#define WR_POSTED 0
#endif

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_tiles = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, dst_addr, page_bytes);
    constexpr uint32_t NB = NUM_DRAM_BANKS;
    // First tile index (within this core's range) that is written non-posted.
#if WR_POSTED == 1
    const uint32_t fence_start = core_tiles > NB ? core_tiles - NB : 0;
#elif WR_POSTED == 2
    const uint32_t fence_start = core_tiles;
#else
    const uint32_t fence_start = 0;
#endif

#if WR_MODE != 0
    uint32_t coord[NB];
    uint32_t lo[NB];
    const uint32_t ncache = core_tiles < NB ? core_tiles : NB;
    for (uint32_t k = 0; k < ncache; ++k) {
        const uint64_t a = output_accessor.get_noc_addr(start_tile + k);
        coord[k] = (uint32_t)(a >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
        lo[k] = (uint32_t)a;
    }
    uint32_t k = 0;
#endif
#if WR_MODE == 2
    if (core_tiles > 0) {
        if (fence_start > 0) {
            ncrisc_noc_write_set_state<true /*posted*/, true /*one_packet*/>(
                noc_index, write_cmd_buf, output_accessor.get_noc_addr(start_tile), page_bytes, NOC_UNICAST_WRITE_VC);
        } else {
            ncrisc_noc_write_set_state<false, true>(
                noc_index, write_cmd_buf, output_accessor.get_noc_addr(start_tile), page_bytes, NOC_UNICAST_WRITE_VC);
        }
    }
#endif

    {
        MaybeDeviceZoneScope("writer_write");
        uint32_t t = 0;  // tile index within this core's range
        for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
            const uint32_t remaining = core_tiles - done;
            const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
            cb_wait_front(cb_output_tiles, block_tiles);
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
#if WR_MODE == 0
            const uint32_t page0 = start_tile + done;
#endif
            for (uint32_t i = 0; i < valid && !WR_STUB; ++i, ++t) {
                const bool posted = t < fence_start;
#if WR_MODE == 0
                if (posted) {
                    noc_async_write_page<decltype(output_accessor), true, true>(
                        page0 + i, output_accessor, l1_read_addr);
                } else {
                    noc_async_write_page(page0 + i, output_accessor, l1_read_addr);
                }
#elif WR_MODE == 1
                const uint64_t a = ((uint64_t)coord[k] << NOC_ADDR_COORD_SHIFT) | lo[k];
                if (posted) {
                    noc_async_write_one_packet<true, true>(l1_read_addr, a, page_bytes);
                } else {
                    noc_async_write_one_packet<true, false>(l1_read_addr, a, page_bytes);
                }
                lo[k] += page_bytes;
                k = (k + 1 == NB) ? 0 : k + 1;
#else
                while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
                if (t == fence_start && fence_start > 0) {
                    // switch the stateful CTRL to non-posted (RESP_MARKED) for the fence tail
                    NOC_CMD_BUF_WRITE_REG(
                        noc_index,
                        write_cmd_buf,
                        NOC_CTRL,
                        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) |
                            NOC_CMD_RESP_MARKED);
                }
                NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE, coord[k]);
                if (posted) {
                    ncrisc_noc_write_with_state<noc_mode, true, true, true, true>(
                        noc_index, write_cmd_buf, l1_read_addr, lo[k]);
                } else {
                    ncrisc_noc_write_with_state<noc_mode, false, true, true, true>(
                        noc_index, write_cmd_buf, l1_read_addr, lo[k]);
                }
                lo[k] += page_bytes;
                k = (k + 1 == NB) ? 0 : k + 1;
#endif
                l1_read_addr += page_bytes;
            }
#if WR_POSTED == 0
            noc_async_writes_flushed();
#else
            noc_async_writes_flushed();
            noc_async_posted_writes_flushed();
#endif
            cb_pop_front(cb_output_tiles, block_tiles);
        }
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
#if WR_POSTED != 0
        noc_async_posted_writes_flushed();
#endif
    }
}
