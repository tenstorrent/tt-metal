// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// posted_stateful_dm bench reader (NCRISC / NoC0). Same block/CB structure as the op's reader
// (one reserve, `valid` reads, ONE barrier, one push of the nominal block per block).
//
// RD_MODE (define):
//   0 = baseline: noc_async_read_page(page, TensorAccessor, l1)  (the op's current reader)
//   1 = bank-address cache + noc_async_read_one_packet (plain issue, cached 64-bit NoC addr)
//   2 = bank-address cache + one-packet STATEFUL read: ncrisc_noc_read_set_state once (MID, LEN),
//       then per tile: poll cmd buf, rewrite NOC_TARG_ADDR_COORDINATE (bank xy changes every page
//       on interleaved DRAM), issue ncrisc_noc_read_with_state<one_packet, skip_cmdbuf_chk>.
// RAW-LLK justification (mode 2): bypasses the TensorAccessor/noc_async_read_page helper. The public
// one_packet_with_state API cannot change the target coordinate, and interleaved pages rotate
// across banks every tile, so the coordinate register is rewritten raw between the poll and issue.
// Bank cache: contiguous pages p, p+NB, p+2NB... live in the same bank at +page_bytes strides, so
// the accessor (division/bank-table lookup) is called only for the first NB pages of the range.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef RD_STUB
#define RD_STUB 0  // 1 = reads stubbed (writer-only ablation), sync kept
#endif
#ifndef RD_MODE
#define RD_MODE 0
#endif

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_tiles = get_arg_val<uint32_t>(2);

    const auto input_accessor = TensorAccessor(input_args, src_addr, page_bytes);

#if RD_MODE != 0
    constexpr uint32_t NB = NUM_DRAM_BANKS;
    uint32_t coord[NB];
    uint32_t lo[NB];
    const uint32_t ncache = core_tiles < NB ? core_tiles : NB;
    for (uint32_t k = 0; k < ncache; ++k) {
        const uint64_t a = input_accessor.get_noc_addr(start_tile + k);
        coord[k] = (uint32_t)(a >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
        lo[k] = (uint32_t)a;
    }
    uint32_t k = 0;
#endif
#if RD_MODE == 2
    if (core_tiles > 0) {
        ncrisc_noc_read_set_state<noc_mode, true /*one_packet*/>(
            noc_index, read_cmd_buf, input_accessor.get_noc_addr(start_tile), page_bytes);
    }
#endif

    MaybeDeviceZoneScope("reader_read");
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t remaining = core_tiles - done;
        const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
        cb_reserve_back(cb_input_tiles, block_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
#if RD_MODE == 0
        const uint32_t page0 = start_tile + done;
#endif
        for (uint32_t i = 0; i < valid && !RD_STUB; ++i) {
#if RD_MODE == 0
            noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
#elif RD_MODE == 1
            noc_async_read_one_packet(((uint64_t)coord[k] << NOC_ADDR_COORD_SHIFT) | lo[k], l1_write_addr, page_bytes);
            lo[k] += page_bytes;
            k = (k + 1 == NB) ? 0 : k + 1;
#else
            while (!noc_cmd_buf_ready(noc_index, read_cmd_buf));
            NOC_CMD_BUF_WRITE_REG(noc_index, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, coord[k]);
            ncrisc_noc_read_with_state<noc_mode, true, true /*one_packet*/, true /*skip_cmdbuf_chk*/>(
                noc_index, read_cmd_buf, lo[k], l1_write_addr);
            lo[k] += page_bytes;
            k = (k + 1 == NB) ? 0 : k + 1;
#endif
            l1_write_addr += page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, block_tiles);
    }
}
