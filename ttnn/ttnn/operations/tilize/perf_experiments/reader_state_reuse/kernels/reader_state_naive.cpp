// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// reader_state_reuse bench — variant 4/5: STATE_NAIVE (expected NULL).
//
// Tests the "just call set_state/with_state per stick" reading of the idea
// LITERALLY, with no bank grouping: `noc_async_read_one_packet_set_state()`
// (writes NOC_TARG_ADDR_COORDINATE + NOC_AT_LEN_BE) immediately followed by
// `noc_async_read_one_packet_with_state()` (writes NOC_TARG_ADDR_LO +
// NOC_RET_ADDR_LO + NOC_CMD_CTRL) for EVERY stick, address still resolved
// through the accessor's per-page division. Same per-tile-row (32-stick)
// reserve/read/barrier/push chunking as every other variant here.
//
// This is not expected to win and it is measured, not asserted: on a 12-bank
// (non-pow2) box the DRAM target's NoC (x,y) — the coordinate `set_state`
// writes — changes on nearly every one of the 32 consecutive stick reads
// (bank = page % 12 cycles almost every call), so `with_state` cannot skip
// the coordinate rewrite `set_state` just did. Total register pokes per stick
// (COORD+LEN, then LO+LO+CTRL = 5 writes, 2 cmd-buf-ready polls) is the SAME
// as one plain `noc_async_read_one_packet` (COORD+LO+LO+LEN+CTRL = 5 writes,
// 1 poll) plus a second poll — i.e. this variant can only be flat or slightly
// SLOWER than the raw/recurrence variants, never faster. The real state-reuse
// opportunity requires grouping same-bank reads together first
// (`reader_state_grouped.cpp`).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t kTileH = 32;
}

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t bw = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();
    constexpr uint32_t num_chunks = rows / kTileH;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t byte_offset = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t chunk_start_page = start_page + chunk * kTileH;
        uint32_t l1_addr;
        {
            MaybeDeviceZoneScope("reader_reserve");
            cb_reserve_back(cb_in, bw);
            l1_addr = get_write_ptr(cb_in);
        }
        {
            MaybeDeviceZoneScope("reader_issue");
            uint32_t addr = l1_addr;
            for (uint32_t row = 0; row < kTileH; ++row) {
                const uint64_t noc_addr = in_acc.get_noc_addr(chunk_start_page + row, byte_offset);
                noc_async_read_one_packet_set_state(noc_addr, row_bytes);
                noc_async_read_one_packet_with_state(static_cast<uint32_t>(noc_addr), addr);
                addr += row_bytes;
            }
        }
        {
            MaybeDeviceZoneScope("reader_barrier");
            noc_async_read_barrier();
        }
        cb_push_back(cb_in, bw);
    }
}
