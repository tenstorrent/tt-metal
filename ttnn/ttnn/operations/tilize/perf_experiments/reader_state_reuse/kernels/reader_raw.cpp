// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// reader_state_reuse bench — variant 2/5: RAW (helper's own loop, split zones).
//
// Identical arithmetic to `dataflow_kernel_lib::read_sticks_for_tilize`'s
// TILE-mode loop (tilize_helpers_dataflow.inl:107-135) — same per-tile-row
// (32-stick) reserve/read/barrier/push chunking, same
// `accessor.get_noc_addr(page, offset)` call (division-based bank resolve,
// see reader_recurrence.cpp), same `noc_async_read` — but written out here
// with reserve / issue / barrier as THREE separate zones per chunk instead of
// the helper's one fused occupancy zone. Isolates "helper-call overhead" from
// the real issue/barrier split: if this variant's total is ~equal to
// reader_helper.cpp's, the helper itself costs nothing extra and every
// subsequent variant's delta is attributable to the address/state mechanism,
// not to bypassing the helper.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t kTileH = 32;
}

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t bw = get_compile_time_arg_val(2);  // tile-pages to reserve/push per chunk
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
                noc_async_read(noc_addr, addr, row_bytes);
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
