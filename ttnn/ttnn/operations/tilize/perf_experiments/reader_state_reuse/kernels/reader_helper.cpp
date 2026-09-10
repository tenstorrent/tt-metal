// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// reader_state_reuse bench — variant 1/5: HELPER (the honest baseline).
//
// Byte-for-byte the op's real `reader_read_block` branch
// (ttnn/ttnn/operations/tilize/kernels/tilize_reader.cpp): one
// `dataflow_kernel_lib::read_sticks_for_tilize<cb_in, TILE>` call over this
// core's one block. The zone wraps the WHOLE call because the helper owns
// reserve/issue/barrier/push internally (dataflow_kernel_lib::read_sticks_for_
// tilize, tilize_helpers_dataflow.inl:121-135) — per
// .claude/references/device-zone-scope-attribution.md §4 this number is the
// stage's OCCUPANCY, not its issue cost alone. `reader_raw.cpp` is the same
// algorithm with reserve/issue/barrier split into three zones so the two can
// be compared (helper-call overhead vs the real issue/barrier split).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t rows = get_compile_time_arg_val(0);       // stick reads in this block
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);  // bytes per stick
    constexpr auto in_args = TensorAccessorArgs<3>();            // arg 2 is `bw`, unused here

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t byte_offset = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr);

    {
        MaybeDeviceZoneScope("reader_read_block");
        dataflow_kernel_lib::read_sticks_for_tilize<cb_in, dataflow_kernel_lib::TilizeGranularity::TILE>(
            in_acc,
            /* total_num_rows          */ rows,
            /* row_bytes               */ row_bytes,
            /* start_page              */ start_page,
            /* byte_offset_within_page */ byte_offset);
    }
}
