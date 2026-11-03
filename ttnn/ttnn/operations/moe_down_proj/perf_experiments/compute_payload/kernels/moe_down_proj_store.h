// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj store_wave, run by whichever dataflow RISC the host assigns (store_on_reader): per tile row m of
// this core's rows, one untilized block of core_n_tiles tiles -> 32 row segments of y, row 32 * m + i at byte offset
// line * seg_bytes; one flush per block before the pop. A call stores up to max_blocks of the wave's blocks starting
// at block `first` and returns the next unstored block, so a caller can pace one wave's store over several calls.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "moe_down_proj_plan.h"

template <uint32_t cb_output_rows, uint32_t core_n_tiles, typename Accessor>
uint32_t moe_store_wave(
    const MoePlan& plan,
    const Accessor& out_accessor,
    uint32_t wave,
    uint32_t group,
    uint32_t line,
    uint32_t first = 0,
    uint32_t max_blocks = UINT32_MAX) {
    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t seg_bytes = core_n_tiles * tile_rows * sizeof(uint16_t);
    const uint32_t seg_offset = line * seg_bytes;
    const uint32_t row_lo = plan.group_first(wave, group);
    const uint32_t my_rows = plan.group_rows(wave, group);
    const uint32_t last = my_rows - first < max_blocks ? my_rows : first + max_blocks;
    if (last <= first) {
        return last;
    }
    MaybeDeviceZoneScope("y_store");
    for (uint32_t r = first; r < last; ++r) {
        cb_wait_front(cb_output_rows, core_n_tiles);
        const uint32_t block_l1 = get_read_ptr(cb_output_rows);
        const uint32_t first_row = (row_lo + r) * tile_rows;
        for (uint32_t i = 0; i < tile_rows; ++i) {
#ifndef ABL_NO_STORE
            noc_async_write(block_l1 + i * seg_bytes, out_accessor.get_noc_addr(first_row + i, seg_offset), seg_bytes);
#endif
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_rows, core_n_tiles);
    }
    return last;
}
