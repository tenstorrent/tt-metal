// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gather_strip bench writer — the production tilize_writer.cpp P_LOCAL_SHARD
// branch, verbatim in structure: cb_output_tiles is ALIASED on this core's
// resident TILE shard, so compute already packed straight into the output
// tensor. The writer issues NO NoC write; it only DRAINS the CB so the CB keeps
// exactly one consumer.
//
// It drains at the SAME granularity compute produces, which is what makes the
// strip arm a drop-in: `drain_wt` is WT_CHUNK in the row arm and PAGE_TILES in
// the strip arms, with the block count scaled to match — the tile SEQUENCE that
// lands in the shard is identical either way.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = 16;
    constexpr uint32_t drain_wt = get_compile_time_arg_val(0);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    for (uint32_t i = 0; i < num_blocks; ++i) {
        {
            MaybeDeviceZoneScope("writer_wait");
            cb_wait_front(cb_output_tiles, drain_wt);
        }
        cb_pop_front(cb_output_tiles, drain_wt);
    }
}
