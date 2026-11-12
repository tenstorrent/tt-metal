// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device valid query counts (MSDA_DEVICE_COUNTS): a data-movement kernel reads
// the (B,) uint32 counts into its own single-page CB, used as plain L1 scratch
// and never pushed, and derives its core's tile run from them with
// msda_work_list.hpp. The reader and the writer each stage their own copy, so
// they agree on the run without signalling each other: nothing writes the
// counts tensor while the program runs.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_work_list.hpp"

namespace fused_msda_work {

template <uint32_t BATCH, uint32_t Q>
struct StagedCounts {
    uint32_t l1 = 0;

    // The counts come from another op's output; one above Q would read and
    // write rows past the tensors, so it is clamped rather than trusted.
    uint32_t operator()(uint32_t b) const {
        const uint32_t n = CoreLocalMem<volatile uint32_t>(l1)[b];
        return n < Q ? n : Q;
    }
};

template <uint32_t BATCH, uint32_t Q, uint32_t NUM_HEADS>
struct DeviceWorkList {
    StagedCounts<BATCH, Q> valid;
    Run run{0, 0};

    Unit unit(uint32_t t) const { return fused_msda_work::unit(valid, BATCH, NUM_HEADS, run.first + t); }
};

template <uint32_t BATCH, uint32_t Q, uint32_t NUM_HEADS, typename CountsAccessor>
DeviceWorkList<BATCH, Q, NUM_HEADS> stage_work_list(
    Noc& noc, const CountsAccessor& counts_acc, uint32_t cb_index, uint32_t core, uint32_t num_cores) {
    CircularBuffer counts_cb(cb_index);
    counts_cb.reserve_back(1);
    DeviceWorkList<BATCH, Q, NUM_HEADS> work;
    work.valid.l1 = counts_cb.get_write_ptr();
    noc.async_read(
        counts_acc,
        CoreLocalMem<uint32_t>(work.valid.l1),
        BATCH * sizeof(uint32_t),
        {.page_id = 0},
        {.offset_bytes = 0});
    noc.async_read_barrier();
    // The NoC write bypasses the RISC data cache; drop any line it holds for
    // these addresses before reading the counts.
    invalidate_l1_cache();
    work.run = core_run(active_tiles(work.valid, BATCH, NUM_HEADS), core, num_cores);
    return work;
}

}  // namespace fused_msda_work
