// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads work units [unit_begin, unit_end) of the input into the stage DFB, up to units_per_batch per barrier.
// Unit u is chunk (u % chunks_per_page) of page (u / chunks_per_page).
//
// Compile-time args: page_bytes, chunk_bytes, chunks_per_page, entry_bytes, units_per_batch
// Runtime args:      unit_begin, unit_end
// Bindings:          dfb::stage (producer), tensor::src

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t chunks_per_page = get_arg(args::chunks_per_page);
    constexpr uint32_t entry_bytes = get_arg(args::entry_bytes);
    constexpr uint32_t units_per_batch = get_arg(args::units_per_batch);
    const uint32_t unit_begin = get_arg(args::unit_begin);
    const uint32_t unit_end = get_arg(args::unit_end);

    DataflowBuffer stage(dfb::stage);
    Noc noc;
    const auto src = TensorAccessor(tensor::src);

    uint32_t page = unit_begin / chunks_per_page;
    uint32_t chunk = unit_begin % chunks_per_page;
    for (uint32_t unit = unit_begin; unit < unit_end;) {
        const uint32_t batch = (unit_end - unit) < units_per_batch ? (unit_end - unit) : units_per_batch;
        stage.reserve_back(batch);
        for (uint32_t k = 0; k < batch; ++k) {
            const uint32_t offset = chunk * chunk_bytes;
            const uint32_t bytes = (page_bytes - offset) < chunk_bytes ? (page_bytes - offset) : chunk_bytes;
            noc.async_read(
                src, stage, bytes, {.page_id = page, .offset_bytes = offset}, {.offset_bytes = k * entry_bytes});
            if (++chunk == chunks_per_page) {
                chunk = 0;
                ++page;
            }
        }
        noc.async_read_barrier();
        stage.push_back(batch);
        unit += batch;
    }
}
