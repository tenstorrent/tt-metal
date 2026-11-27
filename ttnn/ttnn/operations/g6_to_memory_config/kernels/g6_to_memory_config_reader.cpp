// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Staged reader of g6_to_memory_config (regime staged: DRAM -> DRAM, NCRISC / NoC0).
// A NoC transaction cannot go DRAM -> DRAM, so pages pass through cb_staged_pages on this core.
// Loads this core's contiguous page range in blocks of kBlockPages; the CB is reserved and pushed
// in nominal blocks, also for the ragged last one, so its ring stays block-aligned.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_memory_config_move.hpp"

void kernel_main() {
    constexpr uint32_t cb_staged_pages = get_compile_time_arg_val(0);
    constexpr uint32_t kBlockPages = get_compile_time_arg_val(1);
    constexpr uint32_t kPageSize = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t src_base = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const auto src = TensorAccessor(src_args, src_base, kPageSize);

    for (uint32_t done = 0; done < num_pages; done += kBlockPages) {
        const uint32_t block = num_pages - done < kBlockPages ? num_pages - done : kBlockPages;
        {
            DeviceZoneScopedN("G6-STAGED-RESERVE");
            cb_reserve_back(cb_staged_pages, kBlockPages);
        }
        {
            DeviceZoneScopedN("G6-STAGED-READ");
            g6_to_memory_config::move_span<true>(
                src, start_page + done, block, get_write_ptr(cb_staged_pages), kPageSize);
        }
        {
            DeviceZoneScopedN("G6-STAGED-READ-BARRIER");
            noc_async_read_barrier();
        }
        cb_push_back(cb_staged_pages, kBlockPages);
    }
}
