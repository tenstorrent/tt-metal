// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Staged writer of g6_to_memory_config (regime staged: DRAM -> DRAM, BRISC / NoC1).
// Drains cb_staged_pages to the target placement block by block, mirroring the reader's nominal
// block accounting, and ends with one write barrier so no write is in flight at return.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_memory_config_move.hpp"

void kernel_main() {
    constexpr uint32_t cb_staged_pages = get_compile_time_arg_val(0);
    constexpr uint32_t kBlockPages = get_compile_time_arg_val(1);
    constexpr uint32_t kPageSize = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t dst_base = get_common_arg_val<uint32_t>(1);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const auto dst = TensorAccessor(dst_args, dst_base, kPageSize);

    for (uint32_t done = 0; done < num_pages; done += kBlockPages) {
        const uint32_t block = num_pages - done < kBlockPages ? num_pages - done : kBlockPages;
        {
            DeviceZoneScopedN("G6-STAGED-WAIT");
            cb_wait_front(cb_staged_pages, kBlockPages);
        }
        {
            DeviceZoneScopedN("G6-STAGED-WRITE");
            g6_to_memory_config::move_span<false>(
                dst, start_page + done, block, get_read_ptr(cb_staged_pages), kPageSize);
            // The slot is reused by the reader once popped: its bytes must have left L1 first.
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_staged_pages, kBlockPages);
    }
    {
        DeviceZoneScopedN("G6-STAGED-DRAIN");
        noc_async_write_barrier();
    }
}
