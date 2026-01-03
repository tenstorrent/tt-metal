// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint_tile.h"

void kernel_main() {
    // CB indices
    // ----------
    constexpr uint32_t src_cb_idx = tt::CBIndex::c_0;

    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_page_offset = get_arg_val<uint32_t>(1);

    // Compile time args
    // -----------------
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t src_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_group = get_compile_time_arg_val(3);
    constexpr uint32_t pages_per_batch = get_compile_time_arg_val(4);
    constexpr uint32_t num_batches = get_compile_time_arg_val(5);

    // Read this 3 times for mean, var, and output
    constexpr uint32_t num_reads = 3;

    //-------------------------------------------------------------------------
    const auto s_src = get_interleaved_addr_gen<src_is_dram, src_page_size>(src_base_addr);

    for (uint32_t i = 0; i < num_reads; ++i) {
        for (uint32_t batch = 0; batch < num_batches; ++batch) {
            const uint32_t batch_offset = batch * pages_per_batch;

            for (uint32_t page = 0; page < pages_per_group; ++page) {
                const uint64_t src_noc_addr = s_src.get_noc_addr(src_page_offset + batch_offset + page);

                cb_reserve_back(src_cb_idx, tiles_per_page);
                const uint32_t src_cb_addr = get_write_ptr(src_cb_idx);
                noc_async_read(src_noc_addr, src_cb_addr, src_page_size);
                noc_async_read_barrier();
                cb_push_back(src_cb_idx, tiles_per_page);
            }
        }
    }
}
