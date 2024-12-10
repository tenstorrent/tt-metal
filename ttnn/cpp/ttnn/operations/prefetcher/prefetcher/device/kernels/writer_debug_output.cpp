// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    DPRINT << "num_layers: " << num_layers << ENDL();
    DPRINT << "num_tensors: " << num_tensors << ENDL();
    DPRINT << "num_blocks: " << num_blocks << ENDL();

    constexpr uint32_t local_cb_id = 0;  // Writer cb TODO: set this to global cb id (remote cb id)
    constexpr uint32_t out_cb_id = 3;    // Writer output cb

    uint32_t rt_args_idx = 0;
    const uint32_t* page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            uint32_t curr_page_size = page_sizes[t];
            uint32_t curr_block_num_pages = block_num_pages[t];
            uint32_t curr_block_size_bytes = curr_block_num_pages * curr_page_size;

            // Copy from writer to writer output cb
            cb_wait_front(local_cb_id, curr_block_num_pages);
            auto l1_read_addr = get_read_ptr(local_cb_id);
            for (uint32_t block = 0; block < num_blocks; ++block) {
                cb_reserve_back(out_cb_id, curr_block_num_pages);
                auto l1_out_write_addr = get_noc_addr(get_write_ptr(out_cb_id));

                noc_async_write(l1_read_addr, l1_out_write_addr, curr_block_size_bytes);
                l1_read_addr += curr_block_size_bytes;
                cb_push_back(out_cb_id, curr_block_num_pages);
            }
            cb_pop_front(local_cb_id, num_blocks * curr_block_num_pages);
        }
    }
}
