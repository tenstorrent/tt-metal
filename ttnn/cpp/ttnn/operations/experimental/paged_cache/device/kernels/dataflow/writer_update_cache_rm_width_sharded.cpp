// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_batch_idx = get_arg_val<uint32_t>(2);
    uint32_t update_idx = get_arg_val<uint32_t>(3);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr bool use_index_tensor = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_cache_cores = get_compile_time_arg_val(3);
    constexpr uint32_t shard_width_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cache_num_rows = get_compile_time_arg_val(5);
    constexpr uint32_t index_stick_size_B = get_compile_time_arg_val(6);
    constexpr auto index_tensor_args = TensorAccessorArgs<7>();

    CircularBuffer cb_input(input_cb_id);
    cb_input.push_back(1);
    const uint32_t src_base = cb_input.get_read_ptr();

    bool skip_update = false;
    if constexpr (use_index_tensor) {
        CircularBuffer cb_index(cb_index_id);
        const auto addrg = TensorAccessor(index_tensor_args, index_tensor_addr);
        cb_index.reserve_back(1);
        uint32_t index_cb_wr_ptr = cb_index.get_write_ptr();
        noc.async_read(addrg, CoreLocalMem<uint32_t>(index_cb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
        noc.async_read_barrier();
        cb_index.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
        update_idx = index_ptr[my_batch_idx];
        cb_index.pop_front(1);
        if (update_idx == (uint32_t)-1) {
            skip_update = true;
        }
    }

    if (!skip_update && update_idx < cache_num_rows) {
        const uint32_t row_offset = update_idx * shard_width_bytes;
        for (uint32_t c = 0; c < num_cache_cores; ++c) {
            const uint32_t noc_x = get_arg_val<uint32_t>(4 + 2 * c);
            const uint32_t noc_y = get_arg_val<uint32_t>(5 + 2 * c);
            const uint32_t dst = cache_addr + row_offset;
            noc_async_write(src_base + c * shard_width_bytes, get_noc_addr(noc_x, noc_y, dst), shard_width_bytes);
        }
        noc_async_write_barrier();
    }
}
