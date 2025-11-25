// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t tile_height = 32;
    
    uint32_t arg_idx = 0; 
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_width_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t offset_within_stick = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t stick_size = get_compile_time_arg_val(0);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto s = TensorAccessor(dst_args, dst_addr, stick_size);
    constexpr auto num_ct_args = 2;

    uint64_t base_dst_noc_addr[tile_height];

    auto write_tiles = [&](const uint32_t& num_tiles, const uint32_t& width_size, const uint32_t& stride_size) {
        cb_wait_front(cb_id_out0, num_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t k = 0; k < tile_height; k++) {
            uint64_t dst_noc_addr = base_dst_noc_addr[k];
            noc_async_write(l1_read_addr, dst_noc_addr, width_size);
            l1_read_addr += width_size;
            base_dst_noc_addr[k] += width_size + stride_size;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles);
    };

    uint32_t stick_id = start_stick_id;

    uint32_t curr_offset = offset_within_stick;
    for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
        for (uint32_t tile_id = 0; tile_id < num_tiles_per_core; tile_id++) {
            for (uint32_t j = 0; j < tile_height; j++) {
                base_dst_noc_addr[j] = get_noc_addr(j + stick_id, s, curr_offset);
            }
            write_tiles(1, tile_width_size, stick_size - curr_offset - tile_width_size);
            curr_offset += tile_width_size;
        }

        stick_id += tile_height;
    }

#ifdef FUSED_OP_SYNC    
    constexpr uint32_t noc_start_x = get_compile_time_arg_val(num_ct_args);
    constexpr uint32_t noc_start_y = get_compile_time_arg_val(num_ct_args + 1);
    constexpr uint32_t noc_end_x = get_compile_time_arg_val(num_ct_args + 2);
    constexpr uint32_t noc_end_y = get_compile_time_arg_val(num_ct_args + 3);
    constexpr uint32_t num_dests = get_compile_time_arg_val(num_ct_args + 4);

    auto sync_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    auto sync_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_semaphore_addr);

    uint64_t mcast_sync_semaphore_noc_addr =
        get_noc_multicast_addr(noc_start_x, noc_start_y, noc_end_x, noc_end_y, sync_semaphore_addr);

    noc_semaphore_wait(sync_semaphore_ptr, start_stick_id);
    noc_semaphore_set(sync_semaphore_ptr, start_stick_id+num_sticks);
    
    noc_semaphore_set_multicast(sync_semaphore_addr,mcast_sync_semaphore_noc_addr,num_dests);
#endif
}
