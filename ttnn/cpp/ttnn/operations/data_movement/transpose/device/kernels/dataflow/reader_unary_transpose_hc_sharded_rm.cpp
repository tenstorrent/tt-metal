// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
#ifdef USE_SPECIAL_CASE

    bool read_single_h_block_per_core = get_arg_val<uint32_t>(0) == 1;
    uint32_t num_C_blocks_per_core = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_shard_core = get_arg_val<uint32_t>(2);
    uint32_t num_cores_read = get_arg_val<uint32_t>(3);
    uint32_t read_stick_stride = get_arg_val<uint32_t>(4);
    tt_l1_ptr uint32_t* read_stick_offset = (tt_l1_ptr uint32_t*)(get_arg_addr(5));
    tt_l1_ptr uint32_t* noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(5 + num_cores_read));
    tt_l1_ptr uint32_t* noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(5 + num_cores_read * 2));

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(2);

    if (read_single_h_block_per_core) {
        uint32_t write_stick_stride = stick_size_bytes * num_cores_read;
        uint32_t l1_write_offset = 0;

        for (uint32_t core = 0; core < num_cores_read; ++core) {
            uint32_t l1_read_addr = get_read_ptr(cb_in0) + read_stick_offset[core];
            uint64_t noc_read_addr = get_noc_addr(noc_coord_x[core], noc_coord_y[core], l1_read_addr);
            uint32_t l1_write_addr = get_write_ptr(cb_out0) + l1_write_offset;

            noc_async_read_one_packet_set_state(noc_read_addr, stick_size_bytes);

            for (uint32_t i = 0; i < num_sticks_per_shard_core; ++i) {
                noc_async_read_one_packet_with_state(noc_read_addr, l1_write_addr);
                noc_read_addr += read_stick_stride;
                l1_write_addr += write_stick_stride;
            }
            l1_write_offset += stick_size_bytes;
            noc_async_read_barrier();
        }
    } else {
        uint32_t l1_write_addr = get_write_ptr(cb_out0);
        uint32_t l1_read_addr = get_read_ptr(cb_in0);

        for (uint32_t c = 0; c < num_C_blocks_per_core; ++c) {
            for (uint32_t core = 0; core < num_cores_read; ++core) {
                uint64_t noc_read_addr =
                    get_noc_addr(noc_coord_x[core], noc_coord_y[core], l1_read_addr + read_stick_offset[core]);

                noc_async_read_one_packet_set_state(noc_read_addr, stick_size_bytes);

                for (uint32_t i = 0; i < num_sticks_per_shard_core; ++i) {
                    noc_async_read_one_packet_with_state(noc_read_addr, l1_write_addr);
                    noc_read_addr += read_stick_stride;
                    l1_write_addr += stick_size_bytes;
                }
            }

            l1_read_addr += stick_size_bytes;
        }
        noc_async_read_barrier();
    }

#else

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t C = get_compile_time_arg_val(4);
    constexpr uint32_t W_size_bytes = get_compile_time_arg_val(5);
    constexpr bool row_major = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t num_cores_x = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(8);

    uint32_t arg_idx = 0;
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t curr_c = get_arg_val<uint32_t>(arg_idx++);
    uint32_t curr_h = get_arg_val<uint32_t>(arg_idx++);
    uint32_t curr_n = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t* const shard_grid_x_map = reinterpret_cast<const uint32_t* const>(get_arg_addr(arg_idx));
    arg_idx += num_cores_x;
    const uint32_t* const shard_grid_y_map = reinterpret_cast<const uint32_t* const>(get_arg_addr(arg_idx));
    arg_idx += num_cores_y;

    constexpr uint32_t CH = C * H;

    const uint32_t stick_size_bytes = W_size_bytes;

    cb_reserve_back(cb_out0, num_sticks_per_core);
    uint32_t l1_write_addr = get_write_ptr(cb_out0);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core; ++iter) {
        uint32_t shard_id = i_stick / num_sticks_per_core;
        uint32_t stick_id_in_shard = i_stick - (shard_id * num_sticks_per_core);

        uint32_t shard_grid_inner_dim;
        if constexpr (row_major) {
            shard_grid_inner_dim = num_cores_x;
        } else {
            shard_grid_inner_dim = num_cores_y;
        }
        uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
        uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

        uint32_t worker_x_physical, worker_y_physical;
        if constexpr (row_major) {
            worker_x_physical = shard_grid_x_map[shard_grid_inner_dim_id];
            worker_y_physical = shard_grid_y_map[shard_grid_outer_dim_id];
        } else {
            worker_x_physical = shard_grid_x_map[shard_grid_outer_dim_id];
            worker_y_physical = shard_grid_y_map[shard_grid_inner_dim_id];
        }

        uint32_t l1_read_addr = get_read_ptr(cb_in0) + stick_id_in_shard * stick_size_bytes;

        uint64_t read_noc_addr = get_noc_addr(worker_x_physical, worker_y_physical, l1_read_addr);
        noc_async_read(read_noc_addr, l1_write_addr, stick_size_bytes);
        l1_write_addr += stick_size_bytes;

        curr_c++;
        i_stick += H;
        if (curr_c == C) {  // end of channel dim
            curr_h++;
            curr_c = 0;
            if (curr_h == H) {  // end of H dim
                curr_n++;
                curr_c = 0;
                curr_h = 0;
                i_stick = i_stick - H + 1;
            } else {
                i_stick = i_stick - CH + 1;
            }
        }
    }

    noc_async_read_barrier();
    cb_push_back(cb_out0, num_sticks_per_core);

#endif
}
