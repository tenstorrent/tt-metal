// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t in_tile_offset_by_head = get_arg_val<uint32_t>(0);
    uint32_t q_start_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_q_out = get_compile_time_arg_val(2);
    constexpr uint32_t head_size = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t head_size_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t PHASES_TO_READ =
        get_compile_time_arg_val(6);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

    constexpr uint32_t in_num_cores = get_compile_time_arg_val(7);

    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + in_num_cores));

    // Q
    uint32_t cur_core_idx = 0;
    uint32_t total_input_cores = in_num_cores;
    uint32_t num_tiles_per_core = (head_size_num_tiles * batch) / total_input_cores;

    uint64_t qkv_read_addr = get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                             in_tile_offset_by_head;
    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    uint32_t tile_size = head_size / head_size_num_tiles;
    const uint32_t cb_write_ptr_base = get_write_ptr(cb_id_q_out);

    for (uint32_t q = 0; q < batch; ++q) {
        uint32_t wptr_offset = q < 16 ? q * SUBTILE_LINE_BYTES : (q - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;
        uint32_t q_write_addr = cb_write_ptr_base + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc_async_read(qkv_read_addr, q_write_addr, SUBTILE_LINE_BYTES);
            }
            // Read second phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc_async_read(
                    qkv_read_addr + 256 * ELEMENT_SIZE, q_write_addr + 256 * ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            }

            qkv_read_addr += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                cur_core_idx++;
                qkv_read_addr =
                    get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                    in_tile_offset_by_head;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    noc_async_read_barrier();
}
