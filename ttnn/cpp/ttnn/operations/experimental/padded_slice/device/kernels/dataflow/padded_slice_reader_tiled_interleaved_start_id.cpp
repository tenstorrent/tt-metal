// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_dims = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles_per_core = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles_per_barrier = get_arg_val<uint32_t>(4);
    const uint32_t num_unpadded_sticks_addr = get_arg_addr(5);

    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(num_unpadded_sticks_addr);
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;

    uint32_t src_stick_id = start_id;
    uint32_t tiles_read = 0;
    const uint32_t tile_size = get_tile_size(cb_id_in0);
    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = tile_size};

#ifdef DEBUG
    DPRINT << "src_addr: " << src_addr << ", num_dims: " << num_dims << ", start_id: " << start_id
           << ", num_tiles_per_core: " << num_tiles_per_core << ", num_tiles_per_barrier: " << num_tiles_per_barrier
           << ENDL();

    DPRINT << "tile_size: " << tile_size << ", src_stick_id: " << src_stick_id << ", tiles_read: " << tiles_read
           << ENDL();

    DPRINT << "num_unpadded_sticks: " << num_unpadded_sticks[0] << " " << num_unpadded_sticks[1] << " "
           << num_unpadded_sticks[2] << " " << num_unpadded_sticks[3] << " " << ENDL();
    DPRINT << "num_padded_sticks: " << num_padded_sticks[0] << " " << num_padded_sticks[1] << " "
           << num_padded_sticks[2] << " " << num_padded_sticks[3] << " " << ENDL();
#endif
    const uint32_t base_src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    const uint64_t base_noc_addr = get_noc_addr(0, s0);

    while (tiles_read < num_tiles_per_core) {
        cb_reserve_back(cb_id_in0, num_tiles_per_barrier);
        uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);

        for (uint32_t i = 0; i < num_tiles_per_barrier and tiles_read < num_tiles_per_core; ++i) {
            tiles_read++;
            uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0);
            noc_async_read(src_noc_addr, src_buffer_l1_addr, tile_size);
            src_buffer_l1_addr += tile_size;
            src_stick_id++;
#ifdef DEBUG
            DPRINT << "src_stick_id: " << src_stick_id << ", src_noc_addr: " << src_noc_addr
                   << ", src_buffer_l1_addr: " << src_buffer_l1_addr << ", tiles_read: " << tiles_read << ENDL();
#endif
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_tiles_per_barrier);
    }
}
