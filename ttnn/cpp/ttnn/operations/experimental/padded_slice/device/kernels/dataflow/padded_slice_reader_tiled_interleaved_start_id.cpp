// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(0);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_dims = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles_per_core = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles_per_barrier = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles_per_row_this_core = get_arg_val<uint32_t>(5);
    const uint32_t num_unpadded_sticks_addr = get_arg_addr(6);

    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(num_unpadded_sticks_addr);
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;

    constexpr uint32_t cb_id_in0 = 0;

    uint32_t src_stick_id = start_id;
    uint32_t tiles_read = 0;
    const uint32_t tile_size = get_tile_size(cb_id_in0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto s0 = TensorAccessor(src_args, src_addr, tile_size);
    const uint32_t extra_tiles_per_row = num_tiles_per_row - num_tiles_per_row_this_core;

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
    DPRINT << "num_tiles_per_row_this_core: " << num_tiles_per_row_this_core
           << " extra_tiles_per_row: " << extra_tiles_per_row << ENDL();
#endif
    const uint32_t base_src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    const uint64_t base_noc_addr = get_noc_addr(0, s0);
    uint32_t num_tiles_pushed = 0;
    while (tiles_read < num_tiles_per_core) {
        cb_reserve_back(cb_id_in0, num_tiles_per_barrier);
        uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);
#ifdef DEBUG
        DPRINT << "Src Buffer L1 Addr: " << src_buffer_l1_addr << ENDL();
        DPRINT << "Tiles read " << tiles_read << ", Num tiles pushed: " << num_tiles_pushed << ENDL();
#endif
        for (uint32_t i = 0; i < num_tiles_per_barrier and tiles_read < num_tiles_per_core; ++i) {
            tiles_read++;
            if (id_per_dim[0] >= (num_unpadded_sticks[0] - extra_tiles_per_row)) {
#ifdef DEBUG
                DPRINT << "Skipping read for src_stick_id: " << src_stick_id << ", id_per_dim: " << id_per_dim[0] << ","
                       << id_per_dim[1] << "," << id_per_dim[2] << "," << id_per_dim[3]
                       << ", tiles_read: " << tiles_read << ENDL();
#endif
                src_buffer_l1_addr += tile_size;
                src_stick_id++;

            } else {
                uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0);
                noc_async_read(src_noc_addr, src_buffer_l1_addr, tile_size);
#ifdef DEBUG
                DPRINT << "src_stick_id: " << src_stick_id << ", src_buffer_l1_addr: " << src_buffer_l1_addr
                       << ", tiles_read: " << tiles_read << "id " << id_per_dim[0] << "," << id_per_dim[1] << ","
                       << id_per_dim[2] << "," << id_per_dim[3] << ENDL();
#endif
                src_buffer_l1_addr += tile_size;
                src_stick_id++;
            }

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
        num_tiles_pushed += num_tiles_per_barrier;
    }
}
