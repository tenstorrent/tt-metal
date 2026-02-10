// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tile_id_stride = get_compile_time_arg_val(1);
    constexpr uint32_t num_dims = get_compile_time_arg_val(2);
    constexpr uint32_t size_tile = get_compile_time_arg_val(3);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_tile_id_start = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    tt_l1_ptr uint32_t* shape_tiles = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    tt_l1_ptr uint32_t* tile_coord = shape_tiles + num_dims;
    tt_l1_ptr uint32_t* tile_id_acc = tile_coord + num_dims;
    tt_l1_ptr uint32_t* coord_inc = tile_id_acc + num_dims;

    constexpr auto src_args = TensorAccessorArgs<4>();
    const auto s = TensorAccessor(src_args, src_addr, size_tile);

    uint32_t src_tile_id = src_tile_id_start;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(src_tile_id, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);

        for (int32_t j = num_dims - 1; j >= 1; j--) {
            tile_coord[j] += coord_inc[j];
            src_tile_id += coord_inc[j] * tile_id_acc[j];
            if (tile_coord[j] >= shape_tiles[j]) {
                tile_coord[j] -= shape_tiles[j];
                tile_coord[j - 1] += 1;
                src_tile_id += tile_id_acc[j - 1] - shape_tiles[j] * tile_id_acc[j];
            }
        }
    }
}
