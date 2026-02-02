// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tile_id_stride = get_compile_time_arg_val(1);
    constexpr uint32_t num_dims = get_compile_time_arg_val(2);
    constexpr uint32_t size_tile = get_compile_time_arg_val(3);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_tile_id_start = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    tt_l1_ptr uint32_t* shape_tiles = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    tt_l1_ptr uint32_t* tile_coord = shape_tiles + num_dims;

    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto s = TensorAccessor(out_args, out_addr, size_tile);

    uint32_t out_tile_id = out_tile_id_start;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write_page(out_tile_id, s, l1_read_addr);
        noc_async_writes_flushed();
        cb_pop_front(cb_id, 1);

        out_tile_id += tile_id_stride;
        tile_coord[num_dims - 1] += tile_id_stride;

        // for (int32_t j = num_dims - 1; j >= 1; j--) {
        //     while (tile_coord[j] >= shape_tiles[j]) {
        //         tile_coord[j] -= shape_tiles[j];
        //         tile_coord[j - 1]++;
        //     }
        // }
    }
}
