// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(1);
    constexpr uint32_t tile_id_stride = get_compile_time_arg_val(2);
    constexpr uint32_t block_id_stride = get_compile_time_arg_val(3);
    constexpr uint32_t num_dims = get_compile_time_arg_val(4);
    constexpr uint32_t size_tile = get_compile_time_arg_val(5);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_block_id_start = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    tt_l1_ptr uint32_t* shape_blocks = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    tt_l1_ptr uint32_t* block_id_gap = shape_blocks + num_dims;
    tt_l1_ptr uint32_t* block_coord = block_id_gap + num_dims;

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(out_args, out_addr, size_tile);

    auto write_block = [&](uint32_t num_tiles, uint32_t start_id, uint32_t id_step, uint32_t size_tile) {
        cb_wait_front(cb_id, num_tiles);
        uint32_t l1_write_addr = get_read_ptr(cb_id);
        uint32_t id = start_id;
        for (uint32_t k = 0; k < num_tiles; k++) {
            uint64_t out_noc_addr = get_noc_addr(id, s);

            noc_async_write(l1_write_addr, out_noc_addr, size_tile);

            l1_write_addr += size_tile;
            id += id_step;
        }

        noc_async_writes_flushed();
        cb_pop_front(cb_id, num_tiles);
    };

    uint32_t out_block_id = out_block_id_start;

    for (uint32_t i = 0; i < num_blocks; i++) {
        write_block(num_tiles_per_block, out_block_id, tile_id_stride, size_tile);
        out_block_id += block_id_stride;

        for (uint32_t j = num_dims - 1; j >= 0; j--) {
            block_coord[j]++;
            if (block_coord[j] == shape_blocks[j]) {
                block_coord[j] = 0;
                out_block_id += block_id_gap[j];
            } else {
                break;
            }
        }
    }
}
