#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/dataflow/dataflow_buffer.h"

#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t first_tile_offset = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    binary_op_init_common(0, 1, 16);

    uint32_t cb_a = 0;
    uint32_t cb_b = 1;
    uint32_t cb_c = 2;
    uint32_t cb_out = 16;
    uint32_t cb_zero = 8;

    cb_wait_front(cb_a, 1);
    cb_reserve_back(cb_zero, 1);
    binary_tiles_init<false, EltwiseBinaryType::ELWSUB>(cb_a, cb_a);
    tile_regs_acquire();
    sub_tiles(cb_a, cb_a, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_zero);
    tile_regs_release();
    cb_push_back(cb_zero, 1);
    cb_wait_front(cb_zero, 1);

    for (uint32_t t = first_tile_offset; t < first_tile_offset + num_tiles; t++) {
        if (t > first_tile_offset) {
            cb_wait_front(cb_a, 1);
        }
        cb_wait_front(cb_b, 1);
        cb_wait_front(cb_c, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        binary_tiles_init<false, EltwiseBinaryType::ELWMUL>(cb_a, cb_b);
        mul_tiles(cb_a, cb_b, 0, 0, 0);
        binary_tiles_init<false, EltwiseBinaryType::ELWADD>(cb_c, cb_zero, true);
        add_tiles(cb_c, cb_zero, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        cb_pop_front(cb_c, 1);
        cb_push_back(cb_out, 1);
    }
}
