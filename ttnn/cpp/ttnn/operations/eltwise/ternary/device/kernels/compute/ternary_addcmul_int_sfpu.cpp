// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/add_int_sfpu.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    CircularBuffer cb_in0(tt::CBIndex::c_0);  // input_a
    CircularBuffer cb_in1(tt::CBIndex::c_1);  // input_b
    CircularBuffer cb_in2(tt::CBIndex::c_2);  // input_c
    CircularBuffer cb_out(tt::CBIndex::c_3);

    unary_op_init_common(cb_in0.get_cb_id(), cb_out.get_cb_id());

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_in0.wait_front(num_tiles_per_cycle);
        cb_in1.wait_front(num_tiles_per_cycle);
        cb_in2.wait_front(num_tiles_per_cycle);

        cb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        copy_tile_init(cb_in0.get_cb_id());
        copy_tile(cb_in0.get_cb_id(), 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        copy_tile_init(cb_in1.get_cb_id());
        copy_tile(cb_in1.get_cb_id(), 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        copy_tile_init(cb_in2.get_cb_id());
        copy_tile(cb_in2.get_cb_id(), 0 /*in_tile_index*/, 2 /*dst_tile_index*/);

        fill_tile_init();
        fill_tile_int<ADDCMUL_DATA_FORMAT>(3, scalar_arg);

        mul_int_tile_init<ADDCMUL_DATA_FORMAT>();
        mul_int_tile<ADDCMUL_DATA_FORMAT>(3, 1, 3);
        mul_int_tile<ADDCMUL_DATA_FORMAT>(3, 2, 2);

        add_int_tile_init();
        add_int_tile<ADDCMUL_DATA_FORMAT>(0, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out.get_cb_id());

        tile_regs_release();

        cb_out.push_back(num_tiles_per_cycle);
        cb_in0.pop_front(num_tiles_per_cycle);
        cb_in1.pop_front(num_tiles_per_cycle);
        cb_in2.pop_front(num_tiles_per_cycle);
    }
}
