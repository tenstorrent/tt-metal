// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_3;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    CircularBuffer cb_in0_obj(cb_in0);
    CircularBuffer cb_out0_obj(cb_out0);

    binary_op_init_common(cb_in0, cb_in0, cb_out0);
    pack_reconfig_data_format(cb_out0);

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        // Each output tile is an independent reduction of num_input_tiles.
        // The running product lives in DEST.
        tile_regs_acquire();

        // Seed DEST with the first input tile of this reduction.
        cb_in0_obj.wait_front(onetile);
        copy_init(cb_in0);
        copy_tile(cb_in0, 0, dst0);
        cb_in0_obj.pop_front(onetile);

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in0);
        for (uint32_t j = 1; j < num_input_tiles; ++j) {
            cb_in0_obj.wait_front(onetile);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_in0, 0, dst0);
            cb_in0_obj.pop_front(onetile);
        }

        tile_regs_commit();

        cb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, cb_out0);
        cb_out0_obj.push_back(onetile);
        tile_regs_release();
    }
}
