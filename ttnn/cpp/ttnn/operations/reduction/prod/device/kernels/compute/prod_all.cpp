// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const tt::CBIndex input_cb = tt::CBIndex::c_0;
    const tt::CBIndex final_output_cb = tt::CBIndex::c_3;

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer final_output_cb_obj(final_output_cb);

    const int one_tile = 1;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(input_cb, input_cb, final_output_cb);
    pack_reconfig_data_format(final_output_cb);

    final_output_cb_obj.reserve_back(one_tile);

    // The running product lives in DEST for the whole reduction.
    tile_regs_acquire();

    // Seed DEST with the first input tile.
    input_cb_obj.wait_front(one_tile);
    copy_tile_to_dst_init_short(input_cb);
    copy_tile(input_cb, 0, 0);
    input_cb_obj.pop_front(one_tile);

    // Fold each remaining tile in: DEST = DEST * next_tile.
    // DEST_TO_SRCA loads the running product from DEST into SRCA.
    mul_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(input_cb, input_cb);
    for (uint32_t t = 1; t < num_tiles; t++) {
        input_cb_obj.wait_front(one_tile);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(input_cb, 0, 0);
        input_cb_obj.pop_front(one_tile);
    }

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, final_output_cb);
    final_output_cb_obj.push_back(one_tile);
    tile_regs_release();
}
