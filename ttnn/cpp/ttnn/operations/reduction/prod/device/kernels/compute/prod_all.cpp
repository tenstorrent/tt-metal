// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const tt::CBIndex input_dfb = tt::CBIndex::c_0;
    const tt::CBIndex final_output_dfb = tt::CBIndex::c_3;

    DataflowBuffer input_dfb_obj(input_dfb);
    DataflowBuffer final_output_dfb_obj(final_output_dfb);

    const int one_tile = 1;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    binary_op_init_common(input_dfb, input_dfb, final_output_dfb);
    pack_reconfig_data_format(final_output_dfb);

    final_output_dfb_obj.reserve_back(one_tile);

    // The running product lives in DEST for the whole reduction.
    tile_regs_acquire();

    // Seed DEST with the first input tile.
    input_dfb_obj.wait_front(one_tile);
    copy_tile_to_dst_init_short(input_dfb);
    copy_tile(input_dfb, 0, 0);
    input_dfb_obj.pop_front(one_tile);

    // Fold each remaining tile in: DEST = DEST * next_tile.
    // DEST_TO_SRCA loads the running product from DEST into SRCA.
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(input_dfb);
    for (uint32_t t = 1; t < num_tiles; t++) {
        input_dfb_obj.wait_front(one_tile);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(input_dfb, 0, 0);
        input_dfb_obj.pop_front(one_tile);
    }

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, final_output_dfb);
    final_output_dfb_obj.push_back(one_tile);
    tile_regs_release();
}
