// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // dfb::in = input DFB (c_0), dfb::out = final output DFB (c_3).
    DataflowBuffer input_dfb_obj(dfb::in);
    DataflowBuffer final_output_dfb_obj(dfb::out);

    const int one_tile = 1;
    constexpr auto num_tiles = get_arg(args::num_tiles);

    compute_kernel_hw_startup(dfb::in, dfb::in, dfb::out);
    pack_reconfig_data_format(dfb::out);

    final_output_dfb_obj.reserve_back(one_tile);

    // The running product lives in DEST for the whole reduction.
    tile_regs_acquire();

    // Seed DEST with the first input tile.
    input_dfb_obj.wait_front(one_tile);
    copy_init(dfb::in);
    copy_tile(dfb::in, 0, 0);
    input_dfb_obj.pop_front(one_tile);

    // Fold each remaining tile in: DEST = DEST * next_tile.
    // DEST_TO_SRCA loads the running product from DEST into SRCA.
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::in);
    for (std::uint32_t t = 1; t < num_tiles; t++) {
        input_dfb_obj.wait_front(one_tile);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::in, 0, 0);
        input_dfb_obj.pop_front(one_tile);
    }

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, dfb::out);
    final_output_dfb_obj.push_back(one_tile);
    tile_regs_release();
}
