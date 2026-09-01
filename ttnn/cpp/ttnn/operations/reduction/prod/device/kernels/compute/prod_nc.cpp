// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto num_output_tiles = get_arg(args::num_output_tiles);

    // dfb::in = input DFB (c_0), dfb::out = output DFB (c_3).
    constexpr std::uint32_t onetile = 1;
    constexpr std::uint32_t dst0 = 0;

    DataflowBuffer dfb_in0_obj(dfb::in);
    DataflowBuffer dfb_out0_obj(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::in, dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (std::uint32_t i = 0; i < num_output_tiles; ++i) {
        // Each output tile is an independent reduction of num_input_tiles.
        // The running product lives in DEST.
        tile_regs_acquire();

        // Seed DEST with the first input tile of this reduction.
        dfb_in0_obj.wait_front(onetile);
        copy_init(dfb::in);
        copy_tile(dfb::in, 0, dst0);
        dfb_in0_obj.pop_front(onetile);

        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::in);
        for (std::uint32_t j = 1; j < num_input_tiles; ++j) {
            dfb_in0_obj.wait_front(onetile);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::in, 0, dst0);
            dfb_in0_obj.pop_front(onetile);
        }

        tile_regs_commit();

        dfb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, dfb::out);
        dfb_out0_obj.push_back(onetile);
        tile_regs_release();
    }
}
