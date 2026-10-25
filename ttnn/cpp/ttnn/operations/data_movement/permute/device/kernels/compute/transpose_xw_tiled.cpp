// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/transpose.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#ifdef ARCH_QUASAR
#include "api/compute/pack.h"  // pack_init: Quasar packer-BFD retarget before tilize (see loop below)
#endif
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // X = output width
    // Y = output height
    // input shape = (..., H, W)
    // output shape = (..., Y, X)

    /**
     * This kernel takes in the contiguous XW block read in in the reader kernel and transposes is to a WX block, ready
     * to be written out The transpose LLK does not support transposing a tile without faces/subtiles, so we need to
     * rearrange it into its faces, transpose, and then pack it back such that it's de-faced (WX, where X is contiguous
     * and isn't divided into subtiles)
     */
    uint32_t start_block = get_arg(args::start_block);
    uint32_t end_block = get_arg(args::end_block);

    constexpr auto cb_in = dfb::cb_in;
    constexpr auto cb_tilize = dfb::cb_tilize;
    constexpr auto cb_out = dfb::cb_out;

    DataflowBuffer dfb_tilize_exp(cb_tilize);
    DataflowBuffer dfb_out_exp(cb_out);

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    for (uint32_t block = start_block; block < end_block; block++) {
#ifdef ARCH_QUASAR
        // Quasar: the packer's L1 destination (BFD) is baked by pack_init, and Quasar's tilize_init
        // programs unpack+math only. compute_kernel_hw_startup aimed the packer at cb_out and
        // pack_untilize_dest_init below re-aims it at cb_out every iteration, so without this the
        // tilized tile is packed into cb_out's ring (the LLK re-init guard asserts; with asserts off
        // cb_tilize is never written). WH/BH tilize_init programs the packer itself; nothing changes there.
        pack_init(cb_tilize);
#endif
        // Tilize input via unpack and then pack (standard symmetric: 1 tile in → 1 tile out)
        compute_kernel_lib::tilize<
            1,
            cb_in,
            cb_tilize,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

        // transpose input
        dfb_tilize_exp.wait_front(1);

        transpose_init(cb_tilize);
        pack_untilize_dest_init<1>(cb_out);

        tile_regs_acquire();
        transpose_tile(cb_tilize, 0, 0);  // transpose call
        tile_regs_commit();

        // pack and untilize
        dfb_out_exp.reserve_back(1);

        tile_regs_wait();
        pack_untilize_dest<1>(cb_out);  // pack call
        tile_regs_release();

        dfb_out_exp.push_back(1);

        pack_untilize_uninit(cb_out);

        dfb_tilize_exp.pop_front(1);
    }
}
