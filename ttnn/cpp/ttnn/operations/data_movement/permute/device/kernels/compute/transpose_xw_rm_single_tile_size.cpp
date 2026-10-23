// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
    constexpr uint32_t x_block_size = get_arg(args::x_block_size);
    constexpr uint32_t w_block_size = get_arg(args::w_block_size);

    uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr auto cb_in = dfb::cb_in;
    constexpr auto cb_tilize = dfb::cb_tilize;
    constexpr auto cb_out = dfb::cb_out;

    DataflowBuffer dfb_tilize_exp(cb_tilize);
    DataflowBuffer dfb_out_exp(cb_out);

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    for (uint32_t n = 0; n < num_blocks; n++) {
#ifdef ARCH_QUASAR
        // Quasar: the packer's L1 destination (BFD) is baked by pack_init, and Quasar's tilize_init
        // programs unpack+math only. compute_kernel_hw_startup aimed the packer at cb_out and
        // pack_untilize_dest_init below re-aims it at cb_out every iteration, so without this the
        // tilized tile is packed into cb_out's ring (the LLK re-init guard asserts; with asserts off
        // cb_tilize is never written). WH/BH tilize_init programs the packer itself; nothing changes there.
        pack_init(cb_tilize);
#endif
        // Tilize input via unpack and then pack (asymmetric: x_block_size rows → 1 tile)
        compute_kernel_lib::tilize<
            1,
            cb_in,
            cb_tilize,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1, x_block_size);

        // transpose input
        dfb_tilize_exp.wait_front(1);
        transpose_init(cb_tilize);
        pack_untilize_dest_init<1>(cb_out);

        tile_regs_acquire();
        transpose_tile(cb_tilize, 0, 0);  // transpose call
        tile_regs_commit();

        // pack and untilize
        dfb_out_exp.reserve_back(w_block_size);

        tile_regs_wait();
        pack_untilize_dest<1>(cb_out);  // pack call
        tile_regs_release();

        dfb_out_exp.push_back(w_block_size);

        pack_untilize_uninit(cb_out);

        dfb_tilize_exp.pop_front(1);
    }
}
