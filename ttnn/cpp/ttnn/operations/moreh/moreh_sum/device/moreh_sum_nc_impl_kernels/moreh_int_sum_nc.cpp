// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/sfpu_int_sum.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    // compile-time args
    // num_output_tiles carries the per-core work-split count (the host's num_cols_per_core_group_N).
    constexpr uint32_t num_output_tiles = get_arg(args::num_output_tiles);
    constexpr uint32_t num_input_tiles = get_arg(args::num_input_tiles);

    DataflowBuffer dfb_in0_obj(dfb::input);
    DataflowBuffer dfb_out0_obj(dfb::out);
    DataflowBuffer dfb_intermed0_obj(dfb::intermed0);
    constexpr int onetile = 1;
    constexpr int idx0 = 0;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    compute_kernel_hw_startup(dfb::input, dfb::out);
    copy_init(dfb::input);
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            bool last_out = (j == num_input_tiles - 1);
            tile_regs_acquire();
            copy_tile_to_dst(dfb_in0_obj, idx0, dst0);
            if (enable_reload) {
                copy_tile_to_dst(dfb_intermed0_obj, idx0, dst1);
                sfpu_sum_int_init();
                sfpu_add_int(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            // Selected at runtime between the output DFB and the intermediate DFB; stays
            // uint32_t-valued, since the generated dfb:: handles share one type and convert to
            // uint32_t at compile time. Both DFBs are bound to this kernel, so both tokens exist.
            uint32_t out_dfb = (last_out) ? (dfb::out) : (dfb::intermed0);
            pack_tile_from_dst(DataflowBuffer(out_dfb), dst0);
            tile_regs_release();
            enable_reload = true;
        }
    }
}
