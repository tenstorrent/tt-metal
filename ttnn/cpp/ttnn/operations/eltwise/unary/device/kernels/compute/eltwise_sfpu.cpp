// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/dataflow_buffer.h"
#ifdef TT_POLY_SELECTED_CONFIG_HEADER
#include "api/compute/eltwise_unary/tt_poly_compiled.h"
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    DataflowBuffer dfb_in(cb_input);
    DataflowBuffer dfb_out(cb_output);

    compute_kernel_hw_startup(cb_input, cb_output);
    copy_init(cb_input);
#ifdef TT_POLY_BACKWARD_INPUTS
    constexpr auto cb_grad = tt::CBIndex::c_1;
#endif
#ifdef SFPU_OP_PROGRAM_INIT_0
    SFPU_OP_PROGRAM_INIT_0
#endif
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

#ifdef SFPU_OP_COPY_0
        SFPU_OP_COPY_0
#else
        copy_tile(cb_input, 0, 0);
#endif

#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);

        dfb_in.pop_front(1);
        dfb_out.push_back(1);

#ifdef TT_POLY_BACKWARD_INPUTS
        cb_pop_front(cb_grad, 1);
#endif
        tile_regs_release();
    }
#ifdef SFPU_OP_PROGRAM_FINISH_0
    SFPU_OP_PROGRAM_FINISH_0
#endif
}
