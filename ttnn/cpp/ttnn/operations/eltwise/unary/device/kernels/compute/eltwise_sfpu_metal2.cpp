// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of eltwise_sfpu.cpp, which lives beside it. Ops ported to
// Metal 2.0 bind this file; the original serves the consumers still on the legacy API. Until the
// last of them migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::input, dfb::output) and the named argument set are this fork's
// interface: every later consumer inherits them, so they are taken from the kernel's own
// vocabulary rather than any one op's locals, and are not renamed once a consumer exists.

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
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in(dfb::input);
    DataflowBuffer dfb_out(dfb::output);

    compute_kernel_hw_startup(dfb::input, dfb::output);
    copy_init(dfb::input);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        copy_tile(dfb::input, 0, 0);

#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb::output);

        dfb_in.pop_front(1);
        dfb_out.push_back(1);

        tile_regs_release();
    }
}
