// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of eltwise_copy.cpp, which lives beside it.
// Ops ported to Metal 2.0 bind this file; the original serves the consumers still on the legacy API.
// Until the last of them migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::in, dfb::out) and the named argument (per_core_tile_cnt) are this
// fork's interface: every later consumer inherits them, so they are taken from the kernel's own
// vocabulary rather than any one op's locals, and are not renamed once a consumer exists.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);
    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // Pop tile after tile, copy to DST and pack
        dfb_in.wait_front(1);

        tile_regs_acquire();
        copy_tile(dfb::in, 0, 0);
        tile_regs_commit();

        dfb_in.pop_front(1);

        dfb_out.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_out.push_back(1);
    }
}
