// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t seed = get_arg(args::seed);
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to, f2u_scale;
    f2u_from.u = get_arg(args::f2u_from);
    f2u_to.u = get_arg(args::f2u_to);
    f2u_scale.f = f2u_to.f - f2u_from.f;
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t end_id = start_id + num_tiles;

    DataflowBuffer dfb_intermed(dfb::intermed);

    init_sfpu(dfb::intermed, dfb::intermed);

    rand_tile_init(seed);
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_intermed.reserve_back(1);

        tile_regs_acquire();
        rand_tile(0, f2u_from.u, f2u_scale.u);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::intermed, 0);
        tile_regs_release();

        dfb_intermed.push_back(1);
    }
}
