// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local compute kernel for rand. The RNG logic is unchanged from the legacy
// uniform compute kernel (compute_uniform.cpp); only the resource bindings move to the Metal 2.0
// namespaces (dfb::/args::). seed/from/to are per-enqueue named runtime args: they are excluded
// from the program-cache key and re-applied on every dispatch via the factory's
// create_program_run_args (SetProgramRunArgs on cache hit), so they are never frozen.

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
    f2u_from.u = get_arg(args::from_bits);
    f2u_to.u = get_arg(args::to_bits);
    f2u_scale.f = f2u_to.f - f2u_from.f;
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t end_id = start_id + num_tiles;

    DataflowBuffer cb_intermed(dfb::cb_intermed);

    init_sfpu(dfb::cb_intermed, dfb::cb_intermed);

    rand_tile_init(seed);
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_intermed.reserve_back(1);

        tile_regs_acquire();
        rand_tile(0, f2u_from.u, f2u_scale.u);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::cb_intermed, 0);
        tile_regs_release();

        cb_intermed.push_back(1);
    }
}
