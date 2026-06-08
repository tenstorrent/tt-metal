// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"

// Metal 2.0 compute producer of the rand DFB.
// The only change from the descriptor-era kernel: the intermediate CB id and the runtime args
// now come from the spec's binding namespaces (dfb:: / args::) instead of positional compile-time
// and runtime arg slots. `dfb::rand_tiles` implicitly converts to the underlying CB id, so the
// existing compute pipeline (init_sfpu / rand_tile / pack_tile) is used unchanged. Because the
// DFB's data format is the OUTPUT dtype, pack_tile converts fp32 dest -> output format here.
void kernel_main() {
    constexpr uint32_t rand_cb = dfb::rand_tiles;  // DFB token -> CB id for the compute APIs

    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to, f2u_scale;
    f2u_from.u = get_arg(args::from);  // common RTA (broadcast)
    f2u_to.u = get_arg(args::to);
    f2u_scale.f = f2u_to.f - f2u_from.f;
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t end_id = start_id + num_tiles;

    // seed is a single broadcast base (same for every core on the device); per-core distinctness is
    // recovered here as base + start_id (start_id is the core's static tile offset, unique per core).
    // This keeps the dynamic dispatch arg to one broadcast scalar instead of a per-core value.
    const uint32_t seed = get_arg(args::seed) + start_id;

    init_sfpu(rand_cb, rand_cb);

    rand_tile_init(seed);
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(rand_cb, 1);

        tile_regs_acquire();
        rand_tile(0, f2u_from.u, f2u_scale.u);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, rand_cb, 0);
        tile_regs_release();

        cb_push_back(rand_cb, 1);
    }
}
