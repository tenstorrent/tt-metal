// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"

// Metal 2.0 compute producer of the rand DFB. The only changes from the descriptor-era kernel: the
// DFB id and runtime args come from the spec binding namespaces (dfb:: / args::) instead of positional
// compile-time/runtime slots. The DFB is an fp32 intermediate (as before): pack_tile writes fp32->fp32
// and the writer narrows to the output dtype.
//
// seed/from/to are PER-ENQUEUE runtime args (rebuilt and re-applied via UpdateProgramRunArgs on every
// dispatch); start_id/num_tiles are the enqueue-invariant work split (set once on cache miss).
void kernel_main() {
    constexpr uint32_t rand_cb = dfb::rand_tiles;

    const uint32_t seed = get_arg(args::seed);
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to, f2u_scale;
    f2u_from.u = get_arg(args::from);
    f2u_to.u = get_arg(args::to);
    f2u_scale.f = f2u_to.f - f2u_from.f;
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t end_id = start_id + num_tiles;

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
