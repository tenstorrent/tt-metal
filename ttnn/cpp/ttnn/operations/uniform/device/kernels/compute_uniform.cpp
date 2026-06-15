// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_rand.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);

    const uint32_t seed = get_arg_val<uint32_t>(0);
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to, f2u_scale;
    f2u_from.u = get_arg_val<uint32_t>(1);
    f2u_to.u = get_arg_val<uint32_t>(2);
    f2u_scale.f = f2u_to.f - f2u_from.f;
    // num_tiles is the work this core consumes; start_id (tile placement) is the writer's
    // job, so it's not passed to compute.
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);

    init_sfpu(intermed_cb_id, intermed_cb_id);

    // Out-of-band runtime-seed init — RandTile::init() is a no-op because the
    // chain has no way to thread a runtime seed through its static init hook.
    rand_tile_init(seed);

    // Per-tile rand into D0 + pack to intermed_cb. Reconfig: original used
    // init_sfpu at boot then plain pack_tile (no _with_dt) -> PackTileReconfig::None.
    eltwise_chain(
        num_tiles,
        RandTile<Dst::D0>{f2u_from.u, f2u_scale.u},
        PackTile<intermed_cb_id, OutputLifecycle::Streaming, PackTileReconfig::None>{});
}
