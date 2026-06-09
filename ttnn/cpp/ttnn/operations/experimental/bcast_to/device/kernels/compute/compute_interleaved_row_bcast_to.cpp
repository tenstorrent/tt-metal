// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // The compute reads src tiles straight off the CB front in order; the n/c/th/tw
    // start offsets and N/C/Ht/Wt extents only steer the *reader*, never the compute.
    // Row-bcast is 1 src tile -> 1 dst tile, and the original innermost loop did exactly
    // one unary_bcast per tile with ++num_tiles_read, so the whole 4-deep nest is just a
    // streaming walk of `num_tiles` tiles. Flatten it into a single chain call so the
    // per-op bcast init (UnaryBcast::init) hoists once at chain entry; compute_kernel_hw_startup
    // supplies the BIG init the chain expects the caller to own.
    uint32_t num_tiles = get_arg_val<uint32_t>(5);

    constexpr auto cb_id_src = get_compile_time_arg_val(0);
    constexpr auto cb_id_dst = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_id_src, cb_id_dst);

    compute_kernel_lib::unary_bcast<
        compute_kernel_lib::BroadcastDim::Row,
        cb_id_src,
        cb_id_dst,
        compute_kernel_lib::InputLifecycle::Streaming,
        compute_kernel_lib::OutputLifecycle::Streaming,
        compute_kernel_lib::UnaryBcastReconfig::Input,
        compute_kernel_lib::PackTileReconfig::None>(num_tiles);
}
