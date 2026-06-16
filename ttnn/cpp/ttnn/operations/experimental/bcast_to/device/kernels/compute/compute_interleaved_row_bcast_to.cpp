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
    uint32_t arg_index = 0;
    uint32_t start_n = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_c = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_t = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_th = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_tw = get_arg_val<uint32_t>(arg_index++);
    uint32_t num_tiles = get_arg_val<uint32_t>(arg_index++);
    uint32_t n_stride = get_arg_val<uint32_t>(arg_index++);
    uint32_t c_stride = get_arg_val<uint32_t>(arg_index++);
    uint32_t N = get_arg_val<uint32_t>(arg_index++);
    uint32_t C = get_arg_val<uint32_t>(arg_index++);
    uint32_t Ht = get_arg_val<uint32_t>(arg_index++);
    uint32_t Wt = get_arg_val<uint32_t>(arg_index++);

    constexpr auto cb_id_src = get_compile_time_arg_val(0);
    constexpr auto cb_id_dst = get_compile_time_arg_val(1);
    // Standard BIG init only; the chain's UnaryBcast::init supplies the per-Dim bcast MOP
    // each call (the boot unary_bcast_init's MOP was redundant with it).
    compute_kernel_hw_startup(cb_id_src, cb_id_dst);

    // Hoist the chain init out of the per-tile loop: build the chain object once, emit its init
    // once via chain.hoist_init(), then call chain.body(1) per tile instead of the
    // self-initializing unary_bcast(1). Same loop; the UnaryBcast MOP + reconfig is no longer
    // re-emitted every iteration. (compute_kernel_hw_startup is the BIG hw init, kept separate.)
    auto chain = compute_kernel_lib::make_chain(
        compute_kernel_lib::UnaryBcast<
            compute_kernel_lib::BroadcastDim::Row,
            cb_id_src,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::UnaryBcastReconfig::Input>{},
        compute_kernel_lib::PackTile<
            cb_id_dst,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::None>{});
    chain.hoist_init();

    uint32_t HtWt = Ht * Wt;
    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th, start_tw = 0) {
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_read < num_tiles; ++tw) {
                    chain.body(1u);
                    ++num_tiles_read;
                }
            }
        }
    }
}
