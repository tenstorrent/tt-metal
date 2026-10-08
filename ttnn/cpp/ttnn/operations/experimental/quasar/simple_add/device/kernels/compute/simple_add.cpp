// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// simple_add compute: C = A + B, one tile at a time. Runs as num_threads threads, one per Tensix engine
// (4 on a Quasar Neo cluster, 1 on Wormhole/Blackhole). The DFBs are STRIDED, so thread t of N gets tiles
// t, t+N, t+2N, ... of the num_tiles the single reader pushes; it only needs to know how many that is.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t num_input_tiles = get_arg(args::num_input_tiles);
    // This thread's share of the strided input sub-stream {t, t+N, ...}: floor(num_input_tiles/N), plus one for
    // the first num_input_tiles % N threads. Every input must be popped or the reader never drains. Inputs past
    // num_tiles are filler and produce no output tile.
    const uint32_t num_threads = get_num_threads();
    const uint32_t thread_id = get_my_thread_id();
    const uint32_t my_inputs = num_input_tiles / num_threads + (thread_id < num_input_tiles % num_threads ? 1u : 0u);
    constexpr uint32_t dst_reg = 0;

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    add_init(dfb::in0, dfb::in1);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    for (uint32_t i = 0; i < my_inputs; ++i) {
        dfb_in0.wait_front(1);
        dfb_in1.wait_front(1);
        if (thread_id + i * num_threads >= num_tiles) {
            dfb_in0.pop_front(1);
            dfb_in1.pop_front(1);
            continue;
        }

        tile_regs_acquire();
        add_tiles(dfb::in0, dfb::in1, 0, 0, dst_reg);
        tile_regs_commit();

        dfb_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(dst_reg, dfb::out);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_in0.pop_front(1);
        dfb_in1.pop_front(1);
    }
}
