// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) compute kernel for binary_ng's no-broadcast binary op.
//
// Faithful DFB port of the CB compute kernel
//   ttnn/.../binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp
// using the DataflowBuffer pattern from
//   tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary_2_0.cpp
//
// CB->DFB mapping is 1:1: reserve_back / push_back / wait_front / pop_front are unchanged; the
// DataflowBuffer is constructed from a `dfb::<name>` accessor (in0/in1/out) rather than a CBIndex,
// and the LLK compute macros take the dfb accessor directly instead of `.get_cb_id()`.
//
// Defines (set by the program factory):
//   ELTWISE_OP_TYPE / ELTWISE_OP            — the binary op (ADD => add_tiles).
//   PACK_RELU                               — fuse a RELU on the packer (no-bcast fast path).
//   PROCESS_POST_ACTIVATIONS(i)             — optional SFPU post-activation chain.
//
// Runtime arg: num_tiles (this core's shard tile count).
// Compile-time arg: num_tiles_per_cycle (tiles processed per DST acquire; bounded by DST capacity).

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifndef PROCESS_POST_ACTIVATIONS
#define PROCESS_POST_ACTIVATIONS(i)
#endif

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    binary_tiles_init<true, ELTWISE_OP_TYPE>(dfb::in0, dfb::in1);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

    auto process_tiles = [&](uint32_t n) {
        dfb_in0.wait_front(n);
        dfb_in1.wait_front(n);
        dfb_out.reserve_back(n);

        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            ELTWISE_OP(dfb::in0, dfb::in1, i, i, i);
            PROCESS_POST_ACTIVATIONS(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dfb::out);
        }
        tile_regs_release();

        dfb_out.push_back(n);
        dfb_in0.pop_front(n);
        dfb_in1.pop_front(n);
    };

    const uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_tiles(num_tiles_per_cycle);
    }
    const uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles(remainder);
    }
}
