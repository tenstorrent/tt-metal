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
//
// Profiling: when ENABLE_KERNEL_TIMER is defined (by the program factory), this kernel times its
// tile-processing work with the rdcycle-based kernel timer and writes the cycle count to L1 slot 1
// at compile-time arg timer_l1_addr. See api/debug/kernel_timer.h. WALL_CLOCK is NOT usable here —
// reading it hangs the Quasar emulator.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"  // get_my_thread_id / get_num_threads (multi-NEO partition)
#include "experimental/kernel_args.h"

#ifdef ENABLE_KERNEL_TIMER
#include "api/debug/kernel_timer.h"
constexpr uint32_t kTimerSlotCompute = 1;  // reader=0, compute=1, writer=2
#endif

#ifndef PROCESS_POST_ACTIVATIONS
#define PROCESS_POST_ACTIVATIONS(i)
#endif

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

#ifdef ENABLE_KERNEL_TIMER
    KernelTimer _timer;
    _timer.start();
#endif

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    binary_op_init_common(dfb::in0, dfb::in1, dfb::out);
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

    const uint32_t nthreads = get_num_threads();
    if (nthreads <= 1) {
        // Single-NEO fast path: bulk DST-batched chunks over the whole shard (original behavior,
        // byte-identical to the validated single-NEO add).
        const uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
        for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
            process_tiles(num_tiles_per_cycle);
        }
        const uint32_t remainder = num_tiles % num_tiles_per_cycle;
        if (remainder > 0) {
            process_tiles(remainder);
        }
    } else {
        // Multi-NEO path: N threads (one per NEO) consume a STRIDED DFB. Per the DFB tile-counter
        // round-robin (advances once per pop_front/push_back), correctness requires ONETILE
        // granularity + explicit modulo-skip — bulk count-division mis-aligns tiles (verified: a
        // count-division attempt gave wrong PCC). Each thread walks the full tile index and processes
        // tile t where (t % nthreads == my_thread_id), one tile at a time. Matches the BMM precedent
        // (tests/.../compute/bmm.cpp).
        const uint32_t tid = get_my_thread_id();
        for (uint32_t t = 0; t < num_tiles; ++t) {
            if (t % nthreads != tid) {
                continue;
            }
            process_tiles(1);
        }
        dfb_out.finish();
    }

#ifdef ENABLE_KERNEL_TIMER
    kernel_timer_write(get_arg(args::timer_l1_addr), kTimerSlotCompute, _timer.stop());
#endif
}
