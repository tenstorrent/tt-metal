// SPDX-License-Identifier: Apache-2.0
//
// Multi-core matmul with BOTH operands multicast. ONE source, compiled once per
// baby RISC-V thread, run on an R x C grid of cores.
//
// Core (r, c) computes C_block[r][c] = sum over k of A_block[r][k] @ B_block[c][k].
// The K loop lives INSIDE the kernel, exactly as it does in the reference
// (bmm_large_block_zm_fused_bias_activation.cpp, `for block < num_blocks_inner_dim`
// with `spill = num_blocks_inner_dim > 1`): K is not split across cores. Every
// core accumulates its own partials, all of them concurrently, and the broadcast
// of k-block j+1 overlaps the compute of j.
//
// For each k step:
//
//   * A_block[r][k] is broadcast along ROW r      -- sender is core (r, 0)
//   * B_block[c][k] is broadcast down COLUMN c    -- sender is core (0, c)
//
// A core is therefore the sender of one broadcast and a receiver of the other, so
// the two broadcasts need SEPARATE handshake pairs. That falls out for free by
// running them on different data-movement threads: the harness reserves a pair
// per thread, and each noc_load picks up its own thread's pair. No semaphore is
// named anywhere in this kernel.
//
// They cannot share one pair because of the READY counter, not the sent flag:
// noc_semaphore_wait spins until the value EQUALS its target, so if core (1,0)
// finishes its row broadcast and counts itself into core (0,0)'s ready while
// (0,0) is still waiting on (0,1), the counter steps 0 -> 2 and wait(1) never
// matches. (The sent flag is safe either way, since a receiver only signals ready
// after it has finished its own send.)
//
// Running on two threads also means two NOCs, so the broadcasts overlap rather
// than serialize. Worth 8% to 19% over serializing them, most at small grids.
//
// WHICH broadcast gets WHICH NOC is worth more than that, and this file had it
// backwards. The row broadcast belongs on NOC 1 and the column broadcast on NOC 0
// -- MM_IN0_THREAD=1 with MM_IN1_THREAD=0 -- which is faster than the reverse in
// every cell measured, by 6% to 44%, largest where the RHS is the bigger operand
// (8x8 kt=32 2x4: 106.70us -> 59.94us, and 2.60x ttnn -> 1.47x). Both NOCs route
// in opposite directions, so a broadcast pointed the wrong way takes the long
// path; the old assignment pointed BOTH the wrong way, which is why the cost
// looked like it belonged to whichever operand happened to be larger.
//
// The defaults here stay 0/0 all the same, because that is what ttsim can run and
// the suite has to pass there. Hardware callers should pass 1/0; bench_matmul.py
// does.
//
// ttsim cannot multicast on NOC 1 (it does not implement coordinate
// virtualization), so MM_IN1_THREAD=0 puts both on NOC 0 where they serialize.
// The pair is then named explicitly, because the two broadcasts must not share
// one: their ready counters would interleave and a wait-for-equality would miss.
//
// Operands are laid out block-major so every block is contiguous pages: A is
// R*K blocks of rt x kt tiles indexed r*K + k, B is C*K blocks of kt x ct tiles
// indexed c*K + k, and the output is R*C blocks of rt x ct in row-major core
// order.
//
// Compile-time args: a dfb_<name> per buffer.
//
// Runtime args (identical on every core):
//
// A core's row and column are not passed in: it asks where it is. Note that
// LogicalCoord::this_core() is relative to the SUB-DEVICE origin while
// to_physical() indexes the absolute worker-logical tables, so the two agree only
// for a program whose core range starts at (0,0) -- as this one's does.
//
// Defines:
//   MM_RT / MM_CT / MM_KT      output block tiles, per-k-block inner dim
//   MM_K_BLOCKS                k-blocks each core accumulates over
//   MM_GRID_H / MM_GRID_W      core grid
//   MM_ACC_L1                  if set, accumulate in L1 rather than through DST
//   MM_IN1_THREAD              DM thread for the RHS broadcast: 1 on hardware
//                              (second NOC, overlapped), 0 on ttsim
//   MM_IN0_THREAD              DM thread for the LHS broadcast, default 0. 1 with
//                              MM_IN1_THREAD=0 is the MEASURED-BEST assignment on
//                              hardware -- see the note above on which NOC each
//                              direction wants.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t kDfbIn0 = get_arg(args::dfb_in0);
    constexpr uint32_t kDfbIn1 = get_arg(args::dfb_in1);
    constexpr uint32_t kDfbAcc = get_arg(args::dfb_acc);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);

    const u::LogicalCoord me = u::LogicalCoord::this_core();
    const uint32_t out_block = me.y * MM_GRID_W + me.x;

    // The shapes come first: they are what matmul_init programs the block dimensions
    // from, now that the geometry is derived rather than declared.
    using In0 = u::Shape<MM_RT, MM_KT>;
    using In1 = u::Shape<MM_KT, MM_CT>;
    using Out = u::Shape<MM_RT, MM_CT>;

    u::matmul_init<In0, In1>(kDfbIn0, kDfbIn1, kDfbOut);

#ifndef MM_IN0_THREAD
#define MM_IN0_THREAD 0
#endif

    // Each broadcast drives its own thread's handshake pair, which is the natural
    // configuration; they only have to be NAMED when both land on one thread, since two
    // broadcasts sharing a pair would interleave their ready counters and a
    // wait-for-equality would miss.
    constexpr int kIn0Pair = (MM_IN0_THREAD == MM_IN1_THREAD) ? 0 : MM_IN0_THREAD;
    constexpr int kIn1Pair = (MM_IN0_THREAD == MM_IN1_THREAD) ? 1 : MM_IN1_THREAD;
    static_assert(kIn0Pair != kIn1Pair, "the two broadcasts must not share a handshake pair");

    u::Input<MM_IN0_THREAD, kDfbIn0, In0> in0_storage;
    u::Input<MM_IN1_THREAD, kDfbIn1, In1> in1_storage;
    u::Intermediate<kDfbAcc, Out> acc_storage;
    u::Output<0, kDfbOut, Out> out_storage;

    const auto in0 = TensorAccessor(tensor::in0);
    const auto in1 = TensorAccessor(tensor::in1);
    const auto out = TensorAccessor(tensor::out);

    // The row this core sits in, and the column it sits in. Every core in a row
    // runs the same row statement; which side of the handshake it takes is a
    // runtime decision on its own coordinate.
    const u::LogicalMcast row{u::LogicalCoord::yx(me.y, 0), u::Extent::hw(1, MM_GRID_W)};
    const u::LogicalMcast col{u::LogicalCoord::yx(0, me.x), u::Extent::hw(MM_GRID_H, 1)};

#if defined(MM_ACC_L1)
    u::Accumulator<Out, u::AccumulatorMode::L1> acc(acc_storage, out_storage);
#else
    u::Accumulator<Out, u::AccumulatorMode::Dst> acc(acc_storage, out_storage);
#endif
    acc.clear();

    for (uint32_t k = 0; k < MM_K_BLOCKS; ++k) {
        const bool finish = (k == MM_K_BLOCKS - 1);

        // One thread broadcasts the LHS along the row, the other the RHS down the
        // column; each takes a distinct reserved handshake pair, so the two never
        // collide. Both re-run every k step, feeding the next block while the
        // previous one is still being folded in.
        u::ComputeBlock a = u::noc_load<kIn0Pair>(in0_storage, row, in0, me.y * MM_K_BLOCKS + k).wait();
        u::ComputeBlock b = u::noc_load<kIn1Pair>(in1_storage, col, in1, me.x * MM_K_BLOCKS + k).wait();

        u::Block result = acc.accumulate(u::matmul(a, b), finish);
        if (finish) {
            u::noc_store<0>(std::move(result), out, out_block);
        }
    }
}
