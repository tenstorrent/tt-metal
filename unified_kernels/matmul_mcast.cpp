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
// than serialize -- that is MM_IN1_THREAD=1, the hardware configuration.
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
// Compile-time args: a cb_<name> per buffer.
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

#include <tt/unified/core>

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t kCbIn0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t kCbIn1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t kCbAcc = get_named_compile_time_arg_val("cb_acc");
    constexpr uint32_t kCbOut = get_named_compile_time_arg_val("cb_out");

    const u::LogicalCoord me = u::LogicalCoord::this_core();
    const uint32_t out_block = me.y * MM_GRID_W + me.x;

    // The shapes come first: they are what matmul_init programs the block dimensions
    // from, now that the geometry is derived rather than declared.
    using In0 = u::Shape<MM_RT, MM_KT>;
    using In1 = u::Shape<MM_KT, MM_CT>;
    using Out = u::Shape<MM_RT, MM_CT>;

    u::matmul_init<In0, In1>(kCbIn0, kCbIn1, kCbOut);

    u::Storage<In0> in0_storage(kCbIn0);
    u::Storage<In1> in1_storage(kCbIn1);
    u::Storage<Out> acc_storage(kCbAcc);
    u::Storage<Out> out_storage(kCbOut);

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

        // Thread 0 broadcasts the LHS along the row, thread 1 the RHS down the
        // column; each takes its own reserved handshake pair, so the two never
        // collide. Both re-run every k step, feeding the next block while the
        // previous one is still being folded in.
        u::ComputeBlock a = u::noc_load<0, /*pair=*/0>(in0_storage, row, in0, me.y * MM_K_BLOCKS + k).wait();
        u::ComputeBlock b =
            u::noc_load<MM_IN1_THREAD, /*pair=*/1>(in1_storage, col, in1, me.x * MM_K_BLOCKS + k).wait();

        u::Block result = acc.accumulate(u::matmul(a, b), finish);
        if (finish) {
            u::noc_store<0>(std::move(result), out, out_block);
        }
    }
}
