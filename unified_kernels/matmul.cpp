// SPDX-License-Identifier: Apache-2.0
//
// A unified matmul kernel: ONE source, compiled once per baby RISC-V thread.
//
// Computes   C = A @ B   for a single output subblock:
//     A is RT x KT tiles, B is KT x CT tiles, C is RT x CT tiles.
//
// This is the FPU fusion path. Unlike the SFPU path, the FPU consumes the whole
// DST register file -- matmul_block advances dst_index itself across the
// rt_dim x ct_dim subblock -- so there is nothing for the expression allocator
// to hand out and only a unary epilogue could fuse.
//
// What each thread ends up executing:
//
//   NCRISC   fill cb_in0 (RT*KT tiles) and cb_in1 (KT*CT tiles)
//   TRISC    wait both, matmul_block across k accumulating into DST,
//            pack the RT*CT subblock, push cb_out
//   BRISC    drain cb_out (RT*CT tiles)
//
// Compile-time args:
//   0..      TensorAccessorArgs for in0, then in1, then out
//
// Runtime args (identical on all three kernels):
//   0        in0 base address
//   1        in1 base address
//   2        out base address

#include <tt/unified>

namespace u = tt::unified;

constexpr uint32_t kCbIn0 = 0;
constexpr uint32_t kCbIn1 = 1;
constexpr uint32_t kCbOut = 16;

// Geometry is compile-time so the strategy can unroll and the DST budget is
// checked by static_assert. Supplied by the host as -D flags.
using Geom = u::MatmulGeometry<MM_RT_DIM, MM_CT_DIM, MM_KT_DIM>;

constexpr uint32_t kIn0Tiles = MM_RT_DIM * MM_KT_DIM;
constexpr uint32_t kIn1Tiles = MM_KT_DIM * MM_CT_DIM;
constexpr uint32_t kOutTiles = MM_RT_DIM * MM_CT_DIM;

void kernel_main() {
    constexpr auto in0_args = TensorAccessorArgs<0>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t in1_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    u::Storage in0_storage(kCbIn0, kIn0Tiles);
    u::Storage in1_storage(kCbIn1, kIn1Tiles);
    u::Storage out_storage(kCbOut, kOutTiles);

    // The FPU path needs its own hardware startup: SrcOrder::Reverse plus the
    // block dims. compute_init() (init_sfpu) would leave the ALU configured for
    // SFPU work and matmul could not run against it.
    u::matmul_init<Geom>(kCbIn0, kCbIn1, kCbOut);

    const auto in0 = TensorAccessor(in0_args, in0_addr);
    const auto in1 = TensorAccessor(in1_args, in1_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // Reader (DM thread 1) fills both operand buffers; compute waits on them.
    u::ComputeBlock a = u::noc_load<1>(in0_storage, in0, 0).wait();
    u::ComputeBlock b = u::noc_load<1>(in1_storage, in1, 0).wait();

    // The FPU strategy owns the k-loop; `matmul` yields an FPUFusion node.
    u::Block result = out_storage.store(u::matmul<Geom>(a, b));

    // Writer (DM thread 0) drains it.
    u::noc_store<0>(std::move(result), out, 0);
}
