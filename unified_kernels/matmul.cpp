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

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbIn0 = 0;
constexpr uint32_t kCbIn1 = 1;
constexpr uint32_t kCbBias = 2;  // MM_BIAS only: 1 x ct tiles, resident
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kCbAcc = 24;  // running total; a separate CB from kCbOut

// No geometry to declare: the operand SHAPES below are the geometry, and the DST
// budget static_assert keys on the output block they imply. MM_K_BLOCKS stays a plain
// loop bound, which is all it ever was.
//
// The one thing shapes cannot derive is how B's tiles are read, and it is named ONCE
// here because matmul_init and every matmul() must agree and nothing can check that
// they do. With MM_TRANSPOSE the host supplies in1 grid-transposed, so this computes
// A @ B-transpose; see TransposeB in tt/unified/math.hpp for why both halves are needed.
#if defined(MM_TRANSPOSE)
constexpr auto kTransposeB = u::TransposeB::Yes;
#else
constexpr auto kTransposeB = u::TransposeB::No;
#endif
//
// One spelling of the fusion, biased or not, so the relu variants below do not
// each need two forms.
#if defined(MM_BIAS)
#define MM_FUSION(x, y) u::matmul<kTransposeB>(x, y).bias(bias)
#else
#define MM_FUSION(x, y) u::matmul<kTransposeB>(x, y)
#endif

constexpr uint32_t kIn0Tiles = MM_RT_DIM * MM_KT_DIM;
constexpr uint32_t kIn1Tiles = MM_KT_DIM * MM_CT_DIM;
constexpr uint32_t kOutTiles = MM_RT_DIM * MM_CT_DIM;

void kernel_main() {
    constexpr auto in0_args = TensorAccessorArgs<0>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
#if defined(MM_BIAS)
    // Last, so a build without MM_BIAS sees exactly the layout it always did.
    constexpr auto bias_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#endif

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t in1_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
#if defined(MM_BIAS)
    const uint32_t bias_addr = get_arg_val<uint32_t>(3);
#endif

    using In0 = u::Shape<MM_RT_DIM, MM_KT_DIM>;
    using In1 = u::Shape<MM_KT_DIM, MM_CT_DIM>;
    using Out = u::Shape<MM_RT_DIM, MM_CT_DIM>;

    u::Storage<In0> in0_storage(kCbIn0);
    u::Storage<In1> in1_storage(kCbIn1);
    u::Storage<Out> acc_storage(kCbAcc);  // running total -- must NOT be kCbOut
    u::Storage<Out> out_storage(kCbOut);

    // The FPU path needs its own hardware startup: SrcOrder::Reverse plus the
    // block dims. compute_init() (init_sfpu) would leave the ALU configured for
    // SFPU work and matmul could not run against it.
    u::matmul_init<In0, In1, kTransposeB>(kCbIn0, kCbIn1, kCbOut);

    const auto in0 = TensorAccessor(in0_args, in0_addr);
    const auto in1 = TensorAccessor(in1_args, in1_addr);
    const auto out = TensorAccessor(out_args, out_addr);

#if defined(MM_BIAS)
    // KERNEL SCOPE, deliberately. Every finishing k-block reads this, so it must
    // stay in the buffer for the whole kernel -- and a ComputeBlock pops in its
    // destructor, which here is the end of the kernel. Declared inside the loop it
    // would be popped after one use and the next block would hang waiting for a
    // refill that never comes. The CB holds exactly ct tiles, so the one reserve
    // below is the only one it ever gets.
    u::Storage<u::Shape<1, MM_CT_DIM>> bias_storage(kCbBias);
    const auto bias_acc = TensorAccessor(bias_args, bias_addr);
    u::ComputeBlock bias = u::noc_load<1>(bias_storage, bias_acc, 0).wait();
#endif

#if defined(MM_SINGLE_SHOT)
    // Straight through store(), with no accumulation buffer -- and so no rt*ct <= 8
    // limit, because the strategy walks the output in row bands instead. This is the path
    // a fused attention takes, and its shape limits differ from the accumulating one's,
    // which is the whole reason the sweep has to be able to reach it.
    static_assert(MM_K_BLOCKS == 1, "the single-shot path is one k-block by definition");
#elif defined(MM_ACC_L1)
    // L1: the packer sums into acc_storage, so DST only ever holds one block's
    // product and a per-step chain sees that contribution alone.
    u::Accumulator<Out, u::AccumulatorMode::L1> acc(acc_storage, out_storage);
    acc.clear();
#else
    u::Accumulator<Out, u::AccumulatorMode::Dst> acc(acc_storage, out_storage);
    acc.clear();
#endif

    for (uint32_t k = 0; k < MM_K_BLOCKS; ++k) {
        const bool finish = (k == MM_K_BLOCKS - 1);

        u::ComputeBlock a = u::noc_load<1>(in0_storage, in0, k).wait();
        u::ComputeBlock b = u::noc_load<1>(in1_storage, in1, k).wait();

#if defined(MM_SINGLE_SHOT)
        (void)finish;
        u::Block result = out_storage.store(MM_FUSION(a, b));
        u::noc_store<0>(std::move(result), out, 0);
#else

#if defined(MM_RELU_EPILOGUE)
        // finish-only: relu once, on the completed accumulator
        u::Block result = acc.accumulate(MM_FUSION(a, b), finish, [](auto mm) { return u::relu(mm); });
#elif defined(MM_RELU_PER_STEP)
        // per-step: relu on every k-block, carried forward in the accumulator
        u::Block result = acc.accumulate(u::relu(MM_FUSION(a, b)), finish);
#elif defined(MM_RELU_BOTH)
        // both chains at once: relu per k-block, then exp once on the total.
        // exp rather than relu for the epilogue so the two stages stay
        // distinguishable -- relu of a sum of relus would be a no-op.
        u::Block result =
            acc.accumulate(u::relu(u::matmul<kTransposeB>(a, b)), finish, [](auto mm) { return u::exp_(mm); });
#else
        u::Block result = acc.accumulate(MM_FUSION(a, b), finish);
#endif

        if (finish) {
            u::noc_store<0>(std::move(result), out, 0);
        }
#endif
    }
}
