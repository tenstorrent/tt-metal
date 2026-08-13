// SPDX-License-Identifier: Apache-2.0
//
// A unified kernel: ONE source, compiled once per baby RISC-V thread.
//
// Computes   out = exp(in0 + in1)   over `num_blocks` blocks of
// `tiles_per_block` tiles each.
//
// The host points three KernelDescriptors at this file -- a reader
// (RISCV_1/NCRISC), a writer (RISCV_0/BRISC), and a compute kernel -- with
// identical compile-time and runtime args. No per-thread defines are needed:
// <tt/unified> pulls in tt/unified_adaptor_v1.hpp, which derives the projection
// from the defines metal already emits for each build.
//
// What each thread ends up executing:
//
//   NCRISC   cb_reserve(in0) -> noc_read x N -> cb_push(in0)   (and in1)
//   TRISC    cb_wait(in0), cb_wait(in1), cb_reserve(out)
//              per tile: acquire -> copy,copy,add,exp -> commit/wait -> pack
//            cb_push(out), cb_pop(in1), cb_pop(in0)
//   BRISC    cb_wait(out) -> noc_write x N -> cb_pop(out)
//
// Compile-time args:
//   0            num_blocks
//   1            tiles_per_block
//   2..          TensorAccessorArgs for in0, then in1, then out
//
// Runtime args (identical on all three kernels):
//   0            in0 base address
//   1            in1 base address
//   2            out base address

#include <tt/unified>

namespace u = tt::unified;

constexpr uint32_t kCbIn0 = 0;
constexpr uint32_t kCbIn1 = 1;
constexpr uint32_t kCbOut = 16;

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(1);

    // TensorAccessor compile-time args, laid out in0, in1, out.
    constexpr auto in0_args = TensorAccessorArgs<2>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t in1_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    u::compute_init(kCbIn0, kCbOut);

    u::Storage in0_storage(kCbIn0, tiles_per_block);
    u::Storage in1_storage(kCbIn1, tiles_per_block);
    u::Storage out_storage(kCbOut, tiles_per_block);

    const auto in0 = TensorAccessor(in0_args, in0_addr);
    const auto in1 = TensorAccessor(in1_args, in1_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t b = 0; b < num_blocks; ++b) {
#if defined(EA_CUSTOM_LOAD)
        // Same fill, spelled as a custom routine. The lambda's body is compiled
        // on all five projections, so it doubles as the check that the compute
        // projection can see the data-movement intrinsics.
        u::ComputeBlock a = u::noc_load<1>(in0_storage, [&](uint32_t l1, uint32_t bytes) {
                                for (uint32_t p = 0; p < tiles_per_block; ++p) {
                                    noc_async_read(in0.get_noc_addr(b * tiles_per_block + p), l1 + p * bytes, bytes);
                                }
                            }).wait();
        u::ComputeBlock c = u::noc_load<1>(in1_storage, [&](uint32_t l1, uint32_t bytes) {
                                for (uint32_t p = 0; p < tiles_per_block; ++p) {
                                    noc_async_read(in1.get_noc_addr(b * tiles_per_block + p), l1 + p * bytes, bytes);
                                }
                            }).wait();
#else
        // Reader (DM thread 1) fills these; compute waits on them.
        u::ComputeBlock a = u::noc_load<1>(in0_storage, in0, b).wait();
        u::ComputeBlock c = u::noc_load<1>(in1_storage, in1, b).wait();
#endif

        // Compute evaluates the expression; the allocator picks DST slots.
        //   copy a -> dst0, copy c -> dst1, add(dst0,dst1) -> dst0, exp(dst0)
        u::Block result = out_storage.store(u::exp_(a + c));

        // Writer (DM thread 0) drains it.
        u::noc_store<0>(std::move(result), out, b);
    }
}
