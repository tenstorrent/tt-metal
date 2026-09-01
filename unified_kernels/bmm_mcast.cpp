// SPDX-License-Identifier: Apache-2.0
//
// BATCHED multi-core matmul with both operands multicast. ONE source, compiled once per
// baby RISC-V thread, run on an R x C grid of cores.
//
// Compile-time args, all named, plus a dfb_<name> per buffer:
//   batch, rt, ct, kt, k_blocks, grid_h, grid_w
//
// Two knobs stay DEFINES, and neither is a matter of taste:
//   BMM_ACC_L1      selects between two Accumulator TYPES, which a constexpr cannot do.
//   BMM_IN1_THREAD  the `thread` template argument of a noc_load. _thread_of in
//                   unified_harness.py resolves those from the DEFINES a launcher passes,
//                   never from named args, and raises rather than guessing -- so a named
//                   arg here would break endpoint derivation.
//

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t kDfbIn0 = get_arg(args::dfb_in0);
    constexpr uint32_t kDfbIn1 = get_arg(args::dfb_in1);
    constexpr uint32_t kDfbAcc = get_arg(args::dfb_acc);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);

    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t rt = get_arg(args::rt);
    constexpr uint32_t ct = get_arg(args::ct);
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t k_blocks = get_arg(args::k_blocks);
    constexpr uint32_t grid_h = get_arg(args::grid_h);
    constexpr uint32_t grid_w = get_arg(args::grid_w);

    const u::LogicalCoord me = u::LogicalCoord::this_core();

    using In0 = u::Shape<rt, kt>;
    using In1 = u::Shape<kt, ct>;
    using Out = u::Shape<rt, ct>;

    u::matmul_init<In0, In1>(kDfbIn0, kDfbIn1, kDfbOut);

    u::Storage<In0> in0_storage(kDfbIn0);
    u::Storage<In1> in1_storage(kDfbIn1);
    u::Storage<Out> acc_storage(kDfbAcc);
    u::Storage<Out> out_storage(kDfbOut);

    const auto in0 = TensorAccessor(tensor::in0);
    const auto in1 = TensorAccessor(tensor::in1);
    const auto out = TensorAccessor(tensor::out);

    const u::LogicalMcast row{u::LogicalCoord::yx(me.y, 0), u::Extent::hw(1, grid_w)};
    const u::LogicalMcast col{u::LogicalCoord::yx(0, me.x), u::Extent::hw(grid_h, 1)};

#if defined(BMM_ACC_L1)
    u::Accumulator<Out, u::AccumulatorMode::L1> acc(acc_storage, out_storage);
#else
    u::Accumulator<Out, u::AccumulatorMode::Dst> acc(acc_storage, out_storage);
#endif

    for (uint32_t n = 0; n < batch; ++n) {
        acc.clear();

        const uint32_t a_base = (n * grid_h + me.y) * k_blocks;
        const uint32_t b_base = (n * grid_w + me.x) * k_blocks;
        const uint32_t out_block = n * (grid_h * grid_w) + me.y * grid_w + me.x;

        for (uint32_t k = 0; k < k_blocks; ++k) {
            const bool finish = (k == k_blocks - 1);

            u::ComputeBlock a = u::noc_load<0, /*pair=*/0>(in0_storage, row, in0, a_base + k).wait();
            u::ComputeBlock b = u::noc_load<BMM_IN1_THREAD, /*pair=*/1>(in1_storage, col, in1, b_base + k).wait();

            u::Block result = acc.accumulate(u::matmul(a, b), finish);
            if (finish) {
                u::noc_store<0>(std::move(result), out, out_block);
            }
        }
    }
}
