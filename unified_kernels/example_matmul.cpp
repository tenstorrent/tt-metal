// SPDX-License-Identifier: Apache-2.0

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

constexpr uint32_t kGridHeight = 2;
constexpr uint32_t kGridWidth = 2;
constexpr uint32_t kKBlocks = 2;

using A = u::Shape<2, 2>;
using B = u::Shape<2, 2>;
using Out = u::Shape<2, 2>;
using Bias = u::Shape<1, 2>;

void kernel_main() {
    constexpr uint32_t kDfbA = get_arg(args::dfb_a);
    constexpr uint32_t kDfbB = get_arg(args::dfb_b);
    constexpr uint32_t kDfbBias = get_arg(args::dfb_bias);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);
    constexpr uint32_t kDfbPartials = get_arg(args::dfb_partials);

    const auto a = TensorAccessor(tensor::a);
    const auto b = TensorAccessor(tensor::b);
    const auto bias = TensorAccessor(tensor::bias);
    const auto out = TensorAccessor(tensor::out);

    u::matmul_init<A, B>(kDfbA, kDfbB, kDfbOut);

    u::Storage<A> a_storage(kDfbA);
    u::Storage<B> b_storage(kDfbB);
    u::Storage<Bias> bias_storage(kDfbBias);
    u::Storage<Out> partials_storage(kDfbPartials);
    u::Storage<Out> out_storage(kDfbOut);

    const u::LogicalCoord me = u::LogicalCoord::this_core();
    const u::LogicalMcast my_row{u::LogicalCoord::yx(me.y, 0), u::Extent::hw(1, kGridWidth)};
    const u::LogicalMcast my_column{u::LogicalCoord::yx(0, me.x), u::Extent::hw(kGridHeight, 1)};

    u::ComputeBlock bias_row = u::noc_load<0>(bias_storage, bias, me.x).wait();

    u::Accumulator<Out, u::AccumulatorMode::Dst> total(partials_storage, out_storage);
    total.clear();

    for (uint32_t k = 0; k < kKBlocks; ++k) {
        const bool last = (k == kKBlocks - 1);

        u::ComputeBlock a_block = u::noc_load<0>(a_storage, my_row, a, me.y * kKBlocks + k).wait();
        u::ComputeBlock b_block = u::noc_load<1>(b_storage, my_column, b, me.x * kKBlocks + k).wait();

        u::Block result =
            total.accumulate(u::matmul(a_block, b_block), last, [&](auto sum) { return sum.bias(bias_row).relu(); });

        if (last) {
            u::noc_store<0>(std::move(result), out, me.y * kGridWidth + me.x);
        }
    }
}
