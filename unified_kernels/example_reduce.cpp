// SPDX-License-Identifier: Apache-2.0

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

constexpr uint32_t kNumCores = 4;
constexpr auto kAxis = u::ReduceAxis::Rows;

using In = u::Shape<4, 2>;
using Partial = u::reduce_shape<In, kAxis>;
using Gathered = u::Shape<kNumCores * Partial::rows, Partial::cols>;

void kernel_main() {
    constexpr uint32_t kDfbIn = get_arg(args::dfb_in);
    constexpr uint32_t kDfbScaler = get_arg(args::dfb_scaler);
    constexpr uint32_t kDfbPartial = get_arg(args::dfb_partial);
    constexpr uint32_t kDfbGathered = get_arg(args::dfb_gathered);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);

    const auto in = TensorAccessor(tensor::in);
    const auto out = TensorAccessor(tensor::out);

    u::compute_init(kDfbIn, kDfbOut);

    u::Storage<In> in_storage(kDfbIn);
    u::Storage<u::Shape<1, 1>> scaler_storage(kDfbScaler);
    u::Storage<Partial> partial_storage(kDfbPartial);
    u::Storage<Gathered> gathered_storage(kDfbGathered);
    u::Storage<Partial> out_storage(kDfbOut);

    u::ComputeBlock scaler = u::fill_reduce_scaler<1>(scaler_storage);

    const u::LogicalCoord me = u::LogicalCoord::this_core();
    const u::LogicalCoord root = u::LogicalCoord::yx(0, 0);
    const uint32_t my_slot = me.y * Partial::num_entries * u::dfb_entry_bytes(kDfbGathered);

    u::ComputeBlock block = u::noc_load<0>(in_storage, in, me.y).wait();
    u::Block partial = partial_storage.store(u::reduce_sum<kAxis>(block, scaler));

    u::ComputeBlock all_partials =
        u::noc_core_write<0>(gathered_storage, std::move(partial), root, true, my_slot).wait(kNumCores);

    if (me == root) {
        u::Block result = out_storage.store(u::reduce_sum<kAxis>(all_partials, scaler));
        u::noc_store<1>(std::move(result), out, 0);
    }
}
