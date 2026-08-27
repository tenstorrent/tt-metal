// SPDX-License-Identifier: Apache-2.0

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kNumCores = 4;
constexpr auto kAxis = u::ReduceAxis::Rows;

using In = u::Shape<4, 2>;
using Partial = u::reduce_shape<In, kAxis>;
using Gathered = u::Shape<kNumCores * Partial::rows, Partial::cols>;

void kernel_main() {
    constexpr uint32_t kCbIn = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t kCbScaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t kCbPartial = get_named_compile_time_arg_val("cb_partial");
    constexpr uint32_t kCbGathered = get_named_compile_time_arg_val("cb_gathered");
    constexpr uint32_t kCbOut = get_named_compile_time_arg_val("cb_out");

    const auto in = TensorAccessor(tensor::in));
    const auto out = TensorAccessor(tensor::out));

    u::compute_init(kCbIn, kCbOut);

    u::Storage<In> in_storage(kCbIn);
    u::Storage<u::Shape<1, 1>> scaler_storage(kCbScaler);
    u::Storage<Partial> partial_storage(kCbPartial);
    u::Storage<Gathered> gathered_storage(kCbGathered);
    u::Storage<Partial> out_storage(kCbOut);

    u::ComputeBlock scaler = u::fill_reduce_scaler<1>(scaler_storage);

    const u::LogicalCoord me = u::LogicalCoord::this_core();
    const u::LogicalCoord root = u::LogicalCoord::yx(0, 0);
    const uint32_t my_slot = me.y * Partial::num_pages * u::cb_page_bytes(kCbGathered);

    u::ComputeBlock block = u::noc_load<0>(in_storage, in, me.y).wait();
    u::Block partial = partial_storage.store(u::reduce_sum<kAxis>(block, scaler));

    u::ComputeBlock all_partials =
        u::noc_core_write<0>(gathered_storage, std::move(partial), root, true, my_slot).wait(kNumCores);

    if (me == root) {
        u::Block result = out_storage.store(u::reduce_sum<kAxis>(all_partials, scaler));
        u::noc_store<1>(std::move(result), out, 0);
    }
}
