// SPDX-License-Identifier: Apache-2.0
//
// A two-stage reduction tree. Every core reduces its own block, then each COLUMN
// gathers its cores' partials into the column's top core, which reduces the
// gathered set again and writes the answer.
//
// The gather is one noc_core_write per core: every core in the column pushes its
// partial into row 0's tmp1 buffer at its own row's offset, and row 0 waits for
// the whole column before reducing. Every core runs the same statement -- writers
// and reader take their sides from their own coordinates.
//
// The synchronize_cores() before each gather is what makes the loop safe to
// repeat: it establishes that every root has finished collecting the previous
// round before anyone writes into the next one. See noc_core_write.
//
// Compile-time args:
//   0        num_blocks
//   1        in_ht                    this core's block is in_ht x in_wt tiles
//   2        in_wt
//   3        num_cores_y              cores in a column: the gather's height, and
//                                     so the number of writers row 0 collects
//   4..      TensorAccessorArgs for in0, then in1, then out
//
// Runtime args (identical on all three kernels):
//   0        in0 base address
//   1        in1 base address
//   2        out base address

#include <tt/unified>

namespace u = tt::unified;

constexpr uint32_t kCbIn0 = 0;
constexpr uint32_t kCbTmp0 = 1;
constexpr uint32_t kCbTmp1 = 2;
constexpr uint32_t kCbScaler = 3;
constexpr uint32_t kCbOut = 16;

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t in_ht = get_compile_time_arg_val(1);
    constexpr uint32_t in_wt = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(3);

    // Both stages collapse the ROW axis, so each leaves one valid row per tile
    // column. Stage 1 folds this core's block; stage 2 folds the column's stack of
    // stage-1 results, which the gather has laid out as num_cores_y x in_wt.
    constexpr auto kAxis = u::ReduceAxis::Rows;
    using PerCore = u::ReduceGeometry<in_ht, in_wt>;
    using PerColumn = u::ReduceGeometry<num_cores_y, in_wt>;
    constexpr uint32_t reduced_tiles_per_block = PerCore::out_tiles(kAxis);

    // TensorAccessor compile-time args, laid out in0, in1, out -- starting AFTER
    // the four scalars above.
    constexpr auto in0_args = TensorAccessorArgs<4>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    u::compute_init(kCbIn0, kCbOut);

    u::Storage in0_storage(kCbIn0, PerCore::num_tiles);
    u::Storage scaler_storage(kCbScaler, 1);
    u::Storage tmp0_storage(kCbTmp0, reduced_tiles_per_block);
    u::Storage tmp1_storage(kCbTmp1, reduced_tiles_per_block * num_cores_y);
    u::Storage out_storage(kCbOut, reduced_tiles_per_block);

    const auto in0 = TensorAccessor(in0_args, in0_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // Once, before any reduction: every reduce_tile re-reads this page.
    u::fill_reduce_scaler<1>(scaler_storage);

    const auto this_core = u::LogicalCoord::this_core();

    // Row 0 of this core's column does the gathering.
    const u::LogicalCoord root{0, this_core.x};

    // Where this core's partial lands in the gather buffer. In BYTES: the offset
    // goes straight onto a write pointer, and each core owns one slice of
    // reduced_tiles_per_block pages.
    const uint32_t byte_offset = this_core.y * reduced_tiles_per_block * u::cb_page_bytes(kCbTmp1);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        u::ComputeBlock a = u::noc_load<1>(in0_storage, in0, b).wait();

        u::Block per_core_sum = tmp0_storage.store(u::reduce_sum<PerCore, kAxis>(a, scaler_storage));

        // Nobody writes the next round until every root has drained this one.
        u::synchronize_cores<0>();

        // Every core writes; the root also receives, and waits for the column.
        u::ComputeBlock all_per_core_sums =
            u::noc_core_write<0>(tmp1_storage, std::move(per_core_sum), root, true, byte_offset).wait(num_cores_y);

        if (this_core == root) {
            u::Block result = out_storage.store(u::reduce_sum<PerColumn, kAxis>(all_per_core_sums, scaler_storage));
            u::noc_store<0>(std::move(result), out, b);
        }
    }
}
