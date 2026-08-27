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
// Compile-time args, all named:
//   num_blocks
//   in_ht, in_wt             this core's block is in_ht x in_wt tiles
//   num_cores_y              cores in a column: the gather's height, and so the
//                            number of writers row 0 collects
//   cb_<name> per buffer
//
// No runtime args: the tensors are bound, so their addresses ride along with the
// accessors.
//
// Columns are independent: column x reduces input block b * kCoreGridW + x and
// writes result block b * kCoreGridW + x, so `in0` and `out` both hold
// num_blocks x kCoreGridW blocks and the two indexings line up. The cores WITHIN a
// column still share a block, so their gather sums kCoreGridH copies of it.

#include <tt/unified/core>

namespace u = tt::unified;

// Which fold. Sum is the default because it is the only one the TWO-STAGE tree
// can do: stage 2 reduces stage-1 results whose non-result rows the packer zeroed,
// and zeros are harmless to a sum but not to a max (they win when every value is
// negative) or a mean (they are counted, dividing by 32x too many). Use
// RT_SINGLE_STAGE with those.
#if defined(RT_MAX)
#define RT_REDUCE(x) u::reduce_max<kAxis>(x, scaler)
#elif defined(RT_MEAN)
#define RT_REDUCE(x) u::reduce_mean<kAxis>(x, scaler)
#else
#define RT_REDUCE(x) u::reduce_sum<kAxis>(x, scaler)
#endif

void kernel_main() {
    constexpr uint32_t num_blocks = get_named_compile_time_arg_val("num_blocks");
    constexpr uint32_t in_ht = get_named_compile_time_arg_val("in_ht");
    constexpr uint32_t in_wt = get_named_compile_time_arg_val("in_wt");
    constexpr uint32_t num_cores_y = get_named_compile_time_arg_val("num_cores_y");

    // Both stages collapse the ROW axis, so each leaves one valid row per tile
    // column. Stage 1 folds this core's block; stage 2 folds the column's stack of
    // stage-1 results, which the gather has laid out as num_cores_y x in_wt.
    constexpr auto kAxis = u::ReduceAxis::Rows;

    constexpr uint32_t kCbIn0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t kCbTmp0 = get_named_compile_time_arg_val("cb_tmp0");
    constexpr uint32_t kCbTmp1 = get_named_compile_time_arg_val("cb_tmp1");
    constexpr uint32_t kCbScaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t kCbOut = get_named_compile_time_arg_val("cb_out");

    u::compute_init(kCbIn0, kCbOut);

    // Shapes, not page counts: `Gathered` is the stage-1 result stacked once per
    // core in the column, which is exactly stage 2's input shape -- so the
    // relationship the old `reduced_tiles_per_block * num_cores_y` asserted by hand
    // is now one the compiler derives.
    using In = u::Shape<in_ht, in_wt>;
    using Reduced = u::reduce_shape<In, kAxis>;
    using Gathered = u::Shape<num_cores_y * Reduced::rows, Reduced::cols>;

    u::Storage<In> in0_storage(kCbIn0);
    u::Storage<u::Shape<1, 1>> scaler_storage(kCbScaler);
    u::Storage<Reduced> tmp0_storage(kCbTmp0);
    u::Storage<Gathered> tmp1_storage(kCbTmp1);
    u::Storage<Reduced> out_storage(kCbOut);

    const auto in0 = TensorAccessor(tensor::in0);
    const auto out = TensorAccessor(tensor::out);

    // The scaler metal folds into every reduce_tile: 1 for sum and max, 1/N for a
    // mean. Getting this wrong turns a mean into a sum with nothing to say so.
#if defined(RT_MEAN)
    const uint32_t scaler_bits = u::bf16_pair(1.0f / static_cast<float>(u::ReduceGeometry<In>::elements(kAxis)));
#else
    const uint32_t scaler_bits = u::kReduceScalerOne;
#endif

    // KERNEL SCOPE, exactly like a fused bias: every reduce_tile re-reads this
    // page, and a ComputeBlock pops in its destructor -- here, the end of the
    // kernel. Inside the loop it would be popped after the first reduction and the
    // next one would wait forever for a refill nobody issues.
    u::ComputeBlock scaler = u::fill_reduce_scaler<1>(scaler_storage, scaler_bits);

    const auto this_core = u::LogicalCoord::this_core();

    // Row 0 of this core's column does the gathering.
    const u::LogicalCoord root = u::LogicalCoord::yx(0, this_core.x);

    // Where this core's partial lands in the gather buffer. In BYTES: the offset
    // goes straight onto a write pointer, and each core owns one slice of
    // Reduced::num_pages pages.
    const uint32_t byte_offset = this_core.y * Reduced::num_pages * u::cb_page_bytes(kCbTmp1);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        // Column x owns its own input block, the same index its result goes to.
        u::ComputeBlock a = u::noc_load<0>(in0_storage, in0, b * u::kCoreGridW + this_core.x).wait();

        u::Block per_core_sum = tmp0_storage.store(RT_REDUCE(a));

#if defined(RT_SINGLE_STAGE)
        // Stage 1 only -- no gather, no second fold. This is the shape that
        // isolates the reduction itself, and the only one max and mean are correct
        // in; see RT_REDUCE above.
        u::noc_store<1>(std::move(per_core_sum), out, b * u::kCoreGridW + this_core.x);
    }
}
#else
        // Nobody writes the next round until every root has drained this one.
        u::synchronize_cores<0>();

        // Every core writes; the root also receives, and waits for the column.
        u::ComputeBlock all_per_core_sums =
            u::noc_core_write<0>(tmp1_storage, std::move(per_core_sum), root, true, byte_offset).wait(num_cores_y);

        if (this_core == root) {
            u::Block result = out_storage.store(RT_REDUCE(all_per_core_sums));
            // Every column has a root, and they all finish block b together, so the
            // block index alone would have them all writing the same pages. Give
            // each column its own slot: `out` is num_blocks rows of kCoreGridW
            // results, so column x's Reduced::num_pages tiles land contiguously
            // at b * kCoreGridW + x. The width comes from the harness's core-grid
            // define, so it costs no compile-time arg.
            u::noc_store<1>(std::move(result), out, b * u::kCoreGridW + this_core.x);
        }
    }
}
#endif
