// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    DataflowBuffer dfb_recipsumexps_obj(dfb::recip_sum_exps);
    DataflowBuffer dfb_max_obj(dfb::max);

    compute_kernel_hw_startup(dfb::in0, dfb::max_scaler, dfb::out0);

    constexpr uint32_t onetile = 1;

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing and avoid
    // force-unrolling the per-Ht loops (see moreh_softmax_w_large.cpp for the LTO/addrmod rationale).
    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);

    for (std::uint32_t n = 0; n < N; ++n) {
        // SOFTMIN shift: use min(x) instead of max(x). A max-based shift is +inf for a row
        // holding +inf, and exp(max(x) - x) then saturates the whole row (inf / inf = NaN, and the
        // +inf lane itself is inf - inf = NaN) -- see #56371. Shifted by min(x) instead, a row that
        // holds +inf still has a finite minimum, so the +inf lanes exponentiate to 0 and the finite
        // lanes keep their relative weights -- that is torch's distribution. (A row whose minimum is
        // itself -inf is the documented divergent case: the whole mass lands on that lane.)
        // The padding lanes of the last tile are masked to +inf: a 0-masked pad (what the MAX path
        // wants) would win the MIN on an all-positive row.
#ifdef SOFTMAX
        // find max
        if (Ht == 1) {
            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(0, 0, /*pop0=*/1, /*popm=*/0);

            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb::tmp, dfb::max_scaler, dfb::max>(
                ckl::ReduceInputBlockShape::single());
        } else {
            // Phase 1: Reduce Ht-1 tiles
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb::in0, dfb::max_scaler, dfb::max>(
                ckl::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(0, 0, /*pop0=*/1, /*popm=*/0);

            // Phase 2: Reduce final masked tile with accumulation
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb::tmp, dfb::max_scaler, dfb::max>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(dfb::max, 1));  // iteration=1, reload from dfb::max
        }
#else
        // MIN is SFPU-only: a Fast fp32 MIN does not exist, so an fp32 reduce input needs
        // ReduceFp32Mode::Accurate; bf16 ignores fp32_mode entirely. Deduce it from the compile-time
        // formats of the buffers THIS reduce can read (unpack_src_format[] is a constexpr descriptor
        // array, so it is usable as a template argument):
        //   * dfb::in0 -- the leading tiles (the large kernels' phase 1 reduces them straight out of it);
        //   * dfb::tmp -- the masked last tile; the factories declare TMP with intermed_data_format,
        //                 which is Float32 whenever fp32_dest_acc_en is enabled, even for a bf16 tensor.
        // Deriving the mode from dfb::in0 alone picks Fast for bf16 + fp32 accumulation and trips the
        // MIN static_assert in reduce_helpers_compute.inl.
        constexpr DataFormat kSoftminInFormat = static_cast<DataFormat>(unpack_src_format[dfb::in0]);
        constexpr DataFormat kSoftminScratchFormat = static_cast<DataFormat>(unpack_src_format[dfb::tmp]);
        constexpr ReduceFp32Mode kFp32Mode =
            (kSoftminInFormat == DataFormat::Float32 || kSoftminScratchFormat == DataFormat::Float32)
                ? ReduceFp32Mode::Accurate
                : ReduceFp32Mode::Fast;
        // find min
        if (Ht == 1) {
            mask_posinf_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(0, 0, /*pop0=*/1, /*popm=*/0);

            ckl::reduce<
                PoolType::MIN,
                ReduceDim::REDUCE_COL,
                dfb::tmp,
                dfb::max_scaler,
                dfb::max,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kFp32Mode>(
                ckl::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce the leading tiles straight out of dfb::in0 -- a streaming reduce that needs
            // no staging buffer -- and park the partial min in dfb::add, which is otherwise free until the
            // exp-sum accumulation below.
            ckl::reduce<
                PoolType::MIN,
                ReduceDim::REDUCE_COL,
                dfb::in0,
                dfb::max_scaler,
                dfb::add,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kFp32Mode>(
                ckl::ReduceInputBlockShape::col(Ht - 1));

            // Phase 2: mask the last tile's padding lanes to +inf into dfb::tmp and reduce that single
            // tile into dfb::exps. A 0-masked pad (what the MAX path wants) wins the MIN on every
            // all-positive row, so the mask polarity has to flip with the pool type.
            mask_posinf_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(0, 0, /*pop0=*/1, /*popm=*/0);
            ckl::reduce<
                PoolType::MIN,
                ReduceDim::REDUCE_COL,
                dfb::tmp,
                dfb::max_scaler,
                dfb::exps,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kFp32Mode>(
                ckl::ReduceInputBlockShape::single());

            // Combine the two partial minima into dfb::max, where the rest of the kernel reads the row
            // statistic. Both operands are reduced tiles, and the reduce lane is the same lane in both,
            // so the element-wise MIN is exact there. Neither operand is the destination, so the combine
            // never has to free a slot in a DFB it is still reading.
            ckl::binary_sfpu<
                ckl::BinaryMin<>,
                ckl::input(dfb::add, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig),
                ckl::input(dfb::exps, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig),
                ckl::output(dfb::max, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, moreh_data_format_reconfig)>(
                ckl::IterationShape::one_tile());
        }
#endif

        for (std::uint32_t h = 0; h < Ht; h += onetile) {
            // compute exp(x - max(x))
            sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);
            if (h == Ht - 1) {
#ifdef SOFTMAX
                exp_tile_and_mask_tile_to_dfb<dfb::tmp, dfb::mask, dfb::exps>(
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#else
                rexp_tile_and_mask_tile_to_dfb<dfb::tmp, dfb::mask, dfb::exps>(
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#endif
            } else {
#ifdef SOFTMAX
                exp_tile_to_dfb<dfb::tmp, dfb::exps>();
#else
                rexp_tile_to_dfb<dfb::tmp, dfb::exps>();
#endif
            }

            if (h == 0) {
                copy_tile_to_dfb<dfb::exps, dfb::add>();
            } else {
                add_tiles_to_dfb<dfb::add, dfb::exps, dfb::add>();
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb::add,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::single(),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        // compute 1/sum(exp(x)) - pop tile after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb::add,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::single(),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // step 3, compute final result
        for (std::uint32_t h = 0; h < Ht; h += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_rows_to_dfb<dfb::tmp, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // logsoftmin not implemented
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_dfb<dfb::tmp, dfb::exps>();

            mul_tiles_bcast_rows_to_dfb<dfb::exps, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_rows_to_dfb<dfb::in0, dfb::max, dfb::tmp>(0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_dfb<dfb::tmp, dfb::exps>();

            mul_tiles_bcast_rows_to_dfb<dfb::exps, dfb::recip_sum_exps, dfb::out0>(0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
}
