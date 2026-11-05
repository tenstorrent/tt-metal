// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: with fp32_dest_acc_en this kernel must be built at -O3 (the program factory sets the compute
// KernelSpec opt_level to O3, matching legacy). At -O2 GCC fails to constant-fold the LLK addrmod SETC16
// inline-asm "n" immediate in this larger fp32 TU and JIT aborts with "impossible constraint in 'asm'".
// At O3 it folds and no source workaround is needed.

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
    (kSoftminInFormat == DataFormat::Float32 || kSoftminScratchFormat == DataFormat::Float32) ? ReduceFp32Mode::Accurate
                                                                                              : ReduceFp32Mode::Fast;
// The MIN reduce has an SFPU fold only for Int32, Float16_b and Accurate Float32 on WH/BH
// (is_sfpu_reduce_path in reduce_helpers_common.hpp; the FPU has no min pool), and the
// reduce helpers static_assert on any other MIN. Every buffer the min-shift path reduces out of
// has to qualify. A Bfp8_b tensor without fp32 accumulation does not (dfb::in0, and the TMP/staging
// buffers declared with intermed_data_format, are all Bfp8_b there), so SOFTMIN keeps the
// max-shifted path it used before for that configuration. Bfp8_b cannot carry an inf through its
// shared exponent anyway, so the +-inf rows the min shift exists for do not reach this kernel in it.
constexpr bool kSoftminMinShiftSupported =
    is_sfpu_reduce_path<PoolType::MIN, ReduceDim::REDUCE_ROW, kSoftminInFormat, kFp32Mode>() &&
    is_sfpu_reduce_path<PoolType::MIN, ReduceDim::REDUCE_ROW, kSoftminScratchFormat, kFp32Mode>();

// Finds the per-row shift into dfb::max: max(x) for SOFTMAX (and for SOFTMIN when the
// buffers above have no MIN fold), min(x) for SOFTMIN otherwise. kShiftPool is spelled through
// UseMinShift so the branch that is not taken is never instantiated: kernel_main is not a
// template, so a plain if constexpr there would still instantiate the MIN reduce and trip its
// static_assert.
template <
    bool UseMinShift,
    typename DFBIn0,
    typename DFBMask,
    typename DFBTmp,
    typename DFBExps,
    typename DFBAdd,
    typename DFBMax,
    typename DFBMaxScaler>
ALWI void find_shift(
    std::uint32_t Wt,
    DFBIn0& dfb_in0_obj,
    DFBMask& dfb_mask_obj,
    DFBTmp& dfb_tmp_obj,
    DFBExps& dfb_exps_obj,
    DFBAdd& dfb_add_obj,
    DFBMax& dfb_max_obj,
    DFBMaxScaler& dfb_max_scaler_obj) {
    constexpr PoolType kShiftPool = UseMinShift ? PoolType::MIN : PoolType::MAX;
    if constexpr (!UseMinShift) {
        // find max
        if (Wt == 1) {
            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(
                dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb::tmp, dfb::max_scaler, dfb::max>(
                dfb_tmp_obj, dfb_max_scaler_obj, dfb_max_obj, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
#ifdef ARCH_QUASAR
            // Quasar has no MAX + REDUCE_ROW accumulator reload (it needs a within-face transpose that
            // copy_tile_to_dst_init_short rejects there; see reduce_helpers_compute.inl). Reduce the leading
            // tiles and the masked last tile separately, into dfb::add and dfb::exps (both free until the
            // exp-sum below), and combine the partial maxima into dfb::max, as the min-shift path below does.
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb::in0, dfb::max_scaler, dfb::add>(
                dfb_in0_obj, dfb_max_scaler_obj, dfb_add_obj, compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));
            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(
                dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop=*/1, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb::tmp, dfb::max_scaler, dfb::exps>(
                dfb_tmp_obj, dfb_max_scaler_obj, dfb_exps_obj, compute_kernel_lib::ReduceInputBlockShape::single());
            moreh_retarget_packer(dfb::max);
            ckl::binary_sfpu<
                ckl::BinaryMax<>,
                ckl::input(dfb::add, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig),
                ckl::input(dfb::exps, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig),
                ckl::output(
                    dfb::max, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, moreh_data_format_reconfig)>(
                ckl::IterationShape::one_tile(), dfb_add_obj, dfb_exps_obj, dfb_max_obj);
#else
            // Phase 1: reduce Wt-1 full tiles into dfb::max (no accumulation, first call).
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb::in0, dfb::max_scaler, dfb::max>(
                dfb_in0_obj, dfb_max_scaler_obj, dfb_max_obj, compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile and continue reducing into dfb::max via Accumulate.
            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(
                dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop=*/1, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb::tmp, dfb::max_scaler, dfb::max>(
                dfb_tmp_obj,
                dfb_max_scaler_obj,
                dfb_max_obj,
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate(dfb_max_obj, /*iter=*/1));
#endif
        }
    } else {
        // find min
        if (Wt == 1) {
            mask_posinf_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(
                dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<
                kShiftPool,
                ReduceDim::REDUCE_ROW,
                dfb::tmp,
                dfb::max_scaler,
                dfb::max,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kFp32Mode>(
                dfb_tmp_obj, dfb_max_scaler_obj, dfb_max_obj, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce the leading tiles straight out of dfb::in0 -- a streaming reduce that needs
            // no staging buffer -- and park the partial min in dfb::add, which is otherwise free until the
            // exp-sum accumulation below.
            compute_kernel_lib::reduce<
                kShiftPool,
                ReduceDim::REDUCE_ROW,
                dfb::in0,
                dfb::max_scaler,
                dfb::add,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kFp32Mode>(
                dfb_in0_obj, dfb_max_scaler_obj, dfb_add_obj, compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile's padding lanes to +inf into dfb::tmp and reduce that single
            // tile into dfb::exps. A 0-masked pad (what the MAX path wants) wins the MIN on every
            // all-positive row, so the mask polarity has to flip with the pool type.
            mask_posinf_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(
                dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop=*/1, /*popm=*/0);
            ckl::reduce<
                kShiftPool,
                ReduceDim::REDUCE_ROW,
                dfb::tmp,
                dfb::max_scaler,
                dfb::exps,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kFp32Mode>(dfb_tmp_obj, dfb_max_scaler_obj, dfb_exps_obj, ckl::ReduceInputBlockShape::single());

            // Combine the two partial minima into dfb::max, where the rest of the kernel reads the row
            // statistic. Both operands are reduced tiles, and the reduce lane is the same lane in both,
            // so the element-wise MIN is exact there. Neither operand is the destination, so the combine
            // never has to free a slot in a DFB it is still reading.
            moreh_retarget_packer(dfb::max);
            ckl::binary_sfpu<
                ckl::BinaryMin<>,
                ckl::input(dfb::add, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig),
                ckl::input(dfb::exps, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig),
                ckl::output(
                    dfb::max, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, moreh_data_format_reconfig)>(
                ckl::IterationShape::one_tile(), dfb_add_obj, dfb_exps_obj, dfb_max_obj);
        }
    }
}

void kernel_main() {
    DataflowBuffer dfb_recipsumexps_obj(dfb::recip_sum_exps);
    DataflowBuffer dfb_max_obj(dfb::max);
    DataflowBuffer dfb_in0_obj(dfb::in0);
    DataflowBuffer dfb_mask_obj(dfb::mask);
    DataflowBuffer dfb_max_scaler_obj(dfb::max_scaler);
    DataflowBuffer dfb_sum_scaler_obj(dfb::sum_scaler);
    DataflowBuffer dfb_tmp_obj(dfb::tmp);
    DataflowBuffer dfb_exps_obj(dfb::exps);
    DataflowBuffer dfb_add_obj(dfb::add);
    DataflowBuffer dfb_out0_obj(dfb::out0);

    compute_kernel_hw_startup(dfb::in0, dfb::max_scaler, dfb::out0);

    constexpr std::uint32_t onetile = 1;

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing.
    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    dfb_mask_obj.wait_front(onetile);
    dfb_max_scaler_obj.wait_front(onetile);
    dfb_sum_scaler_obj.wait_front(onetile);

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
        find_shift</*UseMinShift=*/false>(
            Wt, dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, dfb_exps_obj, dfb_add_obj, dfb_max_obj, dfb_max_scaler_obj);
#else
        find_shift<kSoftminMinShiftSupported>(
            Wt, dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, dfb_exps_obj, dfb_add_obj, dfb_max_obj, dfb_max_scaler_obj);
#endif

        for (uint32_t w = 0; w < Wt; ++w) {
            // compute exp(x - max(x))
            sub_tiles_bcast_cols_to_dfb<dfb::in0, dfb::max, dfb::tmp>(
                dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
            if (w == Wt - 1) {
#ifdef SOFTMAX
                exp_tile_and_mask_tile_to_dfb<dfb::tmp, dfb::mask, dfb::exps>(
                    dfb_tmp_obj,
                    dfb_mask_obj,
                    dfb_exps_obj,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#else
                rexp_tile_and_mask_tile_to_dfb<dfb::tmp, dfb::mask, dfb::exps>(
                    dfb_tmp_obj,
                    dfb_mask_obj,
                    dfb_exps_obj,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#endif
            } else {
#ifdef SOFTMAX
                exp_tile_to_dfb<dfb::tmp, dfb::exps>(dfb_tmp_obj, dfb_exps_obj);
#else
                rexp_tile_to_dfb<dfb::tmp, dfb::exps>(dfb_tmp_obj, dfb_exps_obj);
#endif
            }

            if (w == 0) {
                copy_tile_to_dfb<dfb::exps, dfb::add>(dfb_exps_obj, dfb_add_obj);
            } else {
                add_tiles_to_dfb<dfb::add, dfb::exps, dfb::add>(dfb_add_obj, dfb_exps_obj, dfb_add_obj);
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb::add,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            dfb_add_obj,
            dfb_sum_scaler_obj,
            dfb_recipsumexps_obj,
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
            ReduceDim::REDUCE_ROW,
            dfb::add,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            dfb_add_obj,
            dfb_sum_scaler_obj,
            dfb_recipsumexps_obj,
            ckl::ReduceInputBlockShape::single(),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // step 3, compute final result
        for (std::uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_bcast_cols_to_dfb<dfb::in0, dfb::max, dfb::tmp>(
                dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_cols_to_dfb<dfb::tmp, dfb::recip_sum_exps, dfb::out0>(
                dfb_tmp_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // logsoftmin not implemented
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_cols_to_dfb<dfb::in0, dfb::max, dfb::tmp>(
                dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_dfb<dfb::tmp, dfb::exps>(dfb_tmp_obj, dfb_exps_obj);

            mul_tiles_bcast_cols_to_dfb<dfb::exps, dfb::recip_sum_exps, dfb::out0>(
                dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_cols_to_dfb<dfb::in0, dfb::max, dfb::tmp>(
                dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_dfb<dfb::tmp, dfb::exps>(dfb_tmp_obj, dfb_exps_obj);

            mul_tiles_bcast_cols_to_dfb<dfb::exps, dfb::recip_sum_exps, dfb::out0>(
                dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
    // The mask and the reduce scalers are each a single tile pushed once by the reader and re-read on
    // every iteration, so they are waited once up front. Pop them here to leave the buffers balanced:
    // on Quasar the reader's DataflowBuffers wait for these pops when they go out of scope.
    dfb_mask_obj.pop_front(onetile);
    dfb_max_scaler_obj.pop_front(onetile);
    dfb_sum_scaler_obj.pop_front(onetile);
}
