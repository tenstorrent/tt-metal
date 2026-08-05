// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/hierarchical_gather_r2) -- NOT the op.
//
// The compute half of the cross-core combine ONLY, re-spelled so that WHICH CORE folds
// and WHICH CORE finalizes are parameters (K slot chunks x m row subsets).
//
// The baseline spelling is the op's CURRENT root chain, carried VERBATIM so that the
// measured delta is against what the op does today and not against what it did before
// Perf 1:
//   * D16  the fold is ONE streaming chain call per row -- CopyTile each contiguous
//          partial into DEST and PACK-ACCUMULATE it into the row's accumulator tile
//          (L1Accumulation::SeedFirst: first tile a plain pack, every later one a
//          pack-add).  The running sum lives in the fp32 CB, not in a DEST register that
//          is 16-bit at fp32_dest_acc_en == False.
//   * D17  the finalize is the raw-sfpi COLUMN-SCOPED chain (`StatFinalize`), which folds
//          *(1/W) and +eps into ONE pass over DEST and walks only the even lanes of faces
//          0/2.  Copied verbatim from kernels/rms_norm_compute.cpp, including the reason.
//   * D19  the finalize READS the accumulator and WRITES cb_stat_handoff in ONE chain --
//          there is no separate handoff copy stage left to measure.
//
// VARIANT 0 FLAT : the root folds GROUP_SIZE per row for ALL rows, then finalizes ALL rows.
// VARIANT 1 GRID : gatherer g(j,w) folds M = GROUP_SIZE/K partials per row for ITS row
//                  subset w.  At K > 1 the j != 0 gatherers pack into cb_subroot_out (a
//                  partial sum -- NO finalize) and g(0,w) folds the K forwarded partials
//                  from cb_stage2 before finalizing subset w.  At K == 1 stage 1 packs
//                  straight into cb_row_stat.  The finalize is per-row and therefore
//                  parallelises over row subsets exactly like the fold.
//
// RING TAIL.  The writer pushes the WHOLE M * RPW ring every round (see its
// ring-discipline note), so after folding the rows this core owns it pops the unused tail
// to return the ring to its base.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

// ---------------------------------------------------------------------------------------
// RAW-LLK, CARRIED VERBATIM FROM THE OP (kernels/rms_norm_compute.cpp, Perf 1 / D17).
// Not a new bypass: the baseline this bench measures against IS this code, so re-spelling
// the finalize with helper calls would make the baseline slower than the op and inflate
// every candidate's speedup.  The op's own justification, in one line: mul_unary_tile /
// add_unary_tile / rsqrt_tile all hard-code VectorMode::RC and expose no VectorMode seam,
// and column parity is the SFPU's INNER walk axis so ITERATIONS cannot reach it; the stat
// is a REDUCE_ROW column vector whose only consumer reads column 0.
// ---------------------------------------------------------------------------------------
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"              // ckernel::sfpu::_calculate_sqrt_body_
#include "ckernel_sfpu_binop_with_unary.h"  // ckernel::sfpu::Converter::as_float

template <int STRIDE, int ITERS>
sfpi_inline void rms_stat_scale_body(uint32_t inv_w_bits, uint32_t eps_bits) {
    const sfpi::vFloat iw = ckernel::sfpu::Converter::as_float(inv_w_bits);
    const sfpi::vFloat ep = ckernel::sfpu::Converter::as_float(eps_bits);
    for (int i = 0; i < ITERS; ++i) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * iw + ep;
        sfpi::dst_reg += STRIDE;
    }
}

template <int STRIDE, int ITERS>
sfpi_inline void rms_stat_rsqrt_body() {
    for (int i = 0; i < ITERS; ++i) {
        sfpi::vFloat t =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}

ALWI void stat_scale_col_skip(uint32_t idst, uint32_t inv_w_bits, uint32_t eps_bits) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<2, 4>, idst, VectorMode::C, inv_w_bits, eps_bits);
}
ALWI void rsqrt_tile_col_skip(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<2, 4>, idst, VectorMode::C);
}
#endif  // TRISC_MATH

template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
ALWI void stat_finalize_payload(uint32_t dst) {
    MATH((stat_scale_col_skip(dst, RMS_INV_W, RMS_EPS)));
    MATH((rsqrt_tile_col_skip(dst)));
}

template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
struct StatFinalize : compute_kernel_lib::UnaryOp<StatFinalize<RMS_INV_W, RMS_EPS>, compute_kernel_lib::Dst::D0> {
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { stat_finalize_payload<RMS_INV_W, RMS_EPS>(slot_offset); }
};

namespace {
constexpr uint32_t cb_partials_gathered = 11;
constexpr uint32_t cb_stage2 = 12;
constexpr uint32_t cb_subroot_out = 13;
constexpr uint32_t cb_row_stat = 14;
constexpr uint32_t cb_stat_handoff = 15;
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t MROW = get_compile_time_arg_val(4);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(6);

    constexpr uint32_t M = (VARIANT == 1) ? (GROUP_SIZE / K) : GROUP_SIZE;
    constexpr uint32_t RPW = (VARIANT == 1) ? ((BLOCK_ROWS + MROW - 1) / MROW) : BLOCK_ROWS;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t my_slot = get_arg_val<uint32_t>(2);

    const uint32_t my_chunk = (VARIANT == 1) ? (my_slot / M) : 0;
    const uint32_t my_pos = (VARIANT == 1) ? (my_slot % M) : 0;
    const bool is_gatherer = (VARIANT == 1) && (my_pos < MROW);
    const uint32_t my_w = my_pos;
    const bool is_stage2 = is_gatherer && (my_chunk == 0);

    // Cores with nothing to compute (inactive, or a plain member of the group) return
    // before any CB or LLK state is touched -- the op does the same.
    const bool participates = (num_rows != 0) && ((VARIANT == 0) ? (is_root != 0) : is_gatherer);
    if (!participates) {
        return;
    }

    // Every CB in this bench is fp32, so srcA == srcB at boot and the per-call
    // DataFormatReconfig::Enabled covers the stage-2 ring for free.
    compute_kernel_hw_startup(cb_partials_gathered, cb_partials_gathered, cb_stat_handoff);

    // (OneUpfront, OneAtEnd) is the policy pair L1 accumulation requires: the whole call
    // pins ONE output tile.  Hence one call per row.
    constexpr auto FOLD_TO_ROW_STAT = ckl::output(
        cb_row_stat,
        ckl::ReservePolicy::OneUpfront,
        ckl::PushPolicy::OneAtEnd,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::SeedFirst);
    constexpr auto FOLD_TO_SUBROOT_OUT = ckl::output(
        cb_subroot_out,
        ckl::ReservePolicy::OneUpfront,
        ckl::PushPolicy::OneAtEnd,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::SeedFirst);
    // At K == 1 stage 1 IS the whole fold, so it packs straight into the accumulator.
    constexpr auto STAGE1_OUT = (K > 1) ? FOLD_TO_SUBROOT_OUT : FOLD_TO_ROW_STAT;

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

        if constexpr (VARIANT == 0) {
            // ---- FLAT: the op's current root chain, verbatim ----------------------
            {
                MaybeDeviceZoneScope("compute_root_sum");
                for (uint32_t r = 0; r < rows; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(GROUP_SIZE),
                        ckl::CopyTile<ckl::input(cb_partials_gathered)>{},
                        ckl::PackTile<FOLD_TO_ROW_STAT>{});
                }
                if (rows < BLOCK_ROWS) {
                    cb_pop_front(cb_partials_gathered, GROUP_SIZE * (BLOCK_ROWS - rows));
                }
            }
            {
                MaybeDeviceZoneScope("compute_root_finalize");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(rows),
                    ckl::CopyTile<ckl::input(cb_row_stat)>{},
                    StatFinalize<INV_W_BITS, EPS_BITS>{},
                    ckl::PackTile<ckl::output(cb_stat_handoff)>{});
            }
        } else {
            // ---- GRID: recompute the SAME row split the writer used ---------------
            const uint32_t w_eff = (MROW < rows) ? MROW : rows;
            if (my_w >= w_eff) {
                continue;  // not a gatherer this round; the writer skipped my ring too
            }
            const uint32_t base = rows / w_eff, extra = rows % w_eff;
            const uint32_t my_rows = base + (my_w < extra ? 1u : 0u);
            {
                MaybeDeviceZoneScope("compute_stage1_fold");
                for (uint32_t r = 0; r < my_rows; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(M),
                        ckl::CopyTile<ckl::input(cb_partials_gathered)>{},
                        ckl::PackTile<STAGE1_OUT>{});
                }
                if (my_rows < RPW) {
                    cb_pop_front(cb_partials_gathered, M * (RPW - my_rows));
                }
            }
            if constexpr (K > 1) {
                if (!is_stage2) {
                    continue;
                }
                MaybeDeviceZoneScope("compute_stage2_fold");
                for (uint32_t r = 0; r < my_rows; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(K),
                        ckl::CopyTile<ckl::input(cb_stage2)>{},
                        ckl::PackTile<FOLD_TO_ROW_STAT>{});
                }
                if (my_rows < RPW) {
                    cb_pop_front(cb_stage2, K * (RPW - my_rows));
                }
            }
            {
                MaybeDeviceZoneScope("compute_root_finalize");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(my_rows),
                    ckl::CopyTile<ckl::input(cb_row_stat)>{},
                    StatFinalize<INV_W_BITS, EPS_BITS>{},
                    ckl::PackTile<ckl::output(cb_stat_handoff)>{});
            }
        }
    }
}
