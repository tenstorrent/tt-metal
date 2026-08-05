// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/hierarchical_gather) -- NOT the op.
//
// The compute half of the combine ONLY: fold a gathered ring of fp32 partials into
// one accumulator tile per row, then (on whoever finalizes) 1/rms = rsqrt(sum/W +
// eps) and hand the result to the writer.  Exactly the op's `is_root` branch,
// re-spelled so the FAN-IN and WHICH CORE finalizes are parameters:
//
//   VARIANT 0 FLAT     : the root folds GROUP_SIZE per row, finalizes, hands off.
//   VARIANT 1 TREE     : every SUB-ROOT folds M = GROUP_SIZE/K per row into
//                        cb_subroot_out (NO finalize -- it is a partial sum);
//                        the root then folds K per row, finalizes, hands off.
//   VARIANT 2 ROWSPLIT : every WORKER folds GROUP_SIZE per row for ITS row range
//                        and finalizes it (the finalize is per-row and therefore
//                        parallelizes over rows exactly like the fold).
//
// The fold is the op's Perf-1/D16 chain: CopyTile each contiguous partial into DEST
// and PACK-ACCUMULATE it into one fp32 accumulator tile (L1Accumulation::SeedFirst
// -- first tile a plain pack, every later one a pack-add).  The running sum lives in
// the fp32 CB, not in a DEST register that is 16-bit at fp32_dest_acc_en == False.
//
// RING TAIL.  The writer pushes the WHOLE FANIN * BLOCK_ROWS ring every round (see
// the writer's ring-discipline note), so after folding the rows this core owns it
// pops the unused tail to return the ring to its base.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace ckl = compute_kernel_lib;

// The op's Lamp-L6b raw-LLK substitution, carried verbatim so the finalize this
// bench measures is the op's finalize.  `rsqrt_tile` hard-codes VectorMode::RC and
// exposes no seam; the stat is a REDUCE_ROW column vector living in faces 0 and 2,
// which is exactly VectorMode::C.
#ifdef TRISC_MATH
template <bool legacy_compat = false, bool FAST_APPROX = false>
ALWI void rsqrt_tile_col(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, FAST_APPROX, legacy_compat),
        idst,
        VectorMode::C);
}
#endif

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
    constexpr uint32_t W_MAX = get_compile_time_arg_val(4);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(6);
    constexpr uint32_t RSQRT_COL = get_compile_time_arg_val(7);

    constexpr uint32_t M = (VARIANT == 1) ? (GROUP_SIZE / K) : GROUP_SIZE;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t is_subroot = get_arg_val<uint32_t>(2);
    const uint32_t my_slot = get_arg_val<uint32_t>(3);

    // Cores with nothing to compute (inactive, or a plain member of the group)
    // return before any CB or LLK state is touched -- the op does the same for its
    // inactive cores.
    const bool participates = (num_rows != 0) && ((VARIANT == 0)   ? (is_root != 0)
                                                  : (VARIANT == 1) ? (is_subroot != 0)
                                                                   : (my_slot < W_MAX));
    if (!participates) {
        return;
    }

    // Every CB in this bench is fp32, so srcA == srcB at boot and the per-call
    // DataFormatReconfig::Enabled covers the stage-2 ring for free.  Booting off
    // cb_partials_gathered alone is what lets the FLAT / ROWSPLIT variants not
    // allocate the tree's stage-2 CBs at all.
    compute_kernel_hw_startup(cb_partials_gathered, cb_partials_gathered, cb_stat_handoff);

    auto finalize = [](uint32_t dst) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, INV_W_BITS);
        add_unary_tile(dst, EPS_BITS);
        rsqrt_tile_init();
        if constexpr (RSQRT_COL != 0) {
            MATH((rsqrt_tile_col(dst)));
        } else {
            rsqrt_tile(dst);
        }
    };

    // (OneUpfront, OneAtEnd) is the policy pair L1 accumulation requires: the whole
    // call pins ONE output tile.  Hence one call per row.
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

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

        if constexpr (VARIANT == 0) {
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
                for (uint32_t i = 0; i < rows; ++i) {
                    ckl::transform_in_place(cb_row_stat, finalize);
                }
            }
            MaybeDeviceZoneScope("compute_stat_handoff");
            ckl::copy<ckl::input(cb_row_stat), ckl::output(cb_stat_handoff)>(ckl::EltwiseShape::tiles(rows));
        } else if constexpr (VARIANT == 1) {
            {
                MaybeDeviceZoneScope("compute_subroot_sum");
                for (uint32_t r = 0; r < rows; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(M),
                        ckl::CopyTile<ckl::input(cb_partials_gathered)>{},
                        ckl::PackTile<FOLD_TO_SUBROOT_OUT>{});
                }
                if (rows < BLOCK_ROWS) {
                    cb_pop_front(cb_partials_gathered, M * (BLOCK_ROWS - rows));
                }
            }
            if (is_root != 0) {
                {
                    MaybeDeviceZoneScope("compute_root_sum");
                    for (uint32_t r = 0; r < rows; ++r) {
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::tiles(K),
                            ckl::CopyTile<ckl::input(cb_stage2)>{},
                            ckl::PackTile<FOLD_TO_ROW_STAT>{});
                    }
                    if (rows < BLOCK_ROWS) {
                        cb_pop_front(cb_stage2, K * (BLOCK_ROWS - rows));
                    }
                }
                {
                    MaybeDeviceZoneScope("compute_root_finalize");
                    for (uint32_t i = 0; i < rows; ++i) {
                        ckl::transform_in_place(cb_row_stat, finalize);
                    }
                }
                MaybeDeviceZoneScope("compute_stat_handoff");
                ckl::copy<ckl::input(cb_row_stat), ckl::output(cb_stat_handoff)>(ckl::EltwiseShape::tiles(rows));
            }
        } else {
            // ROWSPLIT: recompute the SAME row split the writer used.
            const uint32_t w = (W_MAX < rows) ? W_MAX : rows;
            if (my_slot >= w) {
                continue;  // not a worker this round
            }
            const uint32_t base = rows / w, extra = rows % w;
            const uint32_t my_rows = base + (my_slot < extra ? 1u : 0u);
            {
                MaybeDeviceZoneScope("compute_root_sum");
                for (uint32_t r = 0; r < my_rows; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(GROUP_SIZE),
                        ckl::CopyTile<ckl::input(cb_partials_gathered)>{},
                        ckl::PackTile<FOLD_TO_ROW_STAT>{});
                }
                if (my_rows < BLOCK_ROWS) {
                    cb_pop_front(cb_partials_gathered, GROUP_SIZE * (BLOCK_ROWS - my_rows));
                }
            }
            {
                MaybeDeviceZoneScope("compute_root_finalize");
                for (uint32_t i = 0; i < my_rows; ++i) {
                    ckl::transform_in_place(cb_row_stat, finalize);
                }
            }
            MaybeDeviceZoneScope("compute_stat_handoff");
            ckl::copy<ckl::input(cb_row_stat), ckl::output(cb_stat_handoff)>(ckl::EltwiseShape::tiles(my_rows));
        }
    }
}
