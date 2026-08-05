// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/slot_tree_gather) -- NOT the op.
//
// The compute half of the cross-core combine ONLY, re-spelled so that WHICH CORE folds
// WHICH slots is a parameter (an arity list over the SLOT axis, row split OFF).
//
// The baseline spelling is the op's CURRENT root chain, carried VERBATIM so the measured
// delta is against what the op does TODAY and not against what it did before Perf 2:
//
//   D22  THE FUSED ROOT CHAIN.  A tile-row's GATHER_SLOTS partials are accumulated
//        PAIRWISE IN DEST (`add_tiles(..., acc_to_dest=true)` over the two halves of the
//        row's window), the finalize (D17's raw-sfpi column-scoped `*(1/W) + eps` then
//        `rsqrt`) runs on that same DEST slot, and ONE `pack_tile` writes cb_stat_handoff.
//        GATHER_SLOTS == GROUP_SIZE rounded up to even is what makes the pairwise walk
//        universal: a pad slot is boot-zeroed by the writer and adds an exact +0.0.
//
// THE ONE THING THE TREE CHANGES: an INTERIOR gatherer runs the identical pairwise DEST
// fold but MUST NOT finalize -- it packs the RAW sum and forwards it, and only the
// last-level gatherer (slot 0, the multicast root) applies the rsqrt.  A finalize at an
// interior node would rsqrt a partial sum.
//
// AND THE ONE THING THAT IS NOT OPTIONAL AT A NEW FOLD SITE: `reconfig_data_format` +
// `pack_reconfig_data_format`.  Every fold in this bench emits both, exactly as D22 does.
// Omitting them makes the fold unpack fp32 L1 through a bf16 srcA/srcB; the accumulated sum
// reads as ~0, the finalize turns it into rsqrt(eps), and the result is a uniform
// ~1/sqrt(eps) SCALE error that holds pcc at 0.9997 and shows up ONLY in rel-RMS.  That is
// why this bench gates on rel-RMS as well as pcc.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// ---------------------------------------------------------------------------------------
// RAW-LLK, CARRIED VERBATIM FROM THE OP (kernels/rms_norm_compute.cpp, D17 + D22).
// Not a new bypass: the baseline this bench measures against IS this code, so re-spelling
// the finalize (or the fold) with helper calls would make the baseline slower than the op
// and inflate every candidate's speedup.  The op's own justifications, in one line each:
//   * the FINALIZE -- mul_unary_tile / add_unary_tile / rsqrt_tile all hard-code
//     VectorMode::RC and expose no VectorMode seam, and column parity is the SFPU's INNER
//     walk axis so ITERATIONS cannot reach it; the stat is a REDUCE_ROW column vector whose
//     only consumer reads column 0.
//   * the FUSED FOLD -- eltwise_chain runs EVERY element's apply on EVERY inner iteration
//     (eltwise_chain.inl `elem_apply_compute`), so a finalize element after an accumulating
//     BinaryFpu would rsqrt a PARTIAL sum GROUP_SIZE/2 times instead of once on the
//     completed one.  There is no apply-after-the-accumulation element kind and no per-row
//     tail hook.  The op measured the helper-expressible split form at 1.93x against this
//     2.18x and kept both so the gap stays re-checkable.
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

namespace {
constexpr uint32_t cb_gather0 = 11;  // level-l gather ring is cb_gather0 + l
constexpr uint32_t cb_node_out = 15;
constexpr uint32_t cb_stat_handoff = 16;
}  // namespace

// ONE fold, the D22 spelling.  `SLOTS` contiguous fp32 partials per tile-row are folded
// pairwise in DEST; the last level also finalizes.  `WAIT_PAGES` differs between the two
// variants ONLY in what the writer publishes: FLAT publishes `SLOTS * rows` (the op's own
// spelling), the tree publishes the WHOLE ring every round so a remote sender can compute
// the landing address from its own identical CB (see the writer's ring-discipline note).
template <uint32_t CB_IN, uint32_t SLOTS, uint32_t CB_OUT, bool FINALIZE, uint32_t IW_BITS, uint32_t EP_BITS>
ALWI void fold_level(uint32_t rows, uint32_t wait_pages) {
    constexpr uint32_t HALF = SLOTS / 2;
    static_assert(SLOTS % 2 == 0 && HALF >= 1, "the pairwise DEST walk needs an even, non-empty window");
    // The whole block's gather window is waited/popped ONCE: the pairwise walk addresses two
    // tiles of the same CB at a stride, which a per-tile wait cannot express.  Legal because
    // the writer publishes the block atomically and the CB is sized to that same window.
    cb_wait_front(CB_IN, wait_pages);
    reconfig_data_format(CB_IN, CB_IN);
    pack_reconfig_data_format(CB_OUT);
    add_tiles_init(CB_IN, CB_IN, /*acc_to_dest=*/true);
    if constexpr (FINALIZE) {
        // MANDATORY, not decorative: rms_stat_rsqrt_body reads sfpi::vConstIntPrgm0 /
        // vConstFloatPrgm1..2, which sfpu::rsqrt_init programs -- persistent SFPU PROGRAM
        // registers, which is what makes hoisting it out of the per-row loop legal.
        rsqrt_tile_init();
    }
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t base = r * SLOTS;
        tile_regs_acquire();
        for (uint32_t p = 0; p < HALF; ++p) {
            add_tiles(CB_IN, CB_IN, base + p, base + HALF + p, 0);
        }
        if constexpr (FINALIZE) {
            stat_finalize_payload<IW_BITS, EP_BITS>(0);
        }
        tile_regs_commit();
        // Reserve/push PER TILE-ROW, not per block: the writer's next hop (the stat
        // multicast at the last level, the forward at an interior one) starts on the first
        // finished row.  The op measured the cost of keeping that overlap at zero.
        cb_reserve_back(CB_OUT, 1);
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
    }
    cb_pop_front(CB_IN, wait_pages);
}

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_LEVELS = get_compile_time_arg_val(3);
    constexpr uint32_t F0 = get_compile_time_arg_val(4);
    constexpr uint32_t F1 = get_compile_time_arg_val(5);
    constexpr uint32_t F2 = get_compile_time_arg_val(6);
    constexpr uint32_t F3 = get_compile_time_arg_val(7);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(8);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(9);

    constexpr uint32_t GATHER_SLOTS = GROUP_SIZE + GROUP_SIZE % 2;
    constexpr uint32_t SL0 = F0 + F0 % 2;
    constexpr uint32_t SL1 = F1 + F1 % 2;
    constexpr uint32_t SL2 = F2 + F2 % 2;
    constexpr uint32_t SL3 = F3 + F3 % 2;
    // stride[l+1]: the slot spacing of the level-l gatherers.  A core folds at level l iff
    // `my_slot % stride[l+1] == 0` (see the writer's tree-geometry note).
    constexpr uint32_t ST1 = F0;
    constexpr uint32_t ST2 = F0 * F1;
    constexpr uint32_t ST3 = F0 * F1 * F2;
    constexpr uint32_t ST4 = F0 * F1 * F2 * F3;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t my_slot = get_arg_val<uint32_t>(2);

    // Cores with nothing to compute (inactive, or a plain member that only ships) return
    // before any CB or LLK state is touched -- the op does the same.
    const bool participates = (num_rows != 0) && ((VARIANT == 0) ? (is_root != 0) : (my_slot % ST1 == 0));
    if (!participates) {
        return;
    }

    // Every CB in this bench is fp32, so srcA == srcB at boot and the per-fold
    // reconfig_data_format pair covers every later ring for free.
    compute_kernel_hw_startup(cb_gather0, cb_gather0, cb_stat_handoff);

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

        if constexpr (VARIANT == 0) {
            // ---- FLAT: the op's D22 fused root chain, verbatim --------------------
            MaybeDeviceZoneScope("compute_root_fused");
            fold_level<cb_gather0, GATHER_SLOTS, cb_stat_handoff, true, INV_W_BITS, EPS_BITS>(
                rows, GATHER_SLOTS * rows);
        } else {
            // ---- TREE: fold every level at which I am the chunk gatherer ----------
            // The predicate is monotone in l (strides are increasing multiples), so a core
            // folds a PREFIX of the levels and the last-level fold happens on slot 0 alone.
            if constexpr (NUM_LEVELS >= 1) {
                if (my_slot % ST1 == 0) {
                    MaybeDeviceZoneScope("compute_fold_l0");
                    fold_level<
                        cb_gather0 + 0,
                        SL0,
                        (NUM_LEVELS == 1 ? cb_stat_handoff : cb_node_out),
                        NUM_LEVELS == 1,
                        INV_W_BITS,
                        EPS_BITS>(rows, SL0 * BLOCK_ROWS);
                }
            }
            if constexpr (NUM_LEVELS >= 2) {
                if (my_slot % ST2 == 0) {
                    MaybeDeviceZoneScope("compute_fold_l1");
                    fold_level<
                        cb_gather0 + 1,
                        SL1,
                        (NUM_LEVELS == 2 ? cb_stat_handoff : cb_node_out),
                        NUM_LEVELS == 2,
                        INV_W_BITS,
                        EPS_BITS>(rows, SL1 * BLOCK_ROWS);
                }
            }
            if constexpr (NUM_LEVELS >= 3) {
                if (my_slot % ST3 == 0) {
                    MaybeDeviceZoneScope("compute_fold_l2");
                    fold_level<
                        cb_gather0 + 2,
                        SL2,
                        (NUM_LEVELS == 3 ? cb_stat_handoff : cb_node_out),
                        NUM_LEVELS == 3,
                        INV_W_BITS,
                        EPS_BITS>(rows, SL2 * BLOCK_ROWS);
                }
            }
            if constexpr (NUM_LEVELS >= 4) {
                if (my_slot % ST4 == 0) {
                    MaybeDeviceZoneScope("compute_fold_l3");
                    fold_level<cb_gather0 + 3, SL3, cb_stat_handoff, true, INV_W_BITS, EPS_BITS>(
                        rows, SL3 * BLOCK_ROWS);
                }
            }
        }
    }
}
