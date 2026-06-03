// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — Compute kernel.
//
// Algorithm overview (single core, no inter-core comm):
//
//   For each batch n:
//     PHASE R + R/post (per group g, serially):
//       For each (T_in_g_span, r):
//         (RM input only)  tilize 32 sticks → 1 tile in cb_input_tiles_R
//         mul<NONE>(input, mask) → cb_scratch_a  (mask popped per-tile)
//         reduce<SUM, REDUCE_SCALAR, WaitUpfrontNoPop>(cb_scratch_a, ...) → cb_running_acc_sum
//                                                                          (Accumulate)
//         square(cb_scratch_a) → cb_scratch_b  (consumes cb_scratch_a)
//         reduce<SUM, REDUCE_SCALAR>(cb_scratch_b, ...) → cb_running_acc_sumsq (Accumulate)
//       Then compute mean_g / rcp_std_g from the running accumulators and push to
//       cb_group_mean / cb_group_rcp_std (each accumulating G tiles).
//     PHASE A (per output T):
//       1. Read 1 input tile per r from cb_input_tiles_A (or tilize 32 sticks for RM).
//       2. For each g touching T, copy cb_group_mean[g] / cb_group_rcp_std[g] to
//          1-slot active CBs, multiply by the mask (pushed by reader), and accumulate
//          into cb_means_tile_T / cb_rcp_std_tile_T.
//       3. Tilize gamma / beta sticks → cb_gamma_tile / cb_beta_tile.
//       4. For each r: out = ((input - means_T) * rcp_std_T [* gamma_T] [+ beta_T]).
//       5. Pop the per-T staging tiles.
//
// Helpers used:
//   * compute_kernel_lib::tilize         — RM → tile conversion
//   * compute_kernel_lib::mul / sub / add / square (binary_op variants)
//   * compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>
//   * compute_kernel_lib::add_in_place (for the per-T accumulation)
//   * compute_kernel_lib::sfpu_pipeline + sfpu_chain (for var → rsqrt)
//   * Raw `copy_tile` is used ONCE to extract cb_group_mean[g] / cb_group_rcp_std[g]
//     by index into cb_active_mean / cb_active_rcp_std (1-slot CBs), because the
//     binary_op helper indexes B at tile 0 for SCALAR broadcast and does not support
//     arbitrary tile indexing. This is the documented helper limitation; the rest of
//     the kernel uses helpers exclusively.

#include <stdint.h>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"

// CB indices — must match groupnorm_sc_N_1_HW_C_program_descriptor.py
constexpr uint32_t CB_INPUT_RM_R = 0;
constexpr uint32_t CB_INPUT_RM_A = 1;
constexpr uint32_t CB_INPUT_TILES_R = 2;
constexpr uint32_t CB_INPUT_TILES_A = 3;
constexpr uint32_t CB_GAMMA_RM = 4;
constexpr uint32_t CB_BETA_RM = 5;
constexpr uint32_t CB_GAMMA_TILE = 6;
constexpr uint32_t CB_BETA_TILE = 7;
constexpr uint32_t CB_SCALER_ONE = 8;
constexpr uint32_t CB_MASK_STREAM = 9;
constexpr uint32_t CB_INV_N_SCALAR = 10;
constexpr uint32_t CB_RUNNING_ACC_SUM = 12;
constexpr uint32_t CB_RUNNING_ACC_SUMSQ = 13;
constexpr uint32_t CB_OUTPUT_TILES = 16;
constexpr uint32_t CB_ACTIVE_MEAN = 24;
constexpr uint32_t CB_ACTIVE_RCP_STD = 25;
constexpr uint32_t CB_GROUP_MEAN = 26;
constexpr uint32_t CB_GROUP_RCP_STD = 27;
constexpr uint32_t CB_SCRATCH_A = 28;
constexpr uint32_t CB_SCRATCH_B = 29;
constexpr uint32_t CB_MEANS_TILE_T = 30;
constexpr uint32_t CB_RCP_STD_TILE_T = 31;

constexpr uint32_t TILE_DIM = 32;

namespace ckl = compute_kernel_lib;

// Copy tile at index `src_idx` of `src_cb` into the front slot of `dst_cb`
// (which must be reserved/pushed by this helper). Used to "snapshot" a
// specific tile from a multi-tile CB into a single-slot active CB so the
// binary_op helper (which addresses B at tile 0 for SCALAR broadcast) can
// consume it.
//
// The srcA-unpack + pack reconfigs are required: the prior compute op
// (e.g. `sfpu_pipeline` ending phase R/post, or the next phase A binary
// op) leaves the unpacker srcA / packer in a state tuned for a different
// CB. With fp32_dest_acc_en=True and fp32 stats CBs the dst register is
// in 32-bit-cell mode; calling bare `copy_tile_to_dst_init_short` +
// `pack_tile` without re-asserting the data format produces silent dst
// corruption (catastrophic NaN/Inf in the apply phase when Cg % 32 != 0).
FORCE_INLINE void snapshot_tile_to_active_cb(uint32_t src_cb, uint32_t src_idx, uint32_t dst_cb) {
    reconfig_data_format_srca(src_cb);
    pack_reconfig_data_format(dst_cb);

    tile_regs_acquire();
    copy_tile_to_dst_init_short(src_cb);
    copy_tile(src_cb, src_idx, /*dst_idx=*/0);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(dst_cb, 1);
    pack_tile(0, dst_cb);
    cb_push_back(dst_cb, 1);
    tile_regs_release();
}

void kernel_main() {
    // ----- CT args -----
    constexpr uint32_t INPUT_LAYOUT_CODE = get_compile_time_arg_val(0);  // 0=TILE, 1=RM
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(1);
    constexpr uint32_t HAS_BETA = get_compile_time_arg_val(2);
    constexpr uint32_t N = get_compile_time_arg_val(3);
    [[maybe_unused]] constexpr uint32_t HW = get_compile_time_arg_val(4);
    constexpr uint32_t C = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t Cg = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Ct = get_compile_time_arg_val(9);
    // (slot 10 = EPS_BITS — bit-packed for SfpuAddScalar, accessed inside the
    // R/post block where it's actually used. Don't re-declare here.)
    constexpr uint32_t AFFINE_LAYOUT_CODE = get_compile_time_arg_val(11);  // 0=TILE, 1=RM

    // Broadcast mode for the apply-phase gamma / beta operands.
    //   AFFINE_LAYOUT_CODE == 1 (RM): the reader replicated the gamma stick 32×
    //     and compute tilized it → the resulting cb_gamma_tile is row-replicated
    //     across all 32 rows → BroadcastDim::NONE is correct (full-tile mul/add).
    //   AFFINE_LAYOUT_CODE == 0 (TILE): the reader pushed a single TILE-laid
    //     gamma page; only row 0 is logically valid (the (1,1,1,C) → tile pad
    //     zeros rows 1-31) → BroadcastDim::ROW broadcasts row 0 down all 32
    //     input rows.
    constexpr ckl::BroadcastDim AFFINE_BCAST =
        (AFFINE_LAYOUT_CODE == 0) ? ckl::BroadcastDim::ROW : ckl::BroadcastDim::NONE;

    // ----- One-time hardware startup -----
    // Pick CBs that cover the main binary streaming pattern. The helpers handle
    // their own reconfig between operations.
    compute_kernel_hw_startup(CB_INPUT_TILES_R, CB_MASK_STREAM, CB_SCRATCH_A);

    // The one-shot CBs (scaler / inv_N) have been pushed once by the reader.
    // Wait for them upfront; they are never popped during the kernel's lifetime.
    cb_wait_front(CB_SCALER_ONE, 1);
    cb_wait_front(CB_INV_N_SCALAR, 1);

    // ============================================================
    // PER-BATCH MAIN LOOP
    // ============================================================
    for (uint32_t n = 0; n < N; ++n) {
        // ----------------------------------------------------------------
        // PHASE R + R/post — per group g (serial)
        // ----------------------------------------------------------------
        for (uint32_t g = 0; g < G; ++g) {
            // Compute group g's tile span. (Matches reader exactly.)
            const uint32_t g_start_ch = g * Cg;
            const uint32_t g_end_ch_raw = (g + 1) * Cg;
            const uint32_t g_end_ch = g_end_ch_raw < C ? g_end_ch_raw : C;
            const uint32_t T_first = g_start_ch / TILE_DIM;
            const uint32_t T_last = (g_end_ch - 1) / TILE_DIM;
            const uint32_t tile_span_g = T_last - T_first + 1;

            uint32_t iter = 0;
            for (uint32_t T_idx = 0; T_idx < tile_span_g; ++T_idx) {
                for (uint32_t r = 0; r < Ht; ++r) {
                    if constexpr (INPUT_LAYOUT_CODE == 1) {
                        // RM input — tilize 32 sticks (one tile-row block) → 1 tile.
                        // ASYMMETRIC: input CB has row-sized pages (64 bytes); pass
                        // total_input_pages = 32 so the helper waits for 32 row pages.
                        ckl::tilize<1, CB_INPUT_RM_R, CB_INPUT_TILES_R>(1, 32);
                    }

                    // mul<NONE>(cb_input_tiles_R, cb_mask_stream) → cb_scratch_a
                    // Both inputs are per-tile streaming (1 tile each).
                    ckl::mul<ckl::BroadcastDim::NONE>(
                        CB_INPUT_TILES_R, CB_MASK_STREAM, CB_SCRATCH_A, ckl::BinaryInputBlockShape::single());

                    // reduce SUM cb_scratch_a → cb_running_acc_sum (Accumulate, retain cb_scratch_a)
                    // input_policy=WaitUpfrontNoPop so the tile in cb_scratch_a is not popped
                    // (we square it next).
                    ckl::reduce<
                        ckl::PoolType::SUM,
                        ckl::ReduceDim::REDUCE_SCALAR,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                        CB_SCRATCH_A,
                        CB_SCALER_ONE,
                        CB_RUNNING_ACC_SUM,
                        ckl::ReduceInputBlockShape::single(),
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate(ckl::AccumulationConfig::with_cb(CB_RUNNING_ACC_SUM), iter));

                    // Square the (still-resident) masked tile from cb_scratch_a → cb_scratch_b.
                    // Default policy = WaitAndPopPerTile, so cb_scratch_a is consumed here.
                    ckl::square(CB_SCRATCH_A, CB_SCRATCH_B, ckl::BinaryInputBlockShape::single());

                    // reduce SUM cb_scratch_b → cb_running_acc_sumsq (Accumulate, pop scratch_b).
                    ckl::reduce<ckl::PoolType::SUM, ckl::ReduceDim::REDUCE_SCALAR>(
                        CB_SCRATCH_B,
                        CB_SCALER_ONE,
                        CB_RUNNING_ACC_SUMSQ,
                        ckl::ReduceInputBlockShape::single(),
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate(ckl::AccumulationConfig::with_cb(CB_RUNNING_ACC_SUMSQ), iter));

                    iter++;
                }
            }

            // -------- R/post for group g --------
            // After phase R for g: cb_running_acc_sum has sum_g (1 tile);
            //                      cb_running_acc_sumsq has sumsq_g (1 tile).
            //
            // 1) mean_g = sum_g * inv_N → cb_scratch_a (1 tile).
            ckl::mul<
                ckl::BroadcastDim::SCALAR,
                ckl::BinaryInputPolicy::WaitAndPopPerTile,
                ckl::BinaryInputPolicy::NoWaitNoPop>(
                CB_RUNNING_ACC_SUM, CB_INV_N_SCALAR, CB_SCRATCH_A, ckl::BinaryInputBlockShape::single());

            // 2) E[X²] = sumsq_g * inv_N → cb_scratch_b (1 tile).
            ckl::mul<
                ckl::BroadcastDim::SCALAR,
                ckl::BinaryInputPolicy::WaitAndPopPerTile,
                ckl::BinaryInputPolicy::NoWaitNoPop>(
                CB_RUNNING_ACC_SUMSQ, CB_INV_N_SCALAR, CB_SCRATCH_B, ckl::BinaryInputBlockShape::single());

            // 3) Push a copy of mean_g (cb_scratch_a) into cb_group_mean (durably retained
            //    across phase A). Use copy_tiles with NoWaitNoPop so cb_scratch_a stays
            //    populated for the next step.
            ckl::copy_tiles<ckl::CopyInputPolicy::NoWaitNoPop>(
                CB_SCRATCH_A,
                CB_GROUP_MEAN,
                /*num_tiles=*/1);

            // 4) cb_scratch_a = mean_g² (in-place square).
            ckl::square_in_place(CB_SCRATCH_A, ckl::BinaryInputBlockShape::single());

            // 5) var_g = E[X²] − mean² = cb_scratch_b − cb_scratch_a → cb_running_acc_sum
            //    (cb_running_acc_sum is empty at this point; both inputs popped after).
            ckl::sub<ckl::BroadcastDim::NONE>(
                CB_SCRATCH_B, CB_SCRATCH_A, CB_RUNNING_ACC_SUM, ckl::BinaryInputBlockShape::single());

            // 6) rcp_std_g = 1/sqrt(var_g + eps) via SFPU chain:
            //    Load var → AddScalar(eps_bits) → Rsqrt → cb_group_rcp_std (push 1).
            {
                constexpr uint32_t EPS_BITS_CT = get_compile_time_arg_val(10);
                ckl::AddScalar<ckl::Dst::D0> add_eps_op{};
                add_eps_op.scalar = EPS_BITS_CT;
                auto chain = ckl::sfpu_chain(
                    ckl::Load<CB_RUNNING_ACC_SUM, ckl::Dst::D0>{},
                    add_eps_op,
                    ckl::Rsqrt<ckl::Legacy::Off, ckl::Approx::Exact, ckl::Dst::D0>{});
                ckl::sfpu_pipeline(chain, CB_GROUP_RCP_STD, /*num_tiles=*/1);
            }
        }

        // ----------------------------------------------------------------
        // PHASE A — per output T: build per-channel means/rcp_std tile,
        // tilize gamma/beta, then apply the affine transform per HW-row.
        // ----------------------------------------------------------------

        // Wait upfront for all G mean / rcp_std tiles; they will be held across all T's
        // (and never popped during phase A — we copy_tile by index out of them).
        cb_wait_front(CB_GROUP_MEAN, G);
        cb_wait_front(CB_GROUP_RCP_STD, G);

        for (uint32_t T = 0; T < Ct; ++T) {
            const uint32_t T_base = T * TILE_DIM;
            // Which groups touch tile T?
            // First, list them by computing g_start_T and g_end_T.
            uint32_t g_first_T = 0;
            uint32_t g_last_T = 0;
            bool any_touch = false;
            for (uint32_t g = 0; g < G; ++g) {
                const uint32_t g_start = g * Cg;
                const uint32_t g_end_raw = (g + 1) * Cg;
                const uint32_t g_end = g_end_raw < C ? g_end_raw : C;
                if (g_start >= T_base + TILE_DIM) {
                    break;  // no more groups touch this T
                }
                if (g_end <= T_base) {
                    continue;
                }
                if (!any_touch) {
                    g_first_T = g;
                    any_touch = true;
                }
                g_last_T = g;
            }

            // Build cb_means_tile_T and cb_rcp_std_tile_T via accumulation over g.
            // First iteration: write directly (no accumulate); subsequent iterations:
            // mul → cb_scratch_a/b, then add_in_place into the per-T tile.
            bool first_g = true;
            for (uint32_t g = g_first_T; g <= g_last_T; ++g) {
                const uint32_t g_start = g * Cg;
                const uint32_t g_end_raw = (g + 1) * Cg;
                const uint32_t g_end = g_end_raw < C ? g_end_raw : C;
                if (g_start >= T_base + TILE_DIM) {
                    break;
                }
                if (g_end <= T_base) {
                    continue;
                }

                // Snapshot mean_g and rcp_std_g into 1-slot active CBs so the
                // SCALAR-broadcast mul helper can read them at tile-index 0.
                snapshot_tile_to_active_cb(CB_GROUP_MEAN, g, CB_ACTIVE_MEAN);
                snapshot_tile_to_active_cb(CB_GROUP_RCP_STD, g, CB_ACTIVE_RCP_STD);

                if (first_g) {
                    // First group: write directly into the per-T tile. Reader has pushed
                    // one mask tile for (T, g_first_T); take it, multiply with the active
                    // mean / rcp_std, push into cb_means_tile_T / cb_rcp_std_tile_T.
                    //
                    // We need two mul ops, each using a distinct mask. But reader pushed
                    // ONE mask per (T, g) iteration. We need TWO consumers of the same
                    // mask (mean expansion + rcp_std expansion). So we use the same mask
                    // for both via NoWaitNoPop for one of them.
                    //
                    // Strategy: first mul consumes mask (default WaitAndPopPerTile on B);
                    // before the call, we explicitly wait for the mask and use it twice
                    // via NoWaitNoPop. Then pop it after the second use.
                    //
                    // Simplified: wait+pop the mask via the mul's input_a (the mask is
                    // input A in this mul<SCALAR>(mask, mean) call). The first call pops
                    // it; that's a problem because we still need it for the rcp_std mul.
                    //
                    // Solution: cb_wait_front mask manually, then use NoWaitNoPop for both,
                    // and cb_pop_front after both calls.
                    cb_wait_front(CB_MASK_STREAM, 1);

                    // mean expansion: cb_means_tile_T = mask * mean_g
                    ckl::mul<
                        ckl::BroadcastDim::SCALAR,
                        ckl::BinaryInputPolicy::NoWaitNoPop,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_MASK_STREAM, CB_ACTIVE_MEAN, CB_MEANS_TILE_T, ckl::BinaryInputBlockShape::single());

                    // rcp_std expansion: cb_rcp_std_tile_T = mask * rcp_std_g
                    ckl::mul<
                        ckl::BroadcastDim::SCALAR,
                        ckl::BinaryInputPolicy::NoWaitNoPop,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_MASK_STREAM, CB_ACTIVE_RCP_STD, CB_RCP_STD_TILE_T, ckl::BinaryInputBlockShape::single());

                    cb_pop_front(CB_MASK_STREAM, 1);

                    // Active mean / rcp_std were used (NoWaitNoPop), pop them manually.
                    cb_pop_front(CB_ACTIVE_MEAN, 1);
                    cb_pop_front(CB_ACTIVE_RCP_STD, 1);

                    first_g = false;
                } else {
                    // Subsequent groups: mul into a scratch, then add_in_place into the per-T
                    // tile.
                    cb_wait_front(CB_MASK_STREAM, 1);

                    ckl::mul<
                        ckl::BroadcastDim::SCALAR,
                        ckl::BinaryInputPolicy::NoWaitNoPop,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_MASK_STREAM, CB_ACTIVE_MEAN, CB_SCRATCH_A, ckl::BinaryInputBlockShape::single());

                    ckl::mul<
                        ckl::BroadcastDim::SCALAR,
                        ckl::BinaryInputPolicy::NoWaitNoPop,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_MASK_STREAM, CB_ACTIVE_RCP_STD, CB_SCRATCH_B, ckl::BinaryInputBlockShape::single());

                    cb_pop_front(CB_MASK_STREAM, 1);
                    cb_pop_front(CB_ACTIVE_MEAN, 1);
                    cb_pop_front(CB_ACTIVE_RCP_STD, 1);

                    // Accumulate into the per-T tiles.
                    ckl::add_in_place<ckl::BroadcastDim::NONE>(
                        CB_MEANS_TILE_T, CB_SCRATCH_A, ckl::BinaryInputBlockShape::single());
                    ckl::add_in_place<ckl::BroadcastDim::NONE>(
                        CB_RCP_STD_TILE_T, CB_SCRATCH_B, ckl::BinaryInputBlockShape::single());
                }
            }

            // Stage gamma / beta tile for this T.
            //
            // AFFINE_LAYOUT_CODE == 1 (ROW_MAJOR): reader pushed 32 RM sticks
            //   into cb_gamma_rm; tilize them into cb_gamma_tile (row-replicated
            //   full tile). Apply-phase mul/add then uses BroadcastDim::NONE.
            // AFFINE_LAYOUT_CODE == 0 (TILE): reader pushed 1 tile directly
            //   into cb_gamma_tile. Only row 0 is logically valid (the rest is
            //   the (1,1,1,C) → (1,1,32,padded_C) zero pad). Apply-phase mul/add
            //   then uses BroadcastDim::ROW so row 0 broadcasts down all 32
            //   input rows. No tilize step is needed here.
            if constexpr (AFFINE_LAYOUT_CODE == 1) {
                if constexpr (HAS_GAMMA) {
                    ckl::tilize<1, CB_GAMMA_RM, CB_GAMMA_TILE>(1, 32);
                }
                if constexpr (HAS_BETA) {
                    ckl::tilize<1, CB_BETA_RM, CB_BETA_TILE>(1, 32);
                }
            }

            // ----- Apply loop -----
            // Per HW-row tile r: compute output tile = ((input - means_T) * rcp_std_T
            //                                            [* gamma_T] [+ beta_T]).
            // cb_means_tile_T / cb_rcp_std_tile_T / cb_gamma_tile / cb_beta_tile are
            // held with NoWaitNoPop / WaitUpfrontNoPop semantics across the r loop and
            // popped manually after.
            cb_wait_front(CB_MEANS_TILE_T, 1);
            cb_wait_front(CB_RCP_STD_TILE_T, 1);
            if constexpr (HAS_GAMMA) {
                cb_wait_front(CB_GAMMA_TILE, 1);
            }
            if constexpr (HAS_BETA) {
                cb_wait_front(CB_BETA_TILE, 1);
            }

            for (uint32_t r = 0; r < Ht; ++r) {
                if constexpr (INPUT_LAYOUT_CODE == 1) {
                    // RM input — tilize 32 sticks → 1 tile in cb_input_tiles_A.
                    // ASYMMETRIC: row-sized CB pages.
                    ckl::tilize<1, CB_INPUT_RM_A, CB_INPUT_TILES_A>(1, 32);
                }

                // sub<NONE>(input, means_T) → cb_scratch_a
                ckl::sub<
                    ckl::BroadcastDim::NONE,
                    ckl::BinaryInputPolicy::WaitAndPopPerTile,
                    ckl::BinaryInputPolicy::NoWaitNoPop>(
                    CB_INPUT_TILES_A, CB_MEANS_TILE_T, CB_SCRATCH_A, ckl::BinaryInputBlockShape::single());

                if constexpr (HAS_GAMMA && HAS_BETA) {
                    // (input - mean) * rcp_std → cb_scratch_b
                    ckl::mul<
                        ckl::BroadcastDim::NONE,
                        ckl::BinaryInputPolicy::WaitAndPopPerTile,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_SCRATCH_A, CB_RCP_STD_TILE_T, CB_SCRATCH_B, ckl::BinaryInputBlockShape::single());

                    // * gamma → cb_scratch_a  (broadcast: NONE for RM gamma_tile, ROW for TILE gamma_tile)
                    ckl::mul<
                        AFFINE_BCAST,
                        ckl::BinaryInputPolicy::WaitAndPopPerTile,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_SCRATCH_B, CB_GAMMA_TILE, CB_SCRATCH_A, ckl::BinaryInputBlockShape::single());

                    // + beta → cb_output_tiles  (broadcast: NONE for RM beta_tile, ROW for TILE beta_tile)
                    ckl::add<
                        AFFINE_BCAST,
                        ckl::BinaryInputPolicy::WaitAndPopPerTile,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_SCRATCH_A, CB_BETA_TILE, CB_OUTPUT_TILES, ckl::BinaryInputBlockShape::single());
                } else if constexpr (HAS_GAMMA && !HAS_BETA) {
                    ckl::mul<
                        ckl::BroadcastDim::NONE,
                        ckl::BinaryInputPolicy::WaitAndPopPerTile,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_SCRATCH_A, CB_RCP_STD_TILE_T, CB_SCRATCH_B, ckl::BinaryInputBlockShape::single());

                    ckl::mul<
                        AFFINE_BCAST,
                        ckl::BinaryInputPolicy::WaitAndPopPerTile,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_SCRATCH_B, CB_GAMMA_TILE, CB_OUTPUT_TILES, ckl::BinaryInputBlockShape::single());
                } else if constexpr (!HAS_GAMMA && HAS_BETA) {
                    ckl::mul<
                        ckl::BroadcastDim::NONE,
                        ckl::BinaryInputPolicy::WaitAndPopPerTile,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_SCRATCH_A, CB_RCP_STD_TILE_T, CB_SCRATCH_B, ckl::BinaryInputBlockShape::single());

                    ckl::add<
                        AFFINE_BCAST,
                        ckl::BinaryInputPolicy::WaitAndPopPerTile,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_SCRATCH_B, CB_BETA_TILE, CB_OUTPUT_TILES, ckl::BinaryInputBlockShape::single());
                } else {
                    // no_affine
                    ckl::mul<
                        ckl::BroadcastDim::NONE,
                        ckl::BinaryInputPolicy::WaitAndPopPerTile,
                        ckl::BinaryInputPolicy::NoWaitNoPop>(
                        CB_SCRATCH_A, CB_RCP_STD_TILE_T, CB_OUTPUT_TILES, ckl::BinaryInputBlockShape::single());
                }
            }

            // Pop the per-T staging tiles.
            cb_pop_front(CB_MEANS_TILE_T, 1);
            cb_pop_front(CB_RCP_STD_TILE_T, 1);
            if constexpr (HAS_GAMMA) {
                cb_pop_front(CB_GAMMA_TILE, 1);
            }
            if constexpr (HAS_BETA) {
                cb_pop_front(CB_BETA_TILE, 1);
            }
        }

        // End of batch n — pop the G per-group stat tiles.
        cb_pop_front(CB_GROUP_MEAN, G);
        cb_pop_front(CB_GROUP_RCP_STD, G);
    }
}
