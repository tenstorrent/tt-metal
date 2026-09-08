// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes layernorm statistics.
 * For layernorm it computes E(x**2) and E(x) and returns them as a two tile wide output tensor containing E(x**2) and
 * E(x) in the left most columns per tile. For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/layernorm.h"
#include "api/compute/transpose.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// The Welford pass reads either the raw input or the fused a + b result, depending on whether a
// residual was supplied. Only the buffer selected here is bound on this build, so the alias is gated
// at the preprocessor: naming an unbound handle would not compile even on a discarded branch.
#ifdef FUSE_PRE_ADD
constexpr auto dfb_inp_id = dfb::fused;  // fused a + b (sized to a few blocks)
#else
constexpr auto dfb_inp_id = dfb::in0;
#endif

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    namespace kutil = norm::kernel_util;
    namespace generic = kutil::generic;
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto W = get_arg(args::W);
#ifdef FUSE_PRE_ADD
    constexpr auto blk = get_arg(args::blk);
#endif
    // True iff the factory configured the input buffer with UnpackToDestFp32. Used by the
    // non-FUSE branch to gate the welford state re-establishment after the transpose.
    constexpr bool welford_unpack_fp32_active = get_arg(args::welford_unpack_fp32_active) != 0;

#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb::in0, dfb::res, dfb_inp_id);
#else
    compute_kernel_hw_startup(dfb_inp_id, dfb_inp_id, dfb::scratch);
#endif

    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_scratch(dfb::scratch);  // scratch for post-Welford transpose
    DataflowBuffer dfb_inp(dfb_inp_id);
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_res(dfb::res);                // residual b
    DataflowBuffer dfb_mean_spill(dfb::mean_spill);  // Welford mean accumulator spill (1 tile)
    DataflowBuffer dfb_m2_spill(dfb::m2_spill);      // Welford M2 accumulator spill (1 tile)
#endif
    // Get pointer to the reciprocal LUT, which lives in the memory the recip buffer borrows.
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(dfb::recip, 0);
    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    constexpr uint32_t last_tile_rows = (W % 32) == 0 ? 32 : W % 32;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr uint32_t dst0 = 0;
        constexpr uint32_t dst1 = 1;
        constexpr uint32_t dst2 = 2;

#ifdef FUSE_PRE_ADD
        // Block-interleaved pre-add + Welford. The Welford accumulator lives in the SFPU within a
        // tile_regs scope, but the pre-add must use its own tile_regs scope to pack its result to
        // dfb_inp_id before the Welford pass can transpose-read it back. To bridge those scopes the
        // accumulator (mean, M2) is spilled to dfb::mean_spill / dfb::m2_spill between chunks via
        // welford_save_state / welford_restore_state. This lets dfb_inp_id stay sized to a small
        // number of tiles (blk * 2 for double-buffer) regardless of Wt. Larger blk amortizes the
        // save/restore overhead and accuracy loss across more tiles per spill cycle; blk is
        // chosen by the factory as gcd(Wt, DST capacity) so it always divides Wt.

        // Seed the spill buffers with an initialized (zero) Welford state,
        // since iteration 0 below expects it.
        tile_regs_acquire();
        welford_init();
        welford_save_state(dst1);
        tile_regs_commit();
        dfb_mean_spill.reserve_back(1);
        dfb_m2_spill.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::mean_spill);
        pack_tile(dst1, dfb::mean_spill);
        pack_tile(dst2, dfb::m2_spill);
        tile_regs_release();
        dfb_mean_spill.push_back(1);
        dfb_m2_spill.push_back(1);

        uint32_t start_N = 0;
        for (auto block : generic::blocks(Wt, blk)) {
            // --- Pre-add: dfb::in0 + dfb::res -> dfb_inp_id (block tiles in one tile_regs scope) ---
            reconfig_data_format(dfb::in0, dfb::res);
            pack_reconfig_data_format(dfb_inp_id);
            dfb_in0.wait_front(block.size());
            dfb_res.wait_front(block.size());
            dfb_inp.reserve_back(block.size());
            if constexpr (welford_unpack_fp32_active) {
                // SFPU path: copy_tile bypasses SrcA via UnpackToDestEn, preserving full FP32
                copy_init(dfb::in0);
                for (auto i : block.local()) {
                    tile_regs_acquire();
                    copy_tile(dfb::in0, i, 0);
                    reconfig_data_format_srca(dfb::in0, dfb::res);
                    copy_init(dfb::res);
                    copy_tile(dfb::res, i, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, dfb_inp_id);
                    tile_regs_release();
                    reconfig_data_format_srca(dfb::res, dfb::in0);
                    copy_init(dfb::in0);
                }
            } else {
                add_init(dfb::in0, dfb::res);
                tile_regs_acquire();
                for (auto i : block.local()) {
                    add_tiles(dfb::in0, dfb::res, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, dfb_inp_id);
                }
                tile_regs_release();
            }
            dfb_inp.push_back(block.size());
            dfb_in0.pop_front(block.size());
            dfb_res.pop_front(block.size());

            // --- Welford: reload accumulator, update with block tiles, spill back ---
            dfb_mean_spill.wait_front(1);
            dfb_m2_spill.wait_front(1);
            dfb_inp.wait_front(block.size());
            tile_regs_acquire();
            reconfig_data_format_srca(dfb::in0, dfb::mean_spill);
            copy_init(dfb::mean_spill);
            copy_tile(dfb::mean_spill, 0, dst1);
            reconfig_data_format_srca(dfb::mean_spill, dfb::m2_spill);
            copy_init(dfb::m2_spill);
            copy_tile(dfb::m2_spill, 0, dst2);
            welford_restore_state(dst1);

            reconfig_data_format_srca(dfb::m2_spill, dfb_inp_id);
            if constexpr (!welford_unpack_fp32_active) {
                transpose_init(dfb_inp_id);
            }
            for (auto i : block.local()) {
                if constexpr (welford_unpack_fp32_active) {
                    transpose_init(dfb_inp_id);
                }
                transpose_tile(dfb_inp_id, i, dst0);
                if constexpr (welford_unpack_fp32_active) {
                    welford_init<WelfordInitMode::PreserveStats>();
                }
                if (block.to_global(i) < Wt - 1) {
                    welford_update<W>(dst0, start_N, *p_reciprocals);
                } else {
                    welford_update_rows<W>(dst0, start_N, 0, last_tile_rows, *p_reciprocals);
                }
                start_N += 32;
            }
            welford_save_state(dst1);
            tile_regs_commit();
            dfb_mean_spill.pop_front(1);
            dfb_m2_spill.pop_front(1);
            dfb_inp.pop_front(block.size());
            dfb_mean_spill.reserve_back(1);
            dfb_m2_spill.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(dfb_inp_id, dfb::mean_spill);
            pack_tile(dst1, dfb::mean_spill);
            pack_tile(dst2, dfb::m2_spill);
            tile_regs_release();
            dfb_mean_spill.push_back(1);
            dfb_m2_spill.push_back(1);
        }

        // Finalize: reload accumulator and write mean and variance to the scratch buffer.
        dfb_mean_spill.wait_front(1);
        dfb_m2_spill.wait_front(1);
        tile_regs_acquire();
        reconfig_data_format_srca(dfb_inp_id, dfb::mean_spill);
        copy_init(dfb::mean_spill);
        copy_tile(dfb::mean_spill, 0, dst1);
        reconfig_data_format_srca(dfb::mean_spill, dfb::m2_spill);
        copy_init(dfb::m2_spill);
        copy_tile(dfb::m2_spill, 0, dst2);
        welford_restore_state(dst1);
        welford_finalize_to_row<W>(dst1, W - 1, *p_reciprocals);
        tile_regs_commit();
        dfb_mean_spill.pop_front(1);
        dfb_m2_spill.pop_front(1);

        dfb_scratch.reserve_back(2);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::mean_spill, dfb::scratch);
        pack_tile(dst1, dfb::scratch);
        pack_tile(dst2, dfb::scratch);
        dfb_scratch.push_back(2);
        tile_regs_release();
#else
        reconfig_data_format(dfb_inp_id, dfb_inp_id);
        pack_reconfig_data_format(dfb::scratch);

        tile_regs_acquire();
        uint32_t start_N = 0;
        transpose_init(dfb_inp_id);
        welford_init();

        // When the input buffer carries Float32 with fp32_dest_acc_en=true, the program factory
        // sets UnpackToDestFp32 for it so transpose_tile preserves FP32 precision into DEST.
        // Its math-side init (called from transpose_init) records slots [16, 32) of the
        // math-thread replay buffer, clobbering the LREG2 / LREG3 portions of Welford's recurrence
        // (welford records slots [0, 32), which is 4 LREG variants of 8 instructions each, fully unrolled).
        // welford_init<WelfordInitMode::PreserveStats>() after each transpose_tile re-records
        // all 32 slots with the welford recurrence so welford_update replays welford ops instead
        // of stale transpose-dest ops. PreserveStats keeps the running mean / M2 accumulator in
        // LREG4/5, which survive transpose_dest anyway because it only uses FPU MOVs. UNPACK A
        // is left in transpose=1 by transpose_tile; welford_update is pure SFPU and does
        // not consume that state, and the next iteration's transpose_init reprograms
        // it.
        //
        // For bf16 input the unpack-to-DEST fp32 path is inactive: transpose_tile routes
        // through SrcA without touching the math-thread replay buffer, so the recovery is
        // gated out.
        for (uint32_t wt = 0; wt < (Wt - 1); wt++) {
            dfb_inp.wait_front(1);  // cumulative wait
            if constexpr (welford_unpack_fp32_active) {
                transpose_init(dfb_inp_id);
            }
            transpose_tile(dfb_inp_id, 0, dst0);
            if constexpr (welford_unpack_fp32_active) {
                welford_init<WelfordInitMode::PreserveStats>();
            }
            // welford_tile<dst0, dst1, dst2, true, 0>((wt) * 32, W, 0, {});
            welford_update<W>(dst0, start_N, *p_reciprocals);
            start_N += 32;
            dfb_inp.pop_front(1);
        }
        dfb_inp.wait_front(1);  // cumulative wait
        if constexpr (welford_unpack_fp32_active) {
            transpose_init(dfb_inp_id);
        }
        transpose_tile(dfb_inp_id, 0, dst0);
        if constexpr (welford_unpack_fp32_active) {
            welford_init<WelfordInitMode::PreserveStats>();
        }
        welford_update_rows<W>(dst0, start_N, 0, last_tile_rows, *p_reciprocals);
        dfb_inp.pop_front(1);
        welford_finalize_to_row<W>(dst1, W - 1, *p_reciprocals);
        // tt-llk/issues/549
        // BUG: using transpose_dest here causes a bug. where the kernel hangs
        //  transpose_dest_init();
        //  transpose_dest(dst1);
        //  transpose_dest(dst2);
        dfb_scratch.reserve_back(2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst1, dfb::scratch);
        pack_tile(dst2, dfb::scratch);
        dfb_scratch.push_back(2);
        tile_regs_release();
#endif

        reconfig_data_format(dfb::scratch, dfb::scratch);
        pack_reconfig_data_format(dfb::out);
        transpose_init(dfb::scratch);
        tile_regs_acquire();
        dfb_scratch.wait_front(2);  // cumulative wait
        transpose_tile(dfb::scratch, 0, dst0);
        transpose_tile(dfb::scratch, 1, dst1);
        dfb_scratch.pop_front(2);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, dfb::out);
        pack_tile(dst1, dfb::out);
        dfb_out.push_back(2);
        tile_regs_release();
    }
}
