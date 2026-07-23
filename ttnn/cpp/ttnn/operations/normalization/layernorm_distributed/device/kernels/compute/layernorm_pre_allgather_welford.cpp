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
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    namespace kutil = norm::kernel_util;
    namespace generic = kutil::generic;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
#if FUSE_PRE_ADD
    constexpr uint32_t blk = get_compile_time_arg_val(2);
#endif
    // True iff the factory configured cb_inp_idx with UnpackToDestFp32. Used by the
    // non-FUSE branch to gate the welford state re-establishment after the transpose.
    constexpr bool welford_unpack_fp32_active = get_named_compile_time_arg_val("welford_unpack_fp32_active") != 0;

    constexpr uint32_t cb_in0_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_out_idx = tt::CBIndex::c_14;
    constexpr uint32_t cb_scratch_idx = tt::CBIndex::c_1;  // scratch for post-Welford transpose
    constexpr uint32_t cb_reciprocals = tt::CBIndex::c_2;  // recip table
#if FUSE_PRE_ADD
    constexpr uint32_t cb_res_idx = tt::CBIndex::c_5;         // residual b
    constexpr uint32_t cb_inp_idx = tt::CBIndex::c_3;         // fused a + b (sized to a few blocks)
    constexpr uint32_t cb_mean_spill_idx = tt::CBIndex::c_4;  // Welford mean accumulator spill (1 tile)
    constexpr uint32_t cb_m2_spill_idx = tt::CBIndex::c_6;    // Welford M2 accumulator spill (1 tile)
#else
    constexpr uint32_t cb_inp_idx = cb_in0_idx;
#endif

#if FUSE_PRE_ADD
    binary_op_init_common(cb_in0_idx, cb_res_idx, cb_inp_idx);
#else
    compute_kernel_hw_startup(cb_inp_idx, cb_inp_idx, cb_scratch_idx);
#endif

    CircularBuffer cb_out(cb_out_idx);
    CircularBuffer cb_scratch(cb_scratch_idx);
    CircularBuffer cb_inp(cb_inp_idx);
#if FUSE_PRE_ADD
    CircularBuffer cb_in0(cb_in0_idx);
    CircularBuffer cb_res(cb_res_idx);
    CircularBuffer cb_mean_spill(cb_mean_spill_idx);
    CircularBuffer cb_m2_spill(cb_m2_spill_idx);
#endif
    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);
    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    constexpr uint32_t last_tile_rows = (W % 32) == 0 ? 32 : W % 32;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr uint32_t dst0 = 0;
        constexpr uint32_t dst1 = 1;
        constexpr uint32_t dst2 = 2;

#if FUSE_PRE_ADD
        // Block-interleaved pre-add + Welford. The Welford accumulator lives in the SFPU within a
        // tile_regs scope, but the pre-add must use its own tile_regs scope to pack its result to
        // cb_inp_idx before the Welford pass can transpose-read it back. To bridge those scopes the
        // accumulator (mean, M2) is spilled to cb_mean_spill_idx / cb_m2_spill_idx between chunks via
        // welford_save_state / welford_restore_state. This lets cb_inp_idx stay sized to a small
        // number of tiles (blk * 2 for double-buffer) regardless of Wt. Larger blk amortizes the
        // save/restore overhead and accuracy loss across more tiles per spill cycle; blk is
        // chosen by the factory as gcd(Wt, DST capacity) so it always divides Wt.

        // Seed the spill CBs with an initialized (zero) Welford state,
        // since iteration 0 below expects it.
        tile_regs_acquire();
        welford_init();
        welford_save_state(dst1);
        tile_regs_commit();
        cb_mean_spill.reserve_back(1);
        cb_m2_spill.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_mean_spill_idx);
        pack_tile(dst1, cb_mean_spill_idx);
        pack_tile(dst2, cb_m2_spill_idx);
        tile_regs_release();
        cb_mean_spill.push_back(1);
        cb_m2_spill.push_back(1);

        uint32_t start_N = 0;
        for (auto block : generic::blocks(Wt, blk)) {
            // --- Pre-add: cb_in0_idx + cb_res_idx -> cb_inp_idx (block tiles in one tile_regs scope) ---
            reconfig_data_format(cb_in0_idx, cb_res_idx);
            pack_reconfig_data_format(cb_inp_idx);
            cb_in0.wait_front(block.size());
            cb_res.wait_front(block.size());
            cb_inp.reserve_back(block.size());
            if constexpr (welford_unpack_fp32_active) {
                // SFPU path: copy_tile bypasses SrcA via UnpackToDestEn, preserving full FP32
                copy_init(cb_in0_idx);
                for (auto i : block.local()) {
                    tile_regs_acquire();
                    copy_tile(cb_in0_idx, i, 0);
                    reconfig_data_format_srca(cb_in0_idx, cb_res_idx);
                    copy_init(cb_res_idx);
                    copy_tile(cb_res_idx, i, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_inp_idx);
                    tile_regs_release();
                    reconfig_data_format_srca(cb_res_idx, cb_in0_idx);
                    copy_init(cb_in0_idx);
                }
            } else {
                add_tiles_init(cb_in0_idx, cb_res_idx);
                tile_regs_acquire();
                for (auto i : block.local()) {
                    add_tiles(cb_in0_idx, cb_res_idx, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, cb_inp_idx);
                }
                tile_regs_release();
            }
            cb_inp.push_back(block.size());
            cb_in0.pop_front(block.size());
            cb_res.pop_front(block.size());

            // --- Welford: reload accumulator, update with block tiles, spill back ---
            cb_mean_spill.wait_front(1);
            cb_m2_spill.wait_front(1);
            cb_inp.wait_front(block.size());
            tile_regs_acquire();
            reconfig_data_format_srca(cb_in0_idx, cb_mean_spill_idx);
            copy_init(cb_mean_spill_idx);
            copy_tile(cb_mean_spill_idx, 0, dst1);
            reconfig_data_format_srca(cb_mean_spill_idx, cb_m2_spill_idx);
            copy_init(cb_m2_spill_idx);
            copy_tile(cb_m2_spill_idx, 0, dst2);
            welford_restore_state(dst1);

            reconfig_data_format_srca(cb_m2_spill_idx, cb_inp_idx);
            if constexpr (!welford_unpack_fp32_active) {
                transpose_init(cb_inp_idx);
            }
            for (auto i : block.local()) {
                if constexpr (welford_unpack_fp32_active) {
                    transpose_init(cb_inp_idx);
                }
                transpose_tile(cb_inp_idx, i, dst0);
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
            cb_mean_spill.pop_front(1);
            cb_m2_spill.pop_front(1);
            cb_inp.pop_front(block.size());
            cb_mean_spill.reserve_back(1);
            cb_m2_spill.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_inp_idx, cb_mean_spill_idx);
            pack_tile(dst1, cb_mean_spill_idx);
            pack_tile(dst2, cb_m2_spill_idx);
            tile_regs_release();
            cb_mean_spill.push_back(1);
            cb_m2_spill.push_back(1);
        }

        // Finalize: reload accumulator and write mean and variance to cb_scratch_idx.
        cb_mean_spill.wait_front(1);
        cb_m2_spill.wait_front(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_inp_idx, cb_mean_spill_idx);
        copy_init(cb_mean_spill_idx);
        copy_tile(cb_mean_spill_idx, 0, dst1);
        reconfig_data_format_srca(cb_mean_spill_idx, cb_m2_spill_idx);
        copy_init(cb_m2_spill_idx);
        copy_tile(cb_m2_spill_idx, 0, dst2);
        welford_restore_state(dst1);
        welford_finalize_to_row<W>(dst1, W - 1, *p_reciprocals);
        tile_regs_commit();
        cb_mean_spill.pop_front(1);
        cb_m2_spill.pop_front(1);

        cb_scratch.reserve_back(2);
        tile_regs_wait();
        pack_reconfig_data_format(cb_mean_spill_idx, cb_scratch_idx);
        pack_tile(dst1, cb_scratch_idx);
        pack_tile(dst2, cb_scratch_idx);
        cb_scratch.push_back(2);
        tile_regs_release();
#else
        reconfig_data_format(cb_inp_idx, cb_inp_idx);
        pack_reconfig_data_format(cb_scratch_idx);

        tile_regs_acquire();
        uint32_t start_N = 0;
        transpose_init(cb_inp_idx);
        welford_init();

        // When the input CB carries Float32 with fp32_dest_acc_en=true, the program factory
        // sets UnpackToDestFp32 for cb_inp_idx so transpose_tile preserves FP32 precision into DEST.
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
            cb_inp.wait_front(1);  // cumulative wait
            if constexpr (welford_unpack_fp32_active) {
                transpose_init(cb_inp_idx);
            }
            transpose_tile(cb_inp_idx, 0, dst0);
            if constexpr (welford_unpack_fp32_active) {
                welford_init<WelfordInitMode::PreserveStats>();
            }
            // welford_tile<dst0, dst1, dst2, true, 0>((wt) * 32, W, 0, {});
            welford_update<W>(dst0, start_N, *p_reciprocals);
            start_N += 32;
            cb_inp.pop_front(1);
        }
        cb_inp.wait_front(1);  // cumulative wait
        if constexpr (welford_unpack_fp32_active) {
            transpose_init(cb_inp_idx);
        }
        transpose_tile(cb_inp_idx, 0, dst0);
        if constexpr (welford_unpack_fp32_active) {
            welford_init<WelfordInitMode::PreserveStats>();
        }
        welford_update_rows<W>(dst0, start_N, 0, last_tile_rows, *p_reciprocals);
        cb_inp.pop_front(1);
        welford_finalize_to_row<W>(dst1, W - 1, *p_reciprocals);
        // tt-llk/issues/549
        // BUG: using transpose_dest here causes a bug. where the kernel hangs
        //  transpose_dest_init();
        //  transpose_dest(dst1);
        //  transpose_dest(dst2);
        cb_scratch.reserve_back(2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst1, cb_scratch_idx);
        pack_tile(dst2, cb_scratch_idx);
        cb_scratch.push_back(2);
        tile_regs_release();
#endif

        reconfig_data_format(cb_scratch_idx, cb_scratch_idx);
        pack_reconfig_data_format(cb_out_idx);
        transpose_init(cb_scratch_idx);
        tile_regs_acquire();
        cb_scratch.wait_front(2);  // cumulative wait
        transpose_tile(cb_scratch_idx, 0, dst0);
        transpose_tile(cb_scratch_idx, 1, dst1);
        cb_scratch.pop_front(2);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out_idx);
        pack_tile(dst1, cb_out_idx);
        cb_out.push_back(2);
        tile_regs_release();
    }
}
