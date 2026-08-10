// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/transpose.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

namespace generic = norm::kernel_util::generic;

template <
    uint32_t dfb_in_id,
    uint32_t dfb_inb_id,
    uint32_t dfb_interm_pre_add_id,
    uint32_t dfb_ex_id,
    uint32_t dfb_ex2_id,
    uint32_t dfb_ex_welford_id,
    uint32_t dfb_ex2_welford_id,
    bool welford_state_fp32_alias,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t var_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    DataflowBuffer dfb_in_obj(dfb_in_id);
    DataflowBuffer dfb_inb_obj(dfb_inb_id);
    DataflowBuffer dfb_interm_pre_add_obj(dfb_interm_pre_add_id);
    DataflowBuffer dfb_ex_obj(dfb_ex_id);
    DataflowBuffer dfb_ex2_obj(dfb_ex2_id);
    // When welford_state_fp32_alias is true these are c_30/c_31; distinct buffer indices
    // sharing dfb_ex_id/dfb_ex2_id's SRAM allocations but configured with UnpackToDestFp32.
    // When false, dfb_ex_welford_id == dfb_ex_id and dfb_ex2_welford_id == dfb_ex2_id.
    DataflowBuffer dfb_ex_welford_obj(dfb_ex_welford_id);
    DataflowBuffer dfb_ex2_welford_obj(dfb_ex2_welford_id);

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    // When last tile is full the value is 0 and is not used because full update is done.
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;

    tile_regs_acquire();
    welford_init();
    welford_save_state(mean_dst);
    tile_regs_commit();

    dfb_ex_obj.reserve_back(1);
    dfb_ex2_obj.reserve_back(1);
    if constexpr (welford_state_fp32_alias) {
        // Must be done in compute: dfb_ex_id / dfb_ex2_id hold welford state (mean / M2) which are
        // produced by pack_tile below; the reader never writes these DFBs. Aliases share SRAM
        // but have independent read/write counters and need to be kept in sync so the next
        // block's wait_front on the aliases (used by copy_tile for fp32 precision) sees the data.
        dfb_ex_welford_obj.reserve_back(1);
        dfb_ex2_welford_obj.reserve_back(1);
    }
    tile_regs_wait();
    pack_reconfig_data_format(dfb_ex_id);
    pack_tile(mean_dst, dfb_ex_id);
    pack_tile(var_dst, dfb_ex2_id);
    tile_regs_release();
    dfb_ex_obj.push_back(1);
    dfb_ex2_obj.push_back(1);
    if constexpr (welford_state_fp32_alias) {
        dfb_ex_welford_obj.push_back(1);
        dfb_ex2_welford_obj.push_back(1);
    }

    for (auto block : generic::blocks(Wt, blk)) {
        const auto block_shape =
            ckl::IterationShape::tiles(block.size(), block.full_block_size(), ckl::BlockTailSync::FullBlock);
        // Keep pre-add in a separate DFB to avoid the transpose_dest aliasing issue.
        ckl::add<
            ckl::input(dfb_in_id, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::input(
                dfb_inb_id, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::output(dfb_interm_pre_add_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            block_shape);

        // Now run Welfords in these blk number of tiles
        dfb_interm_pre_add_obj.wait_front(block.full_block_size());
        dfb_ex_obj.wait_front(1);
        dfb_ex2_obj.wait_front(1);
        if constexpr (welford_state_fp32_alias) {
            dfb_ex_welford_obj.wait_front(1);
            dfb_ex2_welford_obj.wait_front(1);
        }
        tile_regs_acquire();
        // Reload running mean/M2 from the aliases. With welford_state_fp32_alias active
        // these are c_30/c_31 in UnpackToDestFp32 mode so copy_tile takes the Dst path that
        // preserves the full FP32 precision. Otherwise, dfb_ex_welford_id == dfb_ex_id.
        reconfig_data_format_srca(dfb_in_id, dfb_ex_welford_id);
        copy_tile_init(dfb_ex_welford_id);
        copy_tile(dfb_ex_welford_id, 0, mean_dst);
        reconfig_data_format_srca(dfb_ex_welford_id, dfb_ex2_welford_id);
        copy_tile_to_dst_init_short_with_dt(dfb_ex_welford_id, dfb_ex2_welford_id);
        copy_tile(dfb_ex2_welford_id, 0, var_dst);
        welford_restore_state(mean_dst);

        reconfig_data_format_srca(dfb_ex2_welford_id, dfb_interm_pre_add_id);
        transpose_init(dfb_interm_pre_add_id);
        for (auto i : block.local()) {
            // Welford's needs transposed input tile
            transpose_tile(dfb_interm_pre_add_id, i, input_dst);

            // Welford over this tile: include only valid elements, never padding.
            if constexpr (is_last_tile_full) {
                // All tiles can go through the faster call which does 32 rows
                welford_update<W>(input_dst, sample_idx, reciprocal_lut);
            } else {
                // Last tile in width has padding; process only first last_tile_rows rows.
                if ((block.start() + i) == (Wt - 1)) {
                    welford_update_rows<W>(input_dst, sample_idx, 0, last_tile_rows, reciprocal_lut);
                } else {
                    welford_update<W>(input_dst, sample_idx, reciprocal_lut);
                }
            }
            sample_idx += tile_width;
        }
        welford_save_state(mean_dst);
        tile_regs_commit();
        dfb_interm_pre_add_obj.pop_front(block.full_block_size());
        dfb_ex_obj.pop_front(1);
        dfb_ex2_obj.pop_front(1);
        if constexpr (welford_state_fp32_alias) {
            dfb_ex_welford_obj.pop_front(1);
            dfb_ex2_welford_obj.pop_front(1);
        }

        dfb_ex_obj.reserve_back(1);
        dfb_ex2_obj.reserve_back(1);
        if constexpr (welford_state_fp32_alias) {
            // This alias update must be in the compute kernel.
            // pack_tile below is the producer of dfb_ex_id / dfb_ex2_id.
            dfb_ex_welford_obj.reserve_back(1);
            dfb_ex2_welford_obj.reserve_back(1);
        }
        tile_regs_wait();
        pack_reconfig_data_format(dfb_interm_pre_add_id, dfb_ex_id);
        pack_tile(mean_dst, dfb_ex_id);
        pack_tile(var_dst, dfb_ex2_id);
        tile_regs_release();
        dfb_ex_obj.push_back(1);
        dfb_ex2_obj.push_back(1);
        if constexpr (welford_state_fp32_alias) {
            dfb_ex_welford_obj.push_back(1);
            dfb_ex2_welford_obj.push_back(1);
        }
    }

    reconfig_data_format_srca(dfb_interm_pre_add_id, dfb_ex_welford_id);

    dfb_ex_obj.wait_front(1);
    dfb_ex2_obj.wait_front(1);
    if constexpr (welford_state_fp32_alias) {
        dfb_ex_welford_obj.wait_front(1);
        dfb_ex2_welford_obj.wait_front(1);
    }
    tile_regs_acquire();
    // Reload through the FP32 alias before finalizing.
    copy_tile_init(dfb_ex_welford_id);
    copy_tile(dfb_ex_welford_id, 0, mean_dst);
    copy_tile_to_dst_init_short_with_dt(dfb_ex_welford_id, dfb_ex2_welford_id);
    copy_tile(dfb_ex2_welford_id, 0, var_dst);
    welford_restore_state(mean_dst);
    // Store the mean and variance to the destination registers
    welford_finalize_to_row<W>(mean_dst, W - 1, reciprocal_lut);
    tile_regs_commit();
    dfb_ex_obj.pop_front(1);
    dfb_ex2_obj.pop_front(1);
    if constexpr (welford_state_fp32_alias) {
        dfb_ex_welford_obj.pop_front(1);
        dfb_ex2_welford_obj.pop_front(1);
    }
}

/* @brief: Welford's algorithm for no fused pre-add
 * @param: dfb_in_id: input DFB
 * @param: input_dst: input tile for Welford's algorithm
 * @param: mean_dst: mean tile for Welford's algorithm
 * @param: Wt: width of the input in tiles
 * @param: tile_width: width of each tile
 * @param: W: width of the input
 * @param: p_reciprocals: pointer to the reciprocal LUT
 */
template <
    uint32_t dfb_in_id,
    uint32_t dfb_x_welford_id,
    bool welford_fp32_alias,
    uint32_t dfb_ex_id,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_no_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    DataflowBuffer dfb_in_obj(dfb_in_id);
    DataflowBuffer dfb_x_welford_obj(dfb_x_welford_id);

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    // When last tile is full the value is 0 and is not used because full update is done.
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;
    reconfig_data_format_srca(dfb_x_welford_id);
    // Reconfigure the transpose op for the welford intake DFB. When the alias is active,
    // dfb_x_welford_id has UnpackToDestFp32 mode so transpose_tile preserves fp32 precision.
    transpose_init(dfb_x_welford_id);
    tile_regs_acquire();
    welford_init();

    // Process all but the last tile
    for (uint32_t wt = 0; wt < (Wt - 1); ++wt) {
        if constexpr (welford_fp32_alias) {
            dfb_x_welford_obj.wait_front(1);
            // SFPU replay slots [0, 32) currently hold the welford recurrence (welford uses the
            // full 32-slot math-thread replay buffer; the recovery block below re-records all
            // of it after each transpose). transpose_init re-records slots [16, 32)
            // with the transpose-dest setup so transpose_tile below can replay them.
            transpose_init(dfb_x_welford_id);
        } else {
            dfb_in_obj.wait_front(1);
        }
        // Welford's needs transposed input tile
        transpose_tile(dfb_x_welford_id, 0, input_dst);
        if constexpr (welford_fp32_alias) {
            // transpose_tile took the UnpackToDestFp32 path. Its math-side init clobbered
            // the welford recurrence at SFPU replay slots [16, 32).
            // welford_init<WelfordInitMode::PreserveStats>() re-records all 32 slots with the
            // welford recurrence; PreserveStats keeps the running mean / M2 accumulator in
            // LREG4/5. UNPACK A is left in transpose=1;
            // welford_update is pure SFPU and does not consume that state, and the next
            // iteration's transpose_init reprograms it.
            welford_init<WelfordInitMode::PreserveStats>();
        }
        welford_update<W>(input_dst, sample_idx, reciprocal_lut);

        // Pop the input
        if constexpr (welford_fp32_alias) {
            dfb_x_welford_obj.pop_front(1);
        }
        dfb_in_obj.pop_front(1);
        sample_idx += tile_width;
    }

    // Process the last tile
    // Reader is sending full blocks, so we need to stay in sync.
    // wait/pop the last tile + any remaining in the last block
    const auto num_to_sync = generic::blocks(Wt, blk).back().remainder() + 1;
    if constexpr (welford_fp32_alias) {
        dfb_x_welford_obj.wait_front(num_to_sync);
        transpose_init(dfb_x_welford_id);
    } else {
        dfb_in_obj.wait_front(num_to_sync);
    }
    transpose_tile(dfb_x_welford_id, 0, input_dst);
    if constexpr (welford_fp32_alias) {
        welford_init<WelfordInitMode::PreserveStats>();
    }

    if constexpr (is_last_tile_full) {
        welford_update<W>(input_dst, sample_idx, reciprocal_lut);
    } else {
        welford_update_rows<W>(input_dst, sample_idx, 0, last_tile_rows, reciprocal_lut);
    }

    // Store the mean and variance to the destination registers
    welford_finalize_to_row<W>(mean_dst, W - 1, reciprocal_lut);

    tile_regs_commit();

    if constexpr (welford_fp32_alias) {
        dfb_x_welford_obj.pop_front(num_to_sync);
    }
    dfb_in_obj.pop_front(num_to_sync);
}

void kernel_main() {
    namespace kutil = norm::kernel_util;

    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t do_beta = get_compile_time_arg_val(3);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t tile_width = get_compile_time_arg_val(6);
    constexpr bool fuse_pre_add = static_cast<bool>(get_compile_time_arg_val(8));
    // welford_fp32_alias: when true, dfb_x_welford_id is a multi-buffer-index alias of dfb_x_id
    // configured with UnpackToDestFp32 so the welford section reads full fp32 into DEST
    // while the post-welford eltwise still reads dfb_x_id via SrcA (Tf32).
    // When false, dfb_x_welford_id == dfb_x_id.
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    constexpr auto dfb_x_welford_id = get_named_compile_time_arg_val("cb_x_welford");

    // welford_state_fp32_alias: when true, dfb_ex_welford_id/dfb_ex2_welford_id are c_30/c_31
    // multi-buffer-index aliases of dfb_ex_id (c_18) / dfb_ex2_id (c_19) configured for UnpackToDestFp32.
    // The fused welford path's per-block copy_tile reads of the running mean / M2 use
    // these aliases to take the Dst fp32 path (preserves FP32 precision) instead of the
    // SrcA Tf32 path. When false, dfb_ex_welford_id == dfb_ex_id and dfb_ex2_welford_id == dfb_ex2_id.
    constexpr bool welford_state_fp32_alias = get_named_compile_time_arg_val("welford_state_fp32_alias") != 0;
    constexpr auto dfb_ex_welford_id = get_named_compile_time_arg_val("cb_ex_welford");
    constexpr auto dfb_ex2_welford_id = get_named_compile_time_arg_val("cb_ex2_welford");

    // Note that the entire W dimension must fit in the intermed0 DFB for this kernel to be correct
    // DFB indices - configurable via named compile-time args for kernel chaining support
    constexpr auto dfb_eps_id = get_named_compile_time_arg_val("cb_eps");  // single tile generated by the reader
    constexpr auto dfb_in_id = get_named_compile_time_arg_val("cb_in");    // input x or a for fused pre-add (x=a+b)
    constexpr auto dfb_inb_id = get_named_compile_time_arg_val("cb_inb");  // input b for fused pre-add
    constexpr auto dfb_out_id = get_named_compile_time_arg_val("cb_out");  // output
    constexpr auto dfb_gamma_id = get_named_compile_time_arg_val("cb_gamma");
    constexpr auto dfb_beta_id = get_named_compile_time_arg_val("cb_beta");
    constexpr auto dfb_xmm_id = get_named_compile_time_arg_val("cb_xmm");  // x - E[x]
    uint32_t dfb_xmm_runtime_id = dfb_xmm_id;
    constexpr auto dfb_ex_id = get_named_compile_time_arg_val("cb_ex");             // E[x]
    constexpr auto dfb_ex2_id = get_named_compile_time_arg_val("cb_ex2");           // Var[x] = E[(x-E[x])^2]
    constexpr auto dfb_ex2pe_id = get_named_compile_time_arg_val("cb_ex2pe");       // Var[x]+ε
    constexpr auto dfb_fusion_id = get_named_compile_time_arg_val("cb_fusion");     // stream gamma/beta
    constexpr auto dfb_interm_pre_add_id = get_named_compile_time_arg_val("cb_x");  // intermediate for fused pre-add
    constexpr auto dfb_reciprocals_id = get_named_compile_time_arg_val("cb_reciprocals");  // Pre-computed reciprocals

    DataflowBuffer dfb_eps_obj(dfb_eps_id);
    DataflowBuffer dfb_in_obj(dfb_in_id);
    DataflowBuffer dfb_inb_obj(dfb_inb_id);
    DataflowBuffer dfb_ex_obj(dfb_ex_id);
    DataflowBuffer dfb_ex2_obj(dfb_ex2_id);
    DataflowBuffer dfb_ex2pe_obj(dfb_ex2pe_id);

    constexpr uint32_t onetile = 1;

    // Initialize the hardware based on the first op
    // that will be done
    if constexpr (fuse_pre_add) {
        // Init for x = in + b
        compute_kernel_hw_startup(dfb_in_id, dfb_inb_id, dfb_interm_pre_add_id);
    } else {
        // Init for transpose
        constexpr auto first_out_dfb_id = dfb_ex_id;
        compute_kernel_hw_startup(dfb_in_id, first_out_dfb_id);
    }

    dfb_eps_obj.wait_front(onetile);  // comes from the reader

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;  // Input tile for Welford's algorithm
    constexpr uint32_t mean_dst = 1;   // Mean tile for Welford's
    constexpr uint32_t var_dst = 2;    // Variance tile for Welford's

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(dfb_reciprocals_id, 0);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Depending on whether we need to fuse pre-add, the approach for welford is different.
        // So we move it to a separate function.
        if constexpr (fuse_pre_add) {
            welford_fuse_pre_add<
                dfb_in_id,
                dfb_inb_id,
                dfb_interm_pre_add_id,
                dfb_ex_id,
                dfb_ex2_id,
                dfb_ex_welford_id,
                dfb_ex2_welford_id,
                welford_state_fp32_alias,
                input_dst,
                mean_dst,
                var_dst,
                Wt,
                tile_width,
                W,
                blk>(*p_reciprocals);
        } else {
            welford_no_fuse_pre_add<
                dfb_in_id,
                dfb_x_welford_id,
                welford_fp32_alias,
                dfb_ex_id,
                input_dst,
                mean_dst,
                Wt,
                tile_width,
                W,
                blk>(*p_reciprocals);
        }
        // We should expect that either of the two would have have populated dst regs with mean and
        // variance in mean_dst and var_dst respectively.

        dfb_ex_obj.reserve_back(onetile);
        dfb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb_ex_id);
        pack_tile(mean_dst, dfb_ex_id);
        pack_tile(var_dst, dfb_ex2_id);
        tile_regs_release();
        dfb_ex_obj.push_back(onetile);
        dfb_ex2_obj.push_back(onetile);

        // Transpose mean and variance back to
        // columns and pack back to DFBs
        reconfig_data_format_srca(dfb_ex_id);
        transpose_init(dfb_ex_id);

        dfb_ex_obj.wait_front(onetile);
        dfb_ex2_obj.wait_front(onetile);
        tile_regs_acquire();
        transpose_tile(dfb_ex_id, 0, mean_dst);
        transpose_tile(dfb_ex2_id, 0, var_dst);
        tile_regs_commit();
        dfb_ex_obj.pop_front(onetile);
        dfb_ex2_obj.pop_front(onetile);

        dfb_ex_obj.reserve_back(onetile);
        dfb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb_ex_id);
        pack_tile(mean_dst, dfb_ex_id);
        pack_reconfig_data_format(dfb_ex2_id);
        pack_tile(var_dst, dfb_ex2_id);
        tile_regs_release();
        dfb_ex_obj.push_back(onetile);
        dfb_ex2_obj.push_back(onetile);

        // =====================================
        // Calculate 1/(√(Var(X) + ε))
        // =====================================
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb_ex2_id),
                ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_ex2pe_id,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Disabled)>{});

        ckl::unary_bcast<ckl::BroadcastDim::Col, ckl::input(dfb_ex2pe_id), ckl::output(dfb_ex2pe_id)>(
            ckl::IterationShape::tiles(onetile));

        // =====================================
        // Second pass over the input.
        // Computes the final value:
        //    x-E[x]
        //(---------------*𝛄)+ß
        //  √(Var(x)+ε)
        // =====================================
        dfb_ex2pe_obj.wait_front(onetile);
        dfb_ex_obj.wait_front(onetile);

        // Lockstep the dfb_x_welford_id alias's read/write pointers with dfb_in_id's across the eltwise pass.
        // The reader pushes dfb_x_welford_id in pass 2 to match its pass 1 push (see
        // reader_unary_interleaved_ln_large_tensor_welford.cpp); compute pops it here to match
        // dfb_in_id's pop. Both share SRAM but have independent state; popping dfb_x_welford_id keeps it aligned
        // with dfb_in_id so the next NCHt Welford iteration reads from the correct SRAM offset after DFB wrap.
        DataflowBuffer dfb_x_welford_obj_eltwise(dfb_x_welford_id);

        for (auto block : generic::blocks(Wt, blk)) {
            const auto block_shape =
                ckl::IterationShape::tiles(block.size(), block.full_block_size(), ckl::BlockTailSync::FullBlock);
            // Last block may only be partially-filled,
            // and only tiles that have data in them are
            // processed, but need to sync with reader on full blocks
            dfb_in_obj.wait_front(block.full_block_size());
            if constexpr (welford_fp32_alias && !fuse_pre_add) {
                // dfb_x_welford_id was pushed by the reader in pass 2; wait for the push and pop in
                // lockstep with dfb_in_id. We do not actually read dfb_x_welford_id in the eltwise pass
                // (FPU consumes dfb_in_id via SrcA); this is purely a FIFO-pointer sync.
                dfb_x_welford_obj_eltwise.wait_front(block.full_block_size());
            }
            tile_regs_acquire();
            reconfig_data_format(dfb_in_id, dfb_ex_id);
            sub_bcast_cols_init(dfb_in_id, dfb_ex_id);
            // x-E[x]
            for (auto i : block.local()) {
                sub_tiles_bcast_cols(dfb_in_id, dfb_ex_id, i, 0, i);
            }
            dfb_in_obj.pop_front(block.full_block_size());
            if constexpr (welford_fp32_alias && !fuse_pre_add) {
                dfb_x_welford_obj_eltwise.pop_front(block.full_block_size());
            }

            if constexpr (fuse_pre_add) {
                // Fuse in = in + b
                reconfig_data_format_srca(dfb_in_id, dfb_inb_id);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_inb_id);
                dfb_inb_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_inb_id, i, i);
                }
                dfb_inb_obj.pop_front(block.full_block_size());
            }

            // Multiply by 1/(√(Var(X) + ε)). SrcA currently holds dfb_inb_id (fused) or dfb_in_id
            // (non-fused), the last operand read above; switch it to dfb_ex2pe_id's format.
            reconfig_data_format_srca(fuse_pre_add ? dfb_inb_id : dfb_in_id, dfb_ex2pe_id);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_ex2pe_id);
            for (auto i : block.local()) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_ex2pe_id, 0 /*in_tile_index*/, i);
            }
            tile_regs_commit();

            if constexpr (!(do_gamma == 1 or do_beta == 1)) {
                dfb_xmm_runtime_id = dfb_out_id;
            }

            pack_reconfig_data_format(dfb_xmm_runtime_id);
            // Sync with writer on full blocks
            DataflowBuffer(dfb_xmm_runtime_id).reserve_back(block.full_block_size());
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, dfb_xmm_runtime_id);
            }
            DataflowBuffer(dfb_xmm_runtime_id).push_back(block.full_block_size());
            tile_regs_release();

            if constexpr (do_gamma == 1) {
                constexpr auto dfb_gamma_out_id = do_beta ? dfb_fusion_id : dfb_out_id;
                ckl::mul<
                    ckl::input(
                        dfb_xmm_id,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::OperandKind::Block),
                    ckl::input(
                        dfb_gamma_id,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::OperandKind::Block),
                    ckl::output(dfb_gamma_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                    block_shape);
            }

            if constexpr (do_beta == 1) {
                constexpr auto dfb_beta_input_id = do_gamma ? dfb_fusion_id : dfb_xmm_id;
                ckl::add<
                    ckl::input(
                        dfb_beta_input_id,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::OperandKind::Block),
                    ckl::input(
                        dfb_beta_id,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::OperandKind::Block),
                    ckl::output(dfb_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                    block_shape);
            }
        }

        dfb_xmm_runtime_id = dfb_xmm_id;
        dfb_ex2pe_obj.pop_front(onetile);
        dfb_ex_obj.pop_front(onetile);
    }  // NCHt loop
    // The single eps tile is waited once and reused across all NCHt iterations; pop it at the end
    // so the DFB is left balanced.
    dfb_eps_obj.pop_front(onetile);
}
