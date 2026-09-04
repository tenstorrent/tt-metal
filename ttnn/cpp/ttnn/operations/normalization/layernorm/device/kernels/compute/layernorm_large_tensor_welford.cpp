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
#include "experimental/kernel_args.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

namespace generic = norm::kernel_util::generic;

template <
    uint32_t dfb_in,
    uint32_t dfb_inb,
    uint32_t dfb_interm_pre_add,
    uint32_t dfb_ex,
    uint32_t dfb_ex2,
    uint32_t dfb_ex_welford,
    uint32_t dfb_ex2_welford,
    bool welford_state_fp32_alias,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t var_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_inb_obj(dfb_inb);
    DataflowBuffer dfb_interm_pre_add_obj(dfb_interm_pre_add);
    DataflowBuffer dfb_ex_obj(dfb_ex);
    DataflowBuffer dfb_ex2_obj(dfb_ex2);
    // When welford_state_fp32_alias is true these are c_30/c_31; distinct buffer indices
    // sharing dfb_ex/dfb_ex2's SRAM allocations but configured with UnpackToDestFp32.
    // When false, dfb_ex_welford == dfb_ex and dfb_ex2_welford == dfb_ex2.
    DataflowBuffer dfb_ex_welford_obj(dfb_ex_welford);
    DataflowBuffer dfb_ex2_welford_obj(dfb_ex2_welford);

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
        // Must be done in compute: dfb_ex / dfb_ex2 hold welford state (mean / M2) which are
        // produced by pack_tile below; the reader never writes these DFBs. Aliases share SRAM
        // but have independent read/write counters and need to be kept in sync so the next
        // block's wait_front on the aliases (used by copy_tile for fp32 precision) sees the data.
        dfb_ex_welford_obj.reserve_back(1);
        dfb_ex2_welford_obj.reserve_back(1);
    }
    tile_regs_wait();
    pack_reconfig_data_format(dfb_ex);
    pack_tile(mean_dst, dfb_ex);
    pack_tile(var_dst, dfb_ex2);
    tile_regs_release();
    dfb_ex_obj.push_back(1);
    dfb_ex2_obj.push_back(1);
    if constexpr (welford_state_fp32_alias) {
        dfb_ex_welford_obj.push_back(1);
        dfb_ex2_welford_obj.push_back(1);
    }

    for (auto block : generic::blocks(Wt, blk)) {
        const auto block_shape =
            ckl::IterationShape::tiles(block.size()).block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
        // Keep pre-add in a separate DFB to avoid the transpose_dest aliasing issue.
        ckl::add<
            ckl::input(
                dfb_in, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::input(
                dfb_inb, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::output(dfb_interm_pre_add, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
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
        // preserves the full FP32 precision. Otherwise, dfb_ex_welford == dfb_ex.
        reconfig_data_format_srca(dfb_in, dfb_ex_welford);
        copy_tile_init(dfb_ex_welford);
        copy_tile(dfb_ex_welford, 0, mean_dst);
        reconfig_data_format_srca(dfb_ex_welford, dfb_ex2_welford);
        copy_tile_to_dst_init_short_with_dt(dfb_ex_welford, dfb_ex2_welford);
        copy_tile(dfb_ex2_welford, 0, var_dst);
        welford_restore_state(mean_dst);

        reconfig_data_format_srca(dfb_ex2_welford, dfb_interm_pre_add);
        transpose_init(dfb_interm_pre_add);
        for (auto i : block.local()) {
            // Welford's needs transposed input tile
            transpose_tile(dfb_interm_pre_add, i, input_dst);

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
            // pack_tile below is the producer of dfb_ex / dfb_ex2.
            dfb_ex_welford_obj.reserve_back(1);
            dfb_ex2_welford_obj.reserve_back(1);
        }
        tile_regs_wait();
        pack_reconfig_data_format(dfb_interm_pre_add, dfb_ex);
        pack_tile(mean_dst, dfb_ex);
        pack_tile(var_dst, dfb_ex2);
        tile_regs_release();
        dfb_ex_obj.push_back(1);
        dfb_ex2_obj.push_back(1);
        if constexpr (welford_state_fp32_alias) {
            dfb_ex_welford_obj.push_back(1);
            dfb_ex2_welford_obj.push_back(1);
        }
    }

    reconfig_data_format_srca(dfb_interm_pre_add, dfb_ex_welford);

    dfb_ex_obj.wait_front(1);
    dfb_ex2_obj.wait_front(1);
    if constexpr (welford_state_fp32_alias) {
        dfb_ex_welford_obj.wait_front(1);
        dfb_ex2_welford_obj.wait_front(1);
    }
    tile_regs_acquire();
    // Reload through the FP32 alias before finalizing.
    copy_tile_init(dfb_ex_welford);
    copy_tile(dfb_ex_welford, 0, mean_dst);
    copy_tile_to_dst_init_short_with_dt(dfb_ex_welford, dfb_ex2_welford);
    copy_tile(dfb_ex2_welford, 0, var_dst);
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
 * @param: dfb_in: input DFB
 * @param: input_dst: input tile for Welford's algorithm
 * @param: mean_dst: mean tile for Welford's algorithm
 * @param: Wt: width of the input in tiles
 * @param: tile_width: width of each tile
 * @param: W: width of the input
 * @param: p_reciprocals: pointer to the reciprocal LUT
 */
template <
    uint32_t dfb_in,
    uint32_t dfb_x_welford,
    bool welford_fp32_alias,
    uint32_t dfb_ex,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void welford_no_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_x_welford_obj(dfb_x_welford);

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    // When last tile is full the value is 0 and is not used because full update is done.
    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = (last_tile_rows == 0);

    uint32_t sample_idx = 0;
    reconfig_data_format_srca(dfb_x_welford);
    // Reconfigure the transpose op for the welford intake DFB. When the alias is active,
    // dfb_x_welford has UnpackToDestFp32 mode so transpose_tile preserves fp32 precision.
    transpose_init(dfb_x_welford);
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
            transpose_init(dfb_x_welford);
        } else {
            dfb_in_obj.wait_front(1);
        }
        // Welford's needs transposed input tile
        transpose_tile(dfb_x_welford, 0, input_dst);
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
        transpose_init(dfb_x_welford);
    } else {
        dfb_in_obj.wait_front(num_to_sync);
    }
    transpose_tile(dfb_x_welford, 0, input_dst);
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

    uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t blk = get_arg(args::block_size);
    constexpr uint32_t do_gamma = get_arg(args::do_gamma);
    constexpr uint32_t do_beta = get_arg(args::do_beta);
    constexpr bool fuse_pre_add = get_arg(args::fuse_pre_add) == 1;
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dest_acc_en) == 1;
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t tile_width = get_arg(args::tile_width);
    // welford_fp32_alias: when true, dfb_x_welford is a multi-buffer-index alias of dfb_x
    // configured with UnpackToDestFp32 so the welford section reads full fp32 into DEST
    // while the post-welford eltwise still reads dfb_x via SrcA (Tf32).
    // When false, dfb_x_welford == dfb_x.
#ifdef WELFORD_FP32_ALIAS
    constexpr bool welford_fp32_alias = true;
#else
    constexpr bool welford_fp32_alias = false;
#endif

    // welford_state_fp32_alias: when true, dfb_ex_welford/dfb_ex2_welford are c_30/c_31
    // multi-buffer-index aliases of dfb_ex (c_18) / dfb_ex2 (c_19) configured for UnpackToDestFp32.
    // The fused welford path's per-block copy_tile reads of the running mean / M2 use
    // these aliases to take the Dst fp32 path (preserves FP32 precision) instead of the
    // SrcA Tf32 path. When false, dfb_ex_welford == dfb_ex and dfb_ex2_welford == dfb_ex2.
 #ifdef WELFORD_STATE_FP32_ALIAS
    constexpr bool welford_state_fp32_alias = true;
 #else
    constexpr bool welford_state_fp32_alias = false;
 #endif

    // Note that the entire W dimension must fit in the xmm DFB for this kernel to be correct.
    constexpr auto dfb_eps = dfb::eps;
    constexpr auto dfb_in = dfb::in;
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_inb = dfb::inb;
    constexpr auto dfb_interm_pre_add = dfb::x;
#endif
    constexpr auto dfb_out = dfb::out;
#ifdef FUSE_GAMMA
    constexpr auto dfb_gamma = dfb::gamma;
#else
    constexpr auto dfb_gamma = dfb_out;
#endif
#ifdef FUSE_BETA
    constexpr auto dfb_beta = dfb::beta;
#else
    constexpr auto dfb_beta = dfb_out;
#endif
    constexpr auto dfb_xmm = dfb::xmm;
    constexpr uint32_t dfb_xmm_runtime = (do_gamma == 1 || do_beta == 1) ? dfb_xmm : dfb_out;
    constexpr auto dfb_ex = dfb::ex;
    constexpr auto dfb_ex2 = dfb::ex2;
    constexpr auto dfb_ex2pe = dfb::ex2pe;
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    constexpr auto dfb_fusion = dfb::fusion;
#else
    constexpr auto dfb_fusion = dfb_out;
#endif
    constexpr auto dfb_reciprocals = dfb::reciprocals;

#ifdef FUSE_PRE_ADD
    constexpr auto dfb_x = dfb_interm_pre_add;
#else
    constexpr auto dfb_x = dfb_in;
#endif
#ifdef WELFORD_FP32_ALIAS
    constexpr auto dfb_x_welford = dfb::x_welford;
#else
    constexpr auto dfb_x_welford = dfb_x;
#endif
#ifdef WELFORD_STATE_FP32_ALIAS
    constexpr auto dfb_ex_welford = dfb::ex_welford;
    constexpr auto dfb_ex2_welford = dfb::ex2_welford;
#else
    constexpr auto dfb_ex_welford = dfb_ex;
    constexpr auto dfb_ex2_welford = dfb_ex2;
#endif

    DataflowBuffer dfb_eps_obj(dfb_eps);
    DataflowBuffer dfb_in_obj(dfb_in);
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_inb_obj(dfb_inb);
#endif
    DataflowBuffer dfb_ex_obj(dfb_ex);
    DataflowBuffer dfb_ex2_obj(dfb_ex2);
    DataflowBuffer dfb_ex2pe_obj(dfb_ex2pe);
    DataflowBuffer dfb_xmm_runtime_obj(dfb_xmm_runtime);

    constexpr uint32_t onetile = 1;

    // Initialize the hardware based on the first op
    // that will be done
#ifdef FUSE_PRE_ADD
        // Init for x = in + b
        compute_kernel_hw_startup(dfb_in, dfb_inb, dfb_interm_pre_add);
#else
        // Init for transpose
        constexpr auto first_out_dfb = dfb_ex;
        compute_kernel_hw_startup(dfb_in, first_out_dfb);
#endif

    dfb_eps_obj.wait_front(onetile);  // comes from the reader

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;  // Input tile for Welford's algorithm
    constexpr uint32_t mean_dst = 1;   // Mean tile for Welford's
    constexpr uint32_t var_dst = 2;    // Variance tile for Welford's

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(dfb_reciprocals, 0);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Depending on whether we need to fuse pre-add, the approach for welford is different.
        // So we move it to a separate function.
#ifdef FUSE_PRE_ADD
            welford_fuse_pre_add<
                dfb_in,
                dfb_inb,
                dfb_interm_pre_add,
                dfb_ex,
                dfb_ex2,
                dfb_ex_welford,
                dfb_ex2_welford,
                welford_state_fp32_alias,
                input_dst,
                mean_dst,
                var_dst,
                Wt,
                tile_width,
                W,
                blk>(*p_reciprocals);
#else
            welford_no_fuse_pre_add<
                dfb_in,
                dfb_x_welford,
                welford_fp32_alias,
                dfb_ex,
                input_dst,
                mean_dst,
                Wt,
                tile_width,
                W,
                blk>(*p_reciprocals);
#endif
        // We should expect that either of the two would have have populated dst regs with mean and
        // variance in mean_dst and var_dst respectively.

        dfb_ex_obj.reserve_back(onetile);
        dfb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb_ex);
        pack_tile(mean_dst, dfb_ex);
        pack_tile(var_dst, dfb_ex2);
        tile_regs_release();
        dfb_ex_obj.push_back(onetile);
        dfb_ex2_obj.push_back(onetile);

        // Transpose mean and variance back to
        // columns and pack back to DFBs
        reconfig_data_format_srca(dfb_ex);
        transpose_init(dfb_ex);

        dfb_ex_obj.wait_front(onetile);
        dfb_ex2_obj.wait_front(onetile);
        tile_regs_acquire();
        transpose_tile(dfb_ex, 0, mean_dst);
        transpose_tile(dfb_ex2, 0, var_dst);
        tile_regs_commit();
        dfb_ex_obj.pop_front(onetile);
        dfb_ex2_obj.pop_front(onetile);

        dfb_ex_obj.reserve_back(onetile);
        dfb_ex2_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb_ex);
        pack_tile(mean_dst, dfb_ex);
        pack_reconfig_data_format(dfb_ex2);
        pack_tile(var_dst, dfb_ex2);
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
                ckl::input(dfb_ex2),
                ckl::input(dfb_eps, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_ex2pe,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Disabled)>{});

        // broadcasts the tile since dfb_ex2pe is a column vector that contains the important data
        ckl::unary_bcast<ckl::BroadcastDim::Col, ckl::input(dfb_ex2pe), ckl::output(dfb_ex2pe)>(
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

        // Lockstep the dfb_x_welford alias's read/write pointers with dfb_in's across the eltwise pass.
        // The reader pushes dfb_x_welford in pass 2 to match its pass 1 push (see
        // reader_unary_interleaved_ln_large_tensor_welford.cpp); compute pops it here to match
        // dfb_in's pop. Both share SRAM but have independent state; popping dfb_x_welford keeps it aligned
        // with dfb_in so the next NCHt Welford iteration reads from the correct SRAM offset after DFB wrap.
        DataflowBuffer dfb_x_welford_obj_eltwise(dfb_x_welford);

        for (auto block : generic::blocks(Wt, blk)) {
            const auto block_shape = ckl::IterationShape::tiles(block.size())
                                         .block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
            // Last block may only be partially-filled,
            // and only tiles that have data in them are
            // processed, but need to sync with reader on full blocks
            dfb_in_obj.wait_front(block.full_block_size());
            if constexpr (welford_fp32_alias && !fuse_pre_add) {
                // dfb_x_welford was pushed by the reader in pass 2; wait for the push and pop in
                // lockstep with dfb_in. We do not actually read dfb_x_welford in the eltwise pass
                // (FPU consumes dfb_in via SrcA); this is purely a FIFO-pointer sync.
                dfb_x_welford_obj_eltwise.wait_front(block.full_block_size());
            }
            tile_regs_acquire();
            reconfig_data_format(dfb_in, dfb_ex);
            sub_bcast_cols_init(dfb_in, dfb_ex);
            // x-E[x]
            for (auto i : block.local()) {
                sub_tiles_bcast_cols(dfb_in, dfb_ex, i, 0, i);
            }
            dfb_in_obj.pop_front(block.full_block_size());
            if constexpr (welford_fp32_alias && !fuse_pre_add) {
                dfb_x_welford_obj_eltwise.pop_front(block.full_block_size());
            }

#ifdef FUSE_PRE_ADD
            // Fuse in = in + b
            reconfig_data_format_srca(dfb_in, dfb_inb);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_inb);
            dfb_inb_obj.wait_front(block.full_block_size());
            for (auto i : block.local()) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_inb, i, i);
            }
            dfb_inb_obj.pop_front(block.full_block_size());
#endif

            // Multiply by 1/(√(Var(X) + ε)). SrcA currently holds dfb_inb (fused) or dfb_in
            // (non-fused), the last operand read above; switch it to dfb_ex2pe's format.
#ifdef FUSE_PRE_ADD
            reconfig_data_format_srca(dfb_inb, dfb_ex2pe);
#else
            reconfig_data_format_srca(dfb_in, dfb_ex2pe);
#endif
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_ex2pe);
            for (auto i : block.local()) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_ex2pe, 0 /*in_tile_index*/, i);
            }
            tile_regs_commit();

            pack_reconfig_data_format(dfb_xmm_runtime);
            // Sync with writer on full blocks
            dfb_xmm_runtime_obj.reserve_back(block.full_block_size());
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, dfb_xmm_runtime);
            }
            dfb_xmm_runtime_obj.push_back(block.full_block_size());
            tile_regs_release();

            if constexpr (do_gamma == 1) {
                constexpr auto dfb_gamma_out = do_beta ? dfb_fusion : dfb_out;
                // Multiply by gamma
                ckl::mul<
                    ckl::input(
                        dfb_xmm,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block),
                    ckl::input(
                        dfb_gamma,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block),
                    ckl::output(dfb_gamma_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                    block_shape);
            }

            if constexpr (do_beta == 1) {
                constexpr auto dfb_beta_input = do_gamma ? dfb_fusion : dfb_xmm;
                ckl::add<
                    ckl::input(
                        dfb_beta_input,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block),
                    ckl::input(
                        dfb_beta,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block),
                    ckl::output(dfb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(block_shape);
            }
        }

        dfb_ex2pe_obj.pop_front(onetile);
        dfb_ex_obj.pop_front(onetile);
    }  // NCHt loop
    // The single eps tile is waited once and reused across all NCHt iterations; pop it at the end
    // so the DFB is left balanced.
    dfb_eps_obj.pop_front(onetile);
}
