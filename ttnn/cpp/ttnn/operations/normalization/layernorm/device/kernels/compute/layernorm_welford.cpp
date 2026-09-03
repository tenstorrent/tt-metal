// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/welford.h"
#include "api/compute/transpose.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

namespace kutil = norm::kernel_util;
namespace generic = kutil::generic;

void kernel_main() {
    uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t blk = get_arg(args::block_size);
    constexpr uint32_t do_gamma = get_arg(args::do_gamma);
    constexpr uint32_t do_beta = get_arg(args::do_beta);
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dest_acc_en) == 1;
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t tile_width = get_arg(args::tile_width);
    constexpr bool fuse_pre_add = static_cast<bool>(get_arg(args::fuse_pre_add));

    constexpr uint32_t onetile = 1;

    constexpr auto dfb_eps_id = dfb::eps;
    constexpr auto dfb_in_id = dfb::in;
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_inb_id = dfb::inb;
#endif
    constexpr auto dfb_out_id = dfb::out;
#ifdef FUSE_GAMMA
    constexpr auto dfb_gamma_id = dfb::gamma;
#else
    constexpr auto dfb_gamma_id = dfb_out_id;
#endif
#ifdef FUSE_BETA
    constexpr auto dfb_beta_id = dfb::beta;
#else
    constexpr auto dfb_beta_id = dfb_out_id;
#endif
    constexpr auto dfb_xmm_id = dfb::xmm;
    constexpr auto dfb_ex_id = dfb::ex;
    constexpr auto dfb_ex2_id = dfb::ex2;
    constexpr auto dfb_ex2pe_id = dfb::ex2pe;
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    constexpr auto dfb_fusion_id = dfb::fusion;
#else
    constexpr auto dfb_fusion_id = dfb_out_id;
#endif
    constexpr auto dfb_reciprocals_id = dfb::reciprocals;
    DataflowBuffer dfb_eps_obj(dfb_eps_id);
    DataflowBuffer dfb_xmm_obj(dfb_xmm_id);
    DataflowBuffer dfb_ex_obj(dfb_ex_id);
    DataflowBuffer dfb_ex2_obj(dfb_ex2_id);
    DataflowBuffer dfb_ex2pe_obj(dfb_ex2pe_id);

#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    constexpr auto dfb_im_or_out_id = dfb_fusion_id;
#else
    constexpr auto dfb_im_or_out_id = dfb_out_id;
#endif

    //  Either in or in + b if doing fused pre-add
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_x_id = dfb::x;
#else
    constexpr auto dfb_x_id = dfb_in_id;
#endif
    DataflowBuffer dfb_x_obj(dfb_x_id);

    // Welford-fp32 alias of dfb_x_id. Shares SRAM with dfb_x_id but has its own buffer index
    // configured with UnpackToDestFp32. Welford's transpose_tile reads
    // through dfb_x_welford_id to get full fp32 into DEST; the post-welford eltwise keeps reading
    // dfb_x_id via SrcA. When welford_fp32_alias is false, dfb_x_welford_id == dfb_x_id.
#ifdef WELFORD_FP32_ALIAS
    constexpr auto dfb_x_welford_id = dfb::x_welford;
    constexpr bool welford_fp32_alias = true;
#else
    constexpr auto dfb_x_welford_id = dfb_x_id;
    constexpr bool welford_fp32_alias = false;
#endif
    DataflowBuffer dfb_x_welford_obj(dfb_x_welford_id);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;  // Input tile for Welford's algorithm
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    constexpr uint32_t last_tile_rows = (W % tile_width) == 0 ? tile_width : W % tile_width;

    dfb_eps_obj.wait_front(1);  // comes from the reader

#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb_in_id, dfb_inb_id, dfb_x_id);
    pack_reconfig_data_format(dfb_x_id);
#else
    compute_kernel_hw_startup(dfb_in_id, dfb_ex_id);
    pack_reconfig_data_format(dfb_ex_id);
#endif

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(dfb_reciprocals_id, 0);

    // Intermediate buffers need to be reserved/pushed/popped
    // in full blocks
    const auto total_buffer_size = generic::blocks(Wt, blk).total_with_remainder();

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef FUSE_PRE_ADD
            for (auto block : generic::blocks(Wt, blk)) {
                const auto block_shape = ckl::IterationShape::tiles(block.size())
                                             .block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
                if constexpr (welford_fp32_alias) {
                    // Must be done in the compute kernel: on the fuse_pre_add path compute is the
                    // producer of dfb_x_id via the add_tiles -> pack_tile sequence below; the reader
                    // never writes dfb_x_id. Push the alias alongside dfb_x_id so Welford's wait_front on
                    // dfb_x_welford_id sees the tiles.
                    dfb_x_welford_obj.reserve_back(block.full_block_size());
                }
                // In/inb come from the reader and need to be
                // synced on full block size. Keep dfb_x_id aligned
                // to full block size as well so pre-add/no-pre-add
                // can be handled the same way.
                ckl::add<
                    ckl::input(
                        dfb_in_id,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block),
                    ckl::input(
                        dfb_inb_id,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block),
                    ckl::output(dfb_x_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                    block_shape);
                if constexpr (welford_fp32_alias) {
                    dfb_x_welford_obj.push_back(block.full_block_size());
                }
            }
            reconfig_data_format(dfb_in_id, dfb_x_id, dfb_inb_id, dfb_ex_id);
#endif

        // Simultaneous calculation of E[x] and Var[x] using Welford's algorithm.
        //
        // Welford reads input tiles through dfb_x_welford_id, which shares SRAM with dfb_x_id but
        // is configured for UnpackToDestFp32 (vs dfb_x_id's default TF32 SrcA path).
        //
        // The post-welford eltwise reads dfb_x_id directly (FPU binary ops can't use UnpackToDest).
        // dfb_x_id and dfb_x_welford_id have independent read/write pointers so we wait_front and pop_front
        // them separately. When welford_fp32_alias is 0, dfb_x_welford_id == dfb_x_id so the two sets
        // of semaphore ops collapse onto the same DFB and the alias-side waits/pops are
        // redundant -- gated by welford_fp32_alias.
        uint32_t start_N = 0;
        reconfig_data_format_srca(dfb_x_welford_id);
        // Reconfigure the transpose op for the welford intake DFB. When the alias is active,
        // dfb_x_welford_id has UnpackToDestFp32 mode so transpose_tile preserves fp32 precision.
        transpose_init(dfb_x_welford_id);
        tile_regs_acquire();
        welford_init();
        // Welford's recurrence and the fp32 transpose collide in the math thread's replay
        // buffer. The buffer has 32 slots, conventionally split between SFPU [0, 16) and
        // FPU [16, 32). Welford violates that split: welford_init records 32 instructions
        // at slots [0, 32) (4 LREG variants of 8 instructions each, fully unrolled), and
        // welford_update replays all four variants per block.
        //
        // When dfb_x_welford_id is configured for UnpackToDest fp32 (welford_fp32_alias=true),
        // transpose_tile takes the UnpackToDest path. Its math-side init
        // (llk_math_transpose_dest_init, invoked from transpose_init inside the loop
        // below) records 16 instructions at slots [16, 32) for the transpose-dest setup,
        // clobbering welford's LREG2/LREG3 portions. The recovery after each transpose_tile
        // re-records all 32 slots with the welford recurrence so welford_update replays welford
        // ops instead of stale transpose-dest ops. LREG4/5 (the running mean / M2 accumulator)
        // survive transpose_dest because it only uses FPU MOVs.
        //
        // When welford_fp32_alias is false (e.g. fp32_dest_acc_en=false path), the unpack dst
        // format is not Float32 so transpose_tile takes the SrcA path. That path skips
        // llk_math_transpose_dest entirely, so the math-thread replay buffer is untouched
        // and no recovery is needed.
        // Process all but the last tile
        for (uint32_t wt = 0; wt < (Wt - 1); ++wt) {
            if constexpr (welford_fp32_alias) {
                dfb_x_welford_obj.wait_front(wt + 1);
                // SFPU replay slots [0, 32) currently hold the welford recurrence (see outer
                // comment block above). transpose_init re-records slots [16, 32) with
                // the transpose-dest setup so transpose_tile below can replay them.
                transpose_init(dfb_x_welford_id);
            } else {
                dfb_x_obj.wait_front(wt + 1);
            }
            transpose_tile(dfb_x_welford_id, wt, input_dst);
            if constexpr (welford_fp32_alias) {
                // transpose_tile took the UnpackToDestFp32 path. Its math-side init clobbered
                // the welford recurrence at SFPU replay slots [16, 32).
                // welford_init<WelfordInitMode::PreserveStats>() re-records all 32 slots with
                // the welford recurrence; PreserveStats keeps the running mean / M2 accumulator
                // in LREG4/5. UNPACK A is left in transpose=1;
                // welford_update is pure SFPU and does not consume that state, and the next
                // iteration's transpose_init reprograms it.
                welford_init<WelfordInitMode::PreserveStats>();
            }
            welford_update<W>(input_dst, start_N, *p_reciprocals);
            start_N += tile_width;
        }

        // Process the last tile
        // dfb_x_id is synced on full blocks, so we need to wait for the
        // last tile + any remaining in the last block
        const auto num_to_wait = generic::blocks(Wt, blk).total_with_remainder();
        if constexpr (welford_fp32_alias) {
            dfb_x_welford_obj.wait_front(num_to_wait);
            transpose_init(dfb_x_welford_id);
        } else {
            dfb_x_obj.wait_front(num_to_wait);
        }
        transpose_tile(dfb_x_welford_id, Wt - 1, input_dst);
        if constexpr (welford_fp32_alias) {
            welford_init<WelfordInitMode::PreserveStats>();
        }
        welford_update_rows<W>(input_dst, start_N, 0, last_tile_rows, *p_reciprocals);

        // Store the mean and variance to the destination registers
        welford_finalize_to_row<W>(mean_dst, W - 1, *p_reciprocals);
        tile_regs_commit();

        // Pop dfb_x_welford_id so its rd_ptr advances in lock-step with dfb_x_id's pop in the eltwise
        // loop below. Multi-buffer-index DFB indices have independent read/write pointers
        // but share the underlying SRAM; popping the alias only
        // advances dfb_x_welford_id's own rd_ptr, leaving dfb_x_id's state untouched. Without this
        // pop, subsequent NCHt iterations would read stale tiles from the start of the buffer
        // (the reader's push_back advances the wr_ptr, but the alias's rd_ptr stays at 0).
        if constexpr (welford_fp32_alias) {
            dfb_x_welford_obj.pop_front(total_buffer_size);
        }

        // Transpose mean and var back to columns
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

        dfb_ex_obj.wait_front(onetile);
        dfb_ex2_obj.wait_front(onetile);
        reconfig_data_format_srca(dfb_ex_id);
        transpose_init(dfb_ex_id);
        tile_regs_acquire();
        transpose_tile(dfb_ex_id, 0, mean_dst);
        transpose_tile(dfb_ex2_id, 0, var_dst);
        tile_regs_commit();

        dfb_ex_obj.pop_front(onetile);
        dfb_ex2_obj.pop_front(onetile);

        dfb_ex_obj.reserve_back(onetile);
        dfb_ex2_obj.reserve_back(onetile);

        pack_reconfig_data_format(dfb_ex_id);
        tile_regs_wait();
        pack_tile(mean_dst, dfb_ex_id);
        pack_reconfig_data_format(dfb_ex2_id);
        pack_tile(var_dst, dfb_ex2_id);
        tile_regs_release();

        dfb_ex_obj.push_back(onetile);
        dfb_ex2_obj.push_back(onetile);

        // Reuse dfb_x_id since we didn't pop anything from it
        dfb_ex_obj.wait_front(onetile);  // should have 1 tile
        ckl::sub<
            ckl::input(
                dfb_x_id, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::input(dfb_ex_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::output(
                dfb_xmm_id,
                ckl::ReservePolicy::Upfront,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>(
            ckl::IterationShape::tiles(total_buffer_size).block_size(/*block_size=*/blk));
        dfb_ex_obj.pop_front(1);
        dfb_xmm_obj.wait_front(total_buffer_size);

        if constexpr (!fuse_pre_add) {
            reconfig_data_format_srca(dfb_x_id, dfb_xmm_id);
        }

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

        // Gamma and beta each contain one row and remain resident across all NCHt rows; tile
        // offsets select the current width block. TODO: wait on gamma/beta only on the first NCHt row.
        // Remainder of the layernorm operation
        // norm(x) * gamma + beta,
        // where norm(x) is:
        // (x - E[x]) / sqrt(E[(x-E[x])^2] + eps)
        dfb_ex2pe_obj.wait_front(onetile);
        for (auto block : generic::blocks(Wt, blk)) {
            const auto block_shape = ckl::IterationShape::tiles(block.size())
                                         .block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
            ckl::eltwise_chain(
                block_shape,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        dfb_xmm_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset),
                    ckl::input(dfb_ex2pe_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{
                    block.start(), 0u},
                // pack either to intermediate (dfb_fusion or out0)
                // if no gamma/beta are provided, this will be passed on to the writer
                ckl::PackTile<ckl::output(
                    dfb_im_or_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});

#ifdef FUSE_GAMMA
            if constexpr (do_gamma) {
                constexpr uint32_t dfb_outg_id = do_beta ? dfb_fusion_id : dfb_out_id;
                ckl::eltwise_chain(
                    block_shape,
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            dfb_fusion_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb_gamma_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileAddressing::Offset)>{0u, block.start()},
                    // pack either to intermediate (dfb_fusion or out0)
                    ckl::PackTile<ckl::output(
                        dfb_outg_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
            }
#endif
#ifdef FUSE_BETA
            if constexpr (do_beta) {
                ckl::eltwise_chain(
                    block_shape,
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            dfb_fusion_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb_beta_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileAddressing::Offset)>{0u, block.start()},
                    ckl::PackTile<ckl::output(
                        dfb_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
            }
#endif
        }
        dfb_ex2pe_obj.pop_front(onetile);
        dfb_xmm_obj.pop_front(total_buffer_size);

    }  // NCHt loop
}
