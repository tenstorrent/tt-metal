// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/sfpu_binary_bcast.h"

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/experimental/layernorm.h"
#include "api/compute/transpose.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/dataflow_buffer.h"

namespace generic = norm::kernel_util::generic;

template <
    uint32_t dfb_in,
    uint32_t dfb_inb,
    uint32_t dfb_in_fp32,
    uint32_t dfb_inb_fp32,
    uint32_t dfb_pre_add_fp32,
    bool fp32_sfpu_finalizer,
    uint32_t dfb_interm_pre_add,
    uint32_t dfb_ex,
    uint32_t dfb_ex2,
    uint32_t dfb_ex_welford,
    uint32_t dfb_ex2_welford,
    bool welford_state_fp32_alias,
    bool fused_pre_add_replay,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t var_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void two_pass_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_inb_obj(dfb_inb);
    DataflowBuffer dfb_in_fp32_obj(dfb_in_fp32);
    DataflowBuffer dfb_inb_fp32_obj(dfb_inb_fp32);
    DataflowBuffer dfb_pre_add_fp32_obj(dfb_pre_add_fp32);
    DataflowBuffer dfb_interm_pre_add_obj(dfb_interm_pre_add);
    DataflowBuffer dfb_ex_obj(dfb_ex);
    DataflowBuffer dfb_ex2_obj(dfb_ex2);
    DataflowBuffer dfb_ex_welford_obj(dfb_ex_welford);
    DataflowBuffer dfb_ex2_welford_obj(dfb_ex2_welford);

    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = last_tile_rows == 0;
    constexpr uint32_t full_block_n = blk * tile_width;
    constexpr uint32_t last_block_start = ((Wt - 1) / blk) * blk * tile_width;
    constexpr uint32_t last_block_n = W - last_block_start;
    const uint32_t full_block_n_bits = generic::bit_cast<uint32_t>(static_cast<float>(full_block_n));
    const uint32_t last_block_n_bits = generic::bit_cast<uint32_t>(static_cast<float>(last_block_n));

    uint32_t accumulated_n = 0;
    for (auto block : generic::blocks(Wt, blk)) {
        // Materialize x = a + b once in L1. Both statistics passes below reread
        // this block locally; they do not add another traversal of a or b.
        dfb_in_obj.wait_front(block.full_block_size());
        dfb_inb_obj.wait_front(block.full_block_size());
        pack_reconfig_data_format(dfb_interm_pre_add);
        dfb_interm_pre_add_obj.reserve_back(block.full_block_size());
        if constexpr (fp32_sfpu_finalizer) {
            dfb_pre_add_fp32_obj.reserve_back(block.full_block_size());
        }
        if constexpr (fp32_sfpu_finalizer) {
            dfb_in_fp32_obj.wait_front(block.full_block_size());
            dfb_inb_fp32_obj.wait_front(block.full_block_size());
            copy_tile_to_dst_init_short(dfb_in_fp32);
            for (uint32_t i = 0; i < block.local().size(); i += 2) {
                const bool has_second_tile = i + 1 < block.local().size();
                tile_regs_acquire();
                copy_tile(dfb_in_fp32, i, 0);
                copy_tile_to_dst_init_short_with_dt(dfb_in_fp32, dfb_inb_fp32);
                copy_tile(dfb_inb_fp32, i, 1);
                if (has_second_tile) {
                    copy_tile_to_dst_init_short_with_dt(dfb_inb_fp32, dfb_in_fp32);
                    copy_tile(dfb_in_fp32, i + 1, 2);
                    copy_tile_to_dst_init_short_with_dt(dfb_in_fp32, dfb_inb_fp32);
                    copy_tile(dfb_inb_fp32, i + 1, 3);
                }
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                if (has_second_tile) {
                    add_binary_tile(2, 3, 2);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb_interm_pre_add);
                if (has_second_tile) {
                    pack_tile(2, dfb_interm_pre_add);
                }
                tile_regs_release();
                copy_tile_to_dst_init_short_with_dt(dfb_inb_fp32, dfb_in_fp32);
            }
            dfb_in_fp32_obj.pop_front(block.full_block_size());
            dfb_inb_fp32_obj.pop_front(block.full_block_size());
        } else {
            reconfig_data_format(dfb_in, dfb_inb);
            add_init(dfb_in, dfb_inb);
            tile_regs_acquire();
            for (auto i : block.local()) {
                add_tiles(dfb_in, dfb_inb, i, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, dfb_interm_pre_add);
            }
            tile_regs_release();
        }
        dfb_in_obj.pop_front(block.full_block_size());
        dfb_inb_obj.pop_front(block.full_block_size());
        dfb_interm_pre_add_obj.push_back(block.full_block_size());
        if constexpr (fp32_sfpu_finalizer) {
            dfb_pre_add_fp32_obj.push_back(block.full_block_size());
        }

        dfb_interm_pre_add_obj.wait_front(block.full_block_size());
        if constexpr (fp32_sfpu_finalizer) {
            dfb_pre_add_fp32_obj.wait_front(block.full_block_size());
        }
        if (!block.is_first()) {
            dfb_ex_obj.wait_front(1);
            dfb_ex2_obj.wait_front(1);
            if constexpr (welford_state_fp32_alias) {
                dfb_ex_welford_obj.wait_front(1);
                dfb_ex2_welford_obj.wait_front(1);
            }
        }

        tile_regs_acquire();
        if (!block.is_first()) {
            // Preserve the previous blocks' raw (mean, M2) in adjacent DEST
            // tiles. The SFPU Chan merge consumes them after this block's M2
            // has been accumulated in LREG4/5/6.
            reconfig_data_format_srca(dfb_ex_welford);
            copy_tile_init(dfb_ex_welford);
            copy_tile(dfb_ex_welford, 0, mean_dst);
            reconfig_data_format_srca(dfb_ex_welford, dfb_ex2_welford);
            copy_tile_to_dst_init_short_with_dt(dfb_ex_welford, dfb_ex2_welford);
            copy_tile(dfb_ex2_welford, 0, var_dst);
        }

        constexpr auto dfb_stats_input = fp32_sfpu_finalizer ? dfb_pre_add_fp32 : dfb_interm_pre_add;
        reconfig_data_format_srca(dfb_stats_input);
        transpose_init(dfb_stats_input);
        two_pass_stats_init_shifted();

        uint32_t block_n = 0;
        {
            constexpr uint32_t i = 0;
            const uint32_t global_tile = block.to_global(i);
            const uint32_t rows = !is_last_tile_full && global_tile == Wt - 1 ? last_tile_rows : tile_width;
            transpose_tile(dfb_stats_input, i, input_dst);
            two_pass_stats_update_shifted_rows<false, true>(input_dst, 0, rows);
            block_n += rows;
        }
        for (uint32_t i = 1; i < block.size(); ++i) {
            const uint32_t global_tile = block.to_global(i);
            const uint32_t rows = !is_last_tile_full && global_tile == Wt - 1 ? last_tile_rows : tile_width;
            transpose_tile(dfb_stats_input, i, input_dst);
            two_pass_stats_update_shifted_rows<false>(input_dst, 0, rows);
            block_n += rows;
        }
        two_pass_stats_finish_shifted_mean<true, true>(reciprocal_lut[block_n - 1]);

        for (auto i : block.local()) {
            const uint32_t global_tile = block.to_global(i);
            const uint32_t rows = !is_last_tile_full && global_tile == Wt - 1 ? last_tile_rows : tile_width;
            transpose_tile(dfb_stats_input, i, input_dst);
            two_pass_stats_update_rows<true>(input_dst, 0, rows);
        }

        if (block.is_first()) {
            two_pass_stats_save_state(mean_dst);
            two_pass_stats_save_anchor_to_state(mean_dst);
        } else {
            two_pass_stats_combine_block(
                mean_dst,
                reciprocal_lut[accumulated_n + block_n - 1],
                block.is_full() ? full_block_n_bits : last_block_n_bits);
        }
        accumulated_n += block_n;
        tile_regs_commit();

        if constexpr (!fused_pre_add_replay) {
            dfb_interm_pre_add_obj.pop_front(block.full_block_size());
        }
        if constexpr (fp32_sfpu_finalizer) {
            dfb_pre_add_fp32_obj.pop_front(block.full_block_size());
        }
        if (!block.is_first()) {
            dfb_ex_obj.pop_front(1);
            dfb_ex2_obj.pop_front(1);
            if constexpr (welford_state_fp32_alias) {
                dfb_ex_welford_obj.pop_front(1);
                dfb_ex2_welford_obj.pop_front(1);
            }
        }

        dfb_ex_obj.reserve_back(1);
        dfb_ex2_obj.reserve_back(1);
        if constexpr (welford_state_fp32_alias) {
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

    dfb_ex_obj.wait_front(1);
    dfb_ex2_obj.wait_front(1);
    if constexpr (welford_state_fp32_alias) {
        dfb_ex_welford_obj.wait_front(1);
        dfb_ex2_welford_obj.wait_front(1);
    }
    tile_regs_acquire();
    reconfig_data_format_srca(dfb_ex_welford);
    copy_tile_init(dfb_ex_welford);
    copy_tile(dfb_ex_welford, 0, mean_dst);
    reconfig_data_format_srca(dfb_ex_welford, dfb_ex2_welford);
    copy_tile_to_dst_init_short_with_dt(dfb_ex_welford, dfb_ex2_welford);
    copy_tile(dfb_ex2_welford, 0, var_dst);
    welford_restore_state(mean_dst);
    if constexpr (fp32_sfpu_finalizer) {
        two_pass_stats_finalize_to_row<false>(mean_dst, reciprocal_lut[W - 1]);
    } else {
        two_pass_stats_restore_anchor_from_state(mean_dst);
        two_pass_stats_finalize_split_mean_to_row<false>(mean_dst, reciprocal_lut[W - 1]);
    }
    tile_regs_commit();
    dfb_ex_obj.pop_front(1);
    dfb_ex2_obj.pop_front(1);
    if constexpr (welford_state_fp32_alias) {
        dfb_ex_welford_obj.pop_front(1);
        dfb_ex2_welford_obj.pop_front(1);
    }
}

template <
    uint32_t dfb_in,
    uint32_t dfb_x_welford,
    uint32_t dfb_in_fp32,
    bool welford_fp32_alias,
    bool fp32_sfpu_finalizer,
    uint32_t input_dst,
    uint32_t mean_dst,
    uint32_t Wt,
    uint32_t tile_width,
    uint32_t W,
    uint32_t blk>
void two_pass_no_fuse_pre_add(const std::array<uint32_t, W>& reciprocal_lut) {
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_x_welford_obj(dfb_x_welford);
    DataflowBuffer dfb_in_fp32_obj(dfb_in_fp32);

    constexpr uint32_t last_tile_rows = W % tile_width;
    constexpr bool is_last_tile_full = last_tile_rows == 0;
    constexpr uint32_t full_block_n = blk * tile_width;
    constexpr uint32_t last_block_start = ((Wt - 1) / blk) * blk * tile_width;
    constexpr uint32_t last_block_n = W - last_block_start;
    const uint32_t full_block_n_bits = generic::bit_cast<uint32_t>(static_cast<float>(full_block_n));
    const uint32_t last_block_n_bits = generic::bit_cast<uint32_t>(static_cast<float>(last_block_n));
    constexpr uint32_t anchor_dst = mean_dst + 2;

    reconfig_data_format_srca(dfb_x_welford);
    transpose_init(dfb_x_welford);
    tile_regs_acquire();
    two_pass_stats_init_shifted();

    uint32_t accumulated_n = 0;
    for (auto block : generic::blocks(Wt, blk)) {
        if constexpr (welford_fp32_alias) {
            dfb_x_welford_obj.wait_front(block.full_block_size());
        } else {
            dfb_in_obj.wait_front(block.full_block_size());
        }
        if constexpr (fp32_sfpu_finalizer) {
            // The finalizer's input alias shares SRAM with dfb_in but has independent FIFO
            // pointers. Consume its first-pass entry here so the second-pass entry identifies
            // the tiles re-read for normalization rather than stale first-pass storage.
            dfb_in_fp32_obj.wait_front(block.full_block_size());
        }

        uint32_t block_n = 0;
        {
            constexpr uint32_t i = 0;
            const uint32_t global_tile = block.to_global(i);
            const uint32_t rows = !is_last_tile_full && global_tile == Wt - 1 ? last_tile_rows : tile_width;
            transpose_tile(dfb_x_welford, i, input_dst);
            two_pass_stats_update_shifted_rows<false, true>(input_dst, 0, rows);
            block_n += rows;
        }
        for (uint32_t i = 1; i < block.size(); ++i) {
            const uint32_t global_tile = block.to_global(i);
            const uint32_t rows = !is_last_tile_full && global_tile == Wt - 1 ? last_tile_rows : tile_width;
            transpose_tile(dfb_x_welford, i, input_dst);
            two_pass_stats_update_shifted_rows<false>(input_dst, 0, rows);
            block_n += rows;
        }
        two_pass_stats_finish_shifted_mean<true, true>(reciprocal_lut[block_n - 1]);

        // The block remains at the CB front, so the second SFPU traversal is
        // an L1 reread rather than another DRAM traversal.
        for (auto i : block.local()) {
            const uint32_t global_tile = block.to_global(i);
            const uint32_t rows = !is_last_tile_full && global_tile == Wt - 1 ? last_tile_rows : tile_width;
            transpose_tile(dfb_x_welford, i, input_dst);
            two_pass_stats_update_rows<true>(input_dst, 0, rows);
        }

        if (block.is_first()) {
            two_pass_stats_save_anchor(anchor_dst);
        }
        if (block.is_first()) {
            two_pass_stats_save_state(mean_dst);
        } else {
            two_pass_stats_combine_block(
                mean_dst,
                reciprocal_lut[accumulated_n + block_n - 1],
                block.is_full() ? full_block_n_bits : last_block_n_bits);
        }
        accumulated_n += block_n;

        if constexpr (welford_fp32_alias) {
            dfb_x_welford_obj.pop_front(block.full_block_size());
        }
        if constexpr (fp32_sfpu_finalizer) {
            dfb_in_fp32_obj.pop_front(block.full_block_size());
        }
        dfb_in_obj.pop_front(block.full_block_size());
    }

    welford_restore_state(mean_dst);
    if constexpr (fp32_sfpu_finalizer) {
        two_pass_stats_finalize_to_row<false>(mean_dst, reciprocal_lut[W - 1]);
    } else {
        two_pass_stats_restore_anchor(anchor_dst);
        two_pass_stats_finalize_split_mean_to_row<false>(mean_dst, reciprocal_lut[W - 1]);
    }
    tile_regs_commit();
}

void kernel_main() {
    namespace kutil = norm::kernel_util;

    uint32_t NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::block_size);
    constexpr auto do_gamma = get_arg(args::do_gamma);
    constexpr auto do_beta = get_arg(args::do_beta);
    constexpr auto W = get_arg(args::W);
    constexpr auto tile_width = get_arg(args::tile_width);
#ifdef FUSE_PRE_ADD
    constexpr bool fuse_pre_add = true;
#else
    constexpr bool fuse_pre_add = false;
#endif
#ifdef FP32_SFPU_FINALIZER
    constexpr bool fp32_sfpu_finalizer = true;
#else
    constexpr bool fp32_sfpu_finalizer = false;
#endif
#ifdef FUSED_PRE_ADD_REPLAY
    constexpr bool fused_pre_add_replay = true;
#else
    constexpr bool fused_pre_add_replay = false;
#endif

    // Note that the entire W dimension must fit in the xmm buffer for this kernel to be correct.
    // Buffer handles come from the host-side bindings; the kernel never sees an index.
    constexpr auto dfb_eps = dfb::eps;  // single tile generated by the reader
    constexpr auto dfb_in = dfb::in;    // input x or a for fused pre-add (x=a+b)
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_inb = dfb::inb;           // input b for fused pre-add
    constexpr auto dfb_interm_pre_add = dfb::x;  // intermediate for fused pre-add
#else
    constexpr auto dfb_inb = dfb_in;
    constexpr auto dfb_interm_pre_add = dfb_in;
#endif
    constexpr auto dfb_out = dfb::out;  // output
#ifdef FUSE_GAMMA
    constexpr auto dfb_gamma = dfb::gamma;
#else
    constexpr auto dfb_gamma = dfb_in;
#endif
#ifdef FUSE_BETA
    constexpr auto dfb_beta = dfb::beta;
#else
    constexpr auto dfb_beta = dfb_in;
#endif
    uint32_t dfb_xmm = dfb::xmm;            // x - E[x]
    constexpr auto dfb_ex = dfb::ex;        // E[x]
    constexpr auto dfb_ex2 = dfb::ex2;      // Var[x] = E[(x-E[x])^2]
    constexpr auto dfb_ex2pe = dfb::ex2pe;  // Var[x]+ε
    constexpr auto dfb_reciprocals = dfb::reciprocals;  // Pre-computed reciprocals

    // The buffer the welford intake reads: the fused pre-add result when there is a residual,
    // otherwise the input itself.
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_x = dfb_interm_pre_add;
#else
    constexpr auto dfb_x = dfb_in;
#endif

    // welford_fp32_alias: when active, dfb_x_welford is a second buffer index over dfb_x's SRAM
    // configured with UnpackToDest so the welford section reads full fp32 into DEST
    // while the post-welford eltwise still reads dfb_x via SrcA (Tf32).
    // When inactive, the name resolves to dfb_x itself.
#ifdef WELFORD_FP32_ALIAS
    constexpr bool welford_fp32_alias = true;
    constexpr auto dfb_x_welford = dfb::x_welford;
#else
    constexpr bool welford_fp32_alias = false;
    constexpr auto dfb_x_welford = dfb_x;
#endif

#ifdef FP32_SFPU_FINALIZER
    constexpr auto dfb_in_fp32 = dfb::in_fp32;
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_inb_fp32 = dfb::inb_fp32;
    constexpr auto dfb_pre_add_fp32 = dfb::pre_add_fp32;
#else
    constexpr auto dfb_inb_fp32 = dfb_in;
#endif
    constexpr auto dfb_ex2pe_fp32 = dfb::ex2pe_fp32;
#else
    constexpr auto dfb_in_fp32 = dfb_in;
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_inb_fp32 = dfb_inb;
    constexpr auto dfb_pre_add_fp32 = dfb_interm_pre_add;
#else
    constexpr auto dfb_inb_fp32 = dfb_in;
#endif
    constexpr auto dfb_ex2pe_fp32 = dfb_ex2pe;
#endif

    // X is sized for a complete row. The statistics pass consumes its FP32 alias,
    // leaving the primary FIFO intact for the normalisation pass.
    constexpr auto dfb_x_replay = dfb_x;

    // welford_state_fp32_alias: when active, dfb_ex_welford / dfb_ex2_welford are second buffer
    // indices over dfb_ex / dfb_ex2's SRAM configured for UnpackToDest.
    // The fused welford path's per-block copy_tile reads of the running mean / M2 use
    // these aliases to take the Dst fp32 path (preserves FP32 precision) instead of the
    // SrcA Tf32 path. When inactive, the names resolve to dfb_ex / dfb_ex2.
#ifdef WELFORD_STATE_FP32_ALIAS
    constexpr bool welford_state_fp32_alias = true;
    constexpr auto dfb_ex_welford = dfb::ex_welford;
    constexpr auto dfb_ex2_welford = dfb::ex2_welford;
#else
    constexpr bool welford_state_fp32_alias = false;
    constexpr auto dfb_ex_welford = dfb_ex;
    constexpr auto dfb_ex2_welford = dfb_ex2;
#endif

    DataflowBuffer dfb_eps_obj(dfb_eps);
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_inb_obj(dfb_inb);
    DataflowBuffer dfb_out_obj(dfb_out);
    DataflowBuffer dfb_gamma_obj(dfb_gamma);
    DataflowBuffer dfb_beta_obj(dfb_beta);
    DataflowBuffer dfb_ex_obj(dfb_ex);
    DataflowBuffer dfb_ex2_obj(dfb_ex2);
    DataflowBuffer dfb_ex_welford_obj_main(dfb_ex_welford);
    DataflowBuffer dfb_ex2_welford_obj_main(dfb_ex2_welford);
    DataflowBuffer dfb_ex2pe_obj(dfb_ex2pe);
    DataflowBuffer dfb_ex2pe_fp32_obj(dfb_ex2pe_fp32);
    DataflowBuffer dfb_x_replay_obj(dfb_x_replay);
    DataflowBuffer dfb_interm_pre_add_obj(dfb_interm_pre_add);

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
    copy_init(dfb_in);
#endif

    dfb_eps_obj.wait_front(onetile);  // comes from the reader

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(dfb_reciprocals, 0);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fused pre-add materializes x = a + b before the two statistics passes.
#ifdef FUSE_PRE_ADD
        two_pass_fuse_pre_add<
            dfb_in,
            dfb_inb,
            dfb_in_fp32,
            dfb_inb_fp32,
            dfb_pre_add_fp32,
            fp32_sfpu_finalizer,
            dfb_interm_pre_add,
            dfb_ex,
            dfb_ex2,
            dfb_ex_welford,
            dfb_ex2_welford,
            welford_state_fp32_alias,
            fused_pre_add_replay,
            input_dst,
            mean_dst,
            var_dst,
            Wt,
            tile_width,
            W,
            blk>(*p_reciprocals);
#else
        two_pass_no_fuse_pre_add<
            dfb_in,
            dfb_x_welford,
            dfb_in_fp32,
            welford_fp32_alias,
            fp32_sfpu_finalizer,
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
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex_welford_obj_main.reserve_back(onetile);
            dfb_ex2_welford_obj_main.reserve_back(onetile);
        }
        tile_regs_wait();
        pack_reconfig_data_format(dfb_ex);
        pack_tile(mean_dst, dfb_ex);
        pack_tile(var_dst, dfb_ex2);
        tile_regs_release();
        dfb_ex_obj.push_back(onetile);
        dfb_ex2_obj.push_back(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex_welford_obj_main.push_back(onetile);
            dfb_ex2_welford_obj_main.push_back(onetile);
        }

        // Transpose mean and variance back to
        // columns and pack back to the buffers
        reconfig_data_format_srca(dfb_ex);
        transpose_init(dfb_ex);

        dfb_ex_obj.wait_front(onetile);
        dfb_ex2_obj.wait_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex_welford_obj_main.wait_front(onetile);
            dfb_ex2_welford_obj_main.wait_front(onetile);
        }
        tile_regs_acquire();
        constexpr auto dfb_mean_transpose = fp32_sfpu_finalizer ? dfb_ex_welford : dfb_ex;
        constexpr auto dfb_var_transpose = fp32_sfpu_finalizer ? dfb_ex2_welford : dfb_ex2;
        reconfig_data_format_srca(dfb_mean_transpose);
        transpose_init(dfb_mean_transpose);
        transpose_tile(dfb_mean_transpose, 0, mean_dst);
        transpose_tile(dfb_var_transpose, 0, var_dst);
        tile_regs_commit();
        dfb_ex_obj.pop_front(onetile);
        dfb_ex2_obj.pop_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex_welford_obj_main.pop_front(onetile);
            dfb_ex2_welford_obj_main.pop_front(onetile);
        }

        dfb_ex_obj.reserve_back(onetile);
        dfb_ex2_obj.reserve_back(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex_welford_obj_main.reserve_back(onetile);
            dfb_ex2_welford_obj_main.reserve_back(onetile);
        }
        tile_regs_wait();
        pack_reconfig_data_format(dfb_ex);
        pack_tile(mean_dst, dfb_ex);
        pack_reconfig_data_format(dfb_ex2);
        pack_tile(var_dst, dfb_ex2);
        tile_regs_release();
        dfb_ex_obj.push_back(onetile);
        dfb_ex2_obj.push_back(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex_welford_obj_main.push_back(onetile);
            dfb_ex2_welford_obj_main.push_back(onetile);
        }

        // =====================================
        // Calculate 1/(√(Var(X) + ε))
        // =====================================
        reconfig_data_format(dfb_ex2, dfb_eps);
        add_init(dfb_ex2, dfb_eps);

        dfb_ex2_obj.wait_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2_welford_obj_main.wait_front(onetile);
        }
        tile_regs_acquire();
        add_tiles(dfb_ex2, dfb_eps, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();
        dfb_ex2_obj.pop_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2_welford_obj_main.pop_front(onetile);
        }

        dfb_ex2pe_obj.reserve_back(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2pe_fp32_obj.reserve_back(onetile);
        }
        tile_regs_wait();
        pack_tile(dst0, dfb_ex2pe);
        tile_regs_release();
        dfb_ex2pe_obj.push_back(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2pe_fp32_obj.push_back(onetile);
        }

        // broadcasts the tile since dfb_ex2pe is a column vector that contains the important data
        dfb_ex2pe_obj.wait_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2pe_fp32_obj.wait_front(onetile);
        }
        tile_regs_acquire();
        reconfig_data_format_srca(dfb_ex2pe);
        // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the
        // pre-cleanup full-init behaviour) should become a targeted DST re-arm.
        compute_kernel_hw_startup(dfb_ex2pe, dfb_ex2pe);
        unary_bcast_init<BroadcastType::COL>(dfb_ex2pe);
        unary_bcast<BroadcastType::COL>(dfb_ex2pe, 0, dst0);
        dfb_ex2pe_obj.pop_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2pe_fp32_obj.pop_front(onetile);
        }
        tile_regs_commit();

        dfb_ex2pe_obj.reserve_back(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2pe_fp32_obj.reserve_back(onetile);
        }
        tile_regs_wait();
        pack_tile(dst0, dfb_ex2pe);
        tile_regs_release();
        dfb_ex2pe_obj.push_back(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2pe_fp32_obj.push_back(onetile);
        }

        // =====================================
        // Second pass over the input.
        // Computes the final value:
        //    x-E[x]
        //(---------------*𝛄)+ß
        //  √(Var(x)+ε)
        // =====================================
        dfb_ex2pe_obj.wait_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2pe_fp32_obj.wait_front(onetile);
        }
        dfb_ex_obj.wait_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex_welford_obj_main.wait_front(onetile);
        }

        // Lockstep the dfb_x_welford alias's read/write pointers with dfb_in's across the eltwise pass.
        // The reader pushes dfb_x_welford in pass 2 to match its pass 1 push (see
        // reader_unary_interleaved_ln_large_tensor_welford.cpp); compute pops it here to match
        // dfb_in's pop. Both share SRAM but have independent state; popping dfb_x_welford keeps it aligned
        // with dfb_in so the next NCHt statistics iteration reads from the correct SRAM offset after CB wrap.
        DataflowBuffer dfb_x_welford_obj_eltwise(dfb_x_welford);

        if constexpr (fp32_sfpu_finalizer) {
            DataflowBuffer dfb_in_fp32_obj_eltwise(dfb_in_fp32);
            DataflowBuffer dfb_inb_fp32_obj_eltwise(dfb_inb_fp32);
            sfpu_bcast_col_init();

            for (auto block : generic::blocks(Wt, blk)) {
                if constexpr (fused_pre_add_replay) {
                    dfb_x_replay_obj.wait_front(block.full_block_size());
                } else {
                    dfb_in_obj.wait_front(block.full_block_size());
                    dfb_in_fp32_obj_eltwise.wait_front(block.full_block_size());
                    if constexpr (fuse_pre_add) {
                        dfb_inb_obj.wait_front(block.full_block_size());
                        dfb_inb_fp32_obj_eltwise.wait_front(block.full_block_size());
                    }
                }
                if constexpr (welford_fp32_alias && !fuse_pre_add) {
                    dfb_x_welford_obj_eltwise.wait_front(block.full_block_size());
                }
                if constexpr (!(do_gamma || do_beta)) {
                    dfb_xmm = dfb_out;
                }
                DataflowBuffer dfb_normalized(dfb_xmm);
                dfb_normalized.reserve_back(block.full_block_size());
                pack_reconfig_data_format(dfb_xmm);
                constexpr uint32_t data_dst = 0;
                constexpr uint32_t residual_dst = 1;
                constexpr uint32_t mean_col_dst = 2;
                constexpr uint32_t inv_std_col_dst = 3;
                constexpr auto first_input_dfb = fused_pre_add_replay ? dfb_x_replay : dfb_in_fp32;
                copy_tile_to_dst_init_short(first_input_dfb);
                // Reuse each mean/inverse-standard-deviation load for two tiles
                // whenever the data is already materialised as a single stream.
                if constexpr (fused_pre_add_replay || !fuse_pre_add) {
                    for (std::uint32_t i = 0; i < block.local().size(); i += 2) {
                        const bool has_second_tile = i + 1 < block.local().size();
                        tile_regs_acquire();
                        copy_tile(first_input_dfb, i, data_dst);
                        if (has_second_tile) {
                            copy_tile(first_input_dfb, i + 1, residual_dst);
                        }
                        copy_tile_to_dst_init_short_with_dt(first_input_dfb, dfb_ex_welford);
                        copy_tile(dfb_ex_welford, 0, mean_col_dst);
                        copy_tile_to_dst_init_short_with_dt(dfb_ex_welford, dfb_ex2pe_fp32);
                        copy_tile(dfb_ex2pe_fp32, 0, inv_std_col_dst);
                        sfpu_normalize_bcast_col(data_dst, mean_col_dst, inv_std_col_dst);
                        if (has_second_tile) {
                            sfpu_normalize_bcast_col(residual_dst, mean_col_dst, inv_std_col_dst);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(data_dst, dfb_xmm);
                        if (has_second_tile) {
                            pack_tile(residual_dst, dfb_xmm);
                        }
                        tile_regs_release();
                        copy_tile_to_dst_init_short_with_dt(dfb_ex2pe_fp32, first_input_dfb);
                    }
                } else {
                    for (auto i : block.local()) {
                        tile_regs_acquire();
                        copy_tile(first_input_dfb, i, data_dst);
                        if constexpr (fuse_pre_add && !fused_pre_add_replay) {
                            copy_tile_to_dst_init_short_with_dt(dfb_in_fp32, dfb_inb_fp32);
                            copy_tile(dfb_inb_fp32, i, residual_dst);
                            copy_tile_to_dst_init_short_with_dt(dfb_inb_fp32, dfb_ex_welford);
                        } else {
                            copy_tile_to_dst_init_short_with_dt(first_input_dfb, dfb_ex_welford);
                        }
                        copy_tile(dfb_ex_welford, 0, mean_col_dst);
                        copy_tile_to_dst_init_short_with_dt(dfb_ex_welford, dfb_ex2pe_fp32);
                        copy_tile(dfb_ex2pe_fp32, 0, inv_std_col_dst);
                        if constexpr (fuse_pre_add && !fused_pre_add_replay) {
                            sfpu_residual_normalize_bcast_col(data_dst, residual_dst, mean_col_dst, inv_std_col_dst);
                        } else {
                            sfpu_normalize_bcast_col(data_dst, mean_col_dst, inv_std_col_dst);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(data_dst, dfb_xmm);
                        tile_regs_release();
                        copy_tile_to_dst_init_short_with_dt(dfb_ex2pe_fp32, first_input_dfb);
                    }
                }
                dfb_normalized.push_back(block.full_block_size());

                if constexpr (fused_pre_add_replay) {
                    dfb_x_replay_obj.pop_front(block.full_block_size());
                } else {
                    dfb_in_obj.pop_front(block.full_block_size());
                    dfb_in_fp32_obj_eltwise.pop_front(block.full_block_size());
                    if constexpr (fuse_pre_add) {
                        dfb_inb_obj.pop_front(block.full_block_size());
                        dfb_inb_fp32_obj_eltwise.pop_front(block.full_block_size());
                    }
                }
                if constexpr (welford_fp32_alias && !fuse_pre_add) {
                    dfb_x_welford_obj_eltwise.pop_front(block.full_block_size());
                }
                if constexpr (do_gamma == 1) {
                    reconfig_data_format(dfb_xmm, dfb_gamma);
                    tile_regs_acquire();
                    dfb_gamma_obj.wait_front(block.full_block_size());
                    DataflowBuffer(dfb_xmm).wait_front(block.full_block_size());
                    mul_bcast_rows_init(dfb_xmm, dfb_gamma);
                    for (auto i : block.local()) {
                        mul_tiles_bcast_rows(dfb_xmm, dfb_gamma, i, i, i);
                    }
                    tile_regs_commit();
                    dfb_gamma_obj.pop_front(block.full_block_size());
                    DataflowBuffer(dfb_xmm).pop_front(block.full_block_size());

                    if constexpr (!do_beta) {
                        pack_reconfig_data_format(dfb_out);
                    }
                    tile_regs_wait();
                    if constexpr (!do_beta) {
                        dfb_out_obj.reserve_back(block.full_block_size());
                        for (auto i : block.local()) {
                            pack_tile(i, dfb_out);
                        }
                        dfb_out_obj.push_back(block.full_block_size());
                    } else {
                        DataflowBuffer(dfb_xmm).reserve_back(block.full_block_size());
                        for (auto i : block.local()) {
                            pack_tile(i, dfb_xmm);
                        }
                        DataflowBuffer(dfb_xmm).push_back(block.full_block_size());
                    }
                    tile_regs_release();
                }

                if constexpr (do_beta == 1) {
                    tile_regs_acquire();
                    reconfig_data_format(dfb_xmm, dfb_beta);
                    add_bcast_rows_init(dfb_xmm, dfb_beta);
                    DataflowBuffer(dfb_xmm).wait_front(block.full_block_size());
                    dfb_beta_obj.wait_front(block.full_block_size());
                    for (auto i : block.local()) {
                        add_tiles_bcast_rows(dfb_xmm, dfb_beta, i, i, i);
                    }
                    tile_regs_commit();
                    dfb_beta_obj.pop_front(block.full_block_size());
                    DataflowBuffer(dfb_xmm).pop_front(block.full_block_size());

                    pack_reconfig_data_format(dfb_out);
                    dfb_out_obj.reserve_back(block.full_block_size());
                    tile_regs_wait();
                    for (auto i : block.local()) {
                        pack_tile(i, dfb_out);
                    }
                    tile_regs_release();
                    dfb_out_obj.push_back(block.full_block_size());
                }
            }
        } else {
            for (auto block : generic::blocks(Wt, blk)) {
                // Last block may only be partially-filled,
                // and only tiles that have data in them are
                // processed, but need to sync with reader on full blocks
                if constexpr (fused_pre_add_replay) {
                    dfb_x_replay_obj.wait_front(block.full_block_size());
                } else {
                    dfb_in_obj.wait_front(block.full_block_size());
                }
                if constexpr (welford_fp32_alias && !fuse_pre_add) {
                    // dfb_x_welford was pushed by the reader in pass 2; wait for the push and pop in
                    // lockstep with dfb_in. We do not actually read dfb_x_welford in the eltwise pass
                    // (FPU consumes dfb_in via SrcA); this is purely a FIFO-pointer sync.
                    dfb_x_welford_obj_eltwise.wait_front(block.full_block_size());
                }
                if constexpr (fuse_pre_add && !fused_pre_add_replay) {
                    // Form a+b in FP32 DEST before centring. Evaluating
                    // (a-anchor)+(anchor-mean)+b instead would cancel two values
                    // near the input base in the FPU and lose low-order variation.
                    dfb_inb_obj.wait_front(block.full_block_size());
                    tile_regs_acquire();
                    reconfig_data_format(dfb_in, dfb_inb);
                    add_init(dfb_in, dfb_inb);
                    for (auto i : block.local()) {
                        add_tiles(dfb_in, dfb_inb, i, i, i);
                    }
                    tile_regs_commit();
                    dfb_in_obj.pop_front(block.full_block_size());
                    dfb_inb_obj.pop_front(block.full_block_size());

                    dfb_interm_pre_add_obj.reserve_back(block.full_block_size());
                    tile_regs_wait();
                    pack_reconfig_data_format(dfb_interm_pre_add);
                    for (auto i : block.local()) {
                        pack_tile(i, dfb_interm_pre_add);
                    }
                    tile_regs_release();
                    dfb_interm_pre_add_obj.push_back(block.full_block_size());

                    dfb_interm_pre_add_obj.wait_front(block.full_block_size());
                    tile_regs_acquire();
                    reconfig_data_format(dfb_interm_pre_add, dfb_ex);
                    sub_bcast_cols_compensated_init(dfb_interm_pre_add, dfb_ex);
                    sub_bcast_cols_compensated(dfb_interm_pre_add, dfb_ex, 0, 0, block.size());
                    dfb_interm_pre_add_obj.pop_front(block.full_block_size());
                } else {
                    tile_regs_acquire();
                    constexpr auto dfb_normalize_in = fused_pre_add_replay ? dfb_x_replay : dfb_in;
                    reconfig_data_format(dfb_normalize_in, dfb_ex);
                    sub_bcast_cols_compensated_init(dfb_normalize_in, dfb_ex);
                    sub_bcast_cols_compensated(dfb_normalize_in, dfb_ex, 0, 0, block.size());
                    if constexpr (fused_pre_add_replay) {
                        dfb_x_replay_obj.pop_front(block.full_block_size());
                    } else {
                        dfb_in_obj.pop_front(block.full_block_size());
                    }
                    if constexpr (welford_fp32_alias && !fuse_pre_add) {
                        dfb_x_welford_obj_eltwise.pop_front(block.full_block_size());
                    }
                }

                // Switch SrcA from the last normalisation input to dfb_ex2pe's format.
                constexpr auto dfb_normalize_srca = fuse_pre_add && !fused_pre_add_replay
                                                        ? dfb_interm_pre_add
                                                        : (fused_pre_add_replay ? dfb_x_replay : dfb_in);
                reconfig_data_format_srca(dfb_normalize_srca, dfb_ex2pe);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_ex2pe);
                for (auto i : block.local()) {
                    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_ex2pe, 0 /*in_tile_index*/, i);
                }
                tile_regs_commit();

                if constexpr (!(do_gamma == 1 or do_beta == 1)) {
                    dfb_xmm = dfb_out;
                }

                pack_reconfig_data_format(dfb_xmm);
                // Sync with writer on full blocks
                DataflowBuffer(dfb_xmm).reserve_back(block.full_block_size());
                tile_regs_wait();
                for (auto i : block.local()) {
                    pack_tile(i, dfb_xmm);
                }
                DataflowBuffer(dfb_xmm).push_back(block.full_block_size());
                tile_regs_release();

                if constexpr (do_gamma == 1) {
                    // Multiply by gamma
                    reconfig_data_format(dfb_xmm, dfb_gamma);
                    tile_regs_acquire();
                    dfb_gamma_obj.wait_front(block.full_block_size());
                    DataflowBuffer(dfb_xmm).wait_front(block.full_block_size());
                    mul_bcast_rows_init(dfb_xmm, dfb_gamma);
                    for (auto i : block.local()) {
                        mul_tiles_bcast_rows(dfb_xmm, dfb_gamma, i, i, i);
                    }
                    tile_regs_commit();
                    dfb_gamma_obj.pop_front(block.full_block_size());
                    DataflowBuffer(dfb_xmm).pop_front(block.full_block_size());

                    if constexpr (!do_beta) {
                        pack_reconfig_data_format(dfb_out);
                    }
                    tile_regs_wait();
                    if constexpr (!do_beta) {
                        dfb_out_obj.reserve_back(block.full_block_size());
                        for (auto i : block.local()) {
                            pack_tile(i, dfb_out);
                        }
                        dfb_out_obj.push_back(block.full_block_size());
                    } else {
                        DataflowBuffer(dfb_xmm).reserve_back(block.full_block_size());
                        for (auto i : block.local()) {
                            pack_tile(i, dfb_xmm);
                        }
                        DataflowBuffer(dfb_xmm).push_back(block.full_block_size());
                    }
                    tile_regs_release();
                }

                if constexpr (do_beta == 1) {
                    // Add beta
                    tile_regs_acquire();
                    reconfig_data_format(dfb_xmm, dfb_beta);
                    add_bcast_rows_init(dfb_xmm, dfb_beta);
                    DataflowBuffer(dfb_xmm).wait_front(block.full_block_size());
                    dfb_beta_obj.wait_front(block.full_block_size());
                    for (auto i : block.local()) {
                        add_tiles_bcast_rows(dfb_xmm, dfb_beta, i, i, i);
                    }
                    tile_regs_commit();
                    dfb_beta_obj.pop_front(block.full_block_size());
                    DataflowBuffer(dfb_xmm).pop_front(block.full_block_size());

                    pack_reconfig_data_format(dfb_out);
                    dfb_out_obj.reserve_back(block.full_block_size());
                    tile_regs_wait();
                    for (auto i : block.local()) {
                        pack_tile(i, dfb_out);
                    }
                    tile_regs_release();
                    dfb_out_obj.push_back(block.full_block_size());
                }
            }
        }

        dfb_xmm = dfb::xmm;  // x minus mean
        dfb_ex2pe_obj.pop_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex2pe_fp32_obj.pop_front(onetile);
        }
        dfb_ex_obj.pop_front(onetile);
        if constexpr (fp32_sfpu_finalizer) {
            dfb_ex_welford_obj_main.pop_front(onetile);
        }
    }  // NCHt loop
    // The single eps tile is waited once and reused across all NCHt iterations; pop it at the end
    // so the buffer is left balanced.
    dfb_eps_obj.pop_front(onetile);
}
