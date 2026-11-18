// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Copied-and-trimmed matmul_block K walk (TileRowMajor + packer L1 accumulation only) with the
// pack-side state hoisted out of the subblock loop. Bench-only; see test_l1acc_pack_hoist.py.
//
// RAW LLK / HELPER BYPASS (why this exists):
//   compute_kernel_lib::matmul_block re-issues, before EVERY output subblock's pack,
//     pack_reconfig_data_format(target)      -> reconfig_packer_data_format(): STALLWAIT(CFG, PACK|THCON)
//                                               + ~15 cfg RMW/WRCFG (formats, strides, exp threshold, ...)
//     llk_pack_reconfig_l1_acc(block ? 1 : 0) -> STALLWAIT(CFG, PACK) + 2 cfg RMWs
//   although both are constant within a K-block. This copy issues them ONCE per K-block (after the
//   PreKBlockFn, which may have dirtied them), which the helper cannot express: its pack-state calls sit
//   inside the subblock loop with no hook to lift them.
//   PackPath::RowMop additionally packs each subblock ROW (w tiles, contiguous in both DEST and the
//   TileRowMajor CB) with ONE packer MOP programmed for w tiles (llk_pack_init(cb, num_tiles = w)) +
//   one llk_pack_impl call, instead of w pack_tile<true> calls (each its own MOP ending in a Last=1
//   tile-close + its own L1_Dest_addr program). There is no compute-API entry for a multi-tile pack MOP
//   on Blackhole (pack_block loops pack_tile), so this is raw LLK.
// Numerics: identical DEST data, formats, L1-acc enable per K-block -> bit-identical outputs expected
// (the bench asserts it).

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

namespace kwalk {

namespace ckl = compute_kernel_lib;

enum class PackPath : uint32_t { PerTile = 0, RowMop = 1 };

template <
    bool caller_owns_pack_target,
    ckl::LastBlockTarget last_block_target,
    PackPath pack_path,
    bool force_no_l1_acc,  // DIAGNOSTIC ONLY (wrong numerics): every K-block overwrites.
    bool keep_fast_drain,  // re-enable fast tile-end drain after the l1-acc reconfig (exact-numerics knob)
    typename PreKBlockFn,
    typename Buf>
ALWI void matmul_block_l1acc_hoisted(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    const ckl::MatmulBlockShape& shape,
    PreKBlockFn pre_k_block,
    uint32_t in1_per_core_w,
    uint32_t out_row_width) {
    constexpr bool pack_last_to_interm = (last_block_target == ckl::LastBlockTarget::Interm);
    constexpr bool pack_relu = (last_block_target == ckl::LastBlockTarget::OutWithRelu);
    static_assert(last_block_target != ckl::LastBlockTarget::OutWithUntilize, "untilize target needs SubblockMajor");
    static_assert(!caller_owns_pack_target || pack_last_to_interm, "caller_owns requires Interm");

    const uint32_t in0_cb_id = ckl::buf_id(in0_buf);
    const uint32_t in1_cb_id = ckl::buf_id(in1_buf);
    const uint32_t out_cb_id = ckl::buf_id(out_buf);
    const uint32_t interm_cb_id = ckl::buf_id(interm_buf);

    const uint32_t out_num_tiles = shape.out_subblock_h * shape.out_subblock_w;
    const uint32_t in0_subblock_num_tiles = shape.out_subblock_h * shape.in0_block_k;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks;
    const uint32_t in1_block_num_tiles = in1_per_core_w * shape.in0_block_k;
    const uint32_t out_block_num_tiles = out_num_tiles * shape.in0_num_subblocks * shape.in1_num_subblocks;
    const uint32_t row_group_tiles = shape.out_subblock_h * out_row_width;

    bool enable_reload = false;
    uint32_t blk_pack_id = interm_cb_id;

    for (uint32_t block = 0; block < shape.num_k_blocks; block++) {
        const bool last_out = block == (shape.num_k_blocks - 1);

        if constexpr (pack_relu) {
            if (last_out) {
                PACK((llk_pack_relu_config(ReluConfig::zero())));
            }
        }

        pre_k_block(block, shape.num_k_blocks, last_out);

        // ShortAfterPreKBlock restore (unpack/math side identical to the helper).
        reconfig_data_format(in1_cb_id, in0_cb_id);
        ckernel::matmul_block_init(
            in0_cb_id, in1_cb_id, false, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);

        // HOISTED pack state: everything the helper re-issues per subblock, issued once per K-block.
        const bool to_out = last_out && !pack_last_to_interm;
        blk_pack_id = to_out ? out_cb_id : interm_cb_id;
        uint32_t blk_l1_acc = to_out ? 0 : (block == 0 ? 0 : 1);
        if constexpr (force_no_l1_acc) {
            blk_l1_acc = 0;
        }
        PACK((pack_reconfig_data_format(blk_pack_id)));
        PACK((llk_pack_reconfig_l1_acc(blk_l1_acc)));
        if constexpr (keep_fast_drain) {
#if defined(TRISC_PACK) && !defined(TT_METAL_TTSIM)
            ckernel::cfg_reg_rmw_tensix<PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain_RMW>(0);
#endif
        }
        if constexpr (pack_path == PackPath::RowMop) {
            // One MOP = w contiguous tiles (DEST slots r*w..r*w+w-1 -> CB tiles row_base+col_base..+w-1).
            PACK((llk_pack_init<PackMode::Default, false, true /*skip_addrmod*/, true /*skip_strides*/>(
                blk_pack_id, shape.out_subblock_w)));
        }

        in0_buf.wait_front(in0_block_num_tiles);
        in1_buf.wait_front(in1_block_num_tiles);

        Buf& pack_target_buf = pack_last_to_interm ? interm_buf : out_buf;

        int in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < shape.in0_num_subblocks; in0_subblock++) {
            if constexpr (!caller_owns_pack_target) {
                if (last_out) {
                    pack_target_buf.reserve_back(row_group_tiles);
                } else {
                    interm_buf.reserve_back(row_group_tiles);
                }
            }

            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < shape.in1_num_subblocks; in1_subblock++) {
                tile_regs_acquire();

                if (enable_reload) {
                    reconfig_data_format_srca(in1_cb_id, interm_cb_id);
                    copy_init(interm_cb_id);
                    if (in1_subblock == 0) {
                        interm_buf.wait_front(row_group_tiles);
                    }
                    ckl::copy_subblock_row_strided(
                        interm_cb_id,
                        in1_subblock * shape.out_subblock_w,
                        out_row_width,
                        shape.out_subblock_h,
                        shape.out_subblock_w);
                    if (in1_subblock == shape.in1_num_subblocks - 1) {
                        interm_buf.pop_front(row_group_tiles);
                    }
                    reconfig_data_format_srca(interm_cb_id, in1_cb_id);
                    ckernel::matmul_block_init(
                        in0_cb_id, in1_cb_id, false, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
                }

                uint32_t dst_index = 0;
                uint32_t in0_index = in0_index_subblock_offset;
                uint32_t in1_index = in1_index_subblock_offset;
                for (uint32_t inner_dim = 0; inner_dim < shape.in0_block_k; inner_dim++) {
#ifndef SKIP_COMPUTE
                    ckernel::matmul_block(
                        in0_cb_id,
                        in1_cb_id,
                        in0_index,
                        in1_index,
                        dst_index,
                        false,
                        shape.out_subblock_w,
                        shape.out_subblock_h,
                        shape.in0_block_k);
#else
                    (void)dst_index;
#endif
                    in0_index++;
                    in1_index += in1_per_core_w;
                }

                tile_regs_commit();
                tile_regs_wait();

                const uint32_t row_base = caller_owns_pack_target ? in0_subblock * row_group_tiles : 0;
                const uint32_t col_base = row_base + in1_subblock * shape.out_subblock_w;
                if constexpr (pack_path == PackPath::RowMop) {
#ifdef TRISC_PACK
                    const std::uint8_t output_id = get_output_id(blk_pack_id);
                    for (uint32_t r = 0; r < shape.out_subblock_h; r++) {
                        const uint32_t addr =
                            get_output_tile_address<true, PackMode::Default>(output_id, r * out_row_width + col_base);
                        llk_pack_impl<DST_ACCUM_MODE, PackMode::Default>(r * shape.out_subblock_w, addr);
                    }
#endif
                } else {
                    ckl::pack_subblock_row_strided(
                        0, blk_pack_id, col_base, out_row_width, shape.out_subblock_h, shape.out_subblock_w);
                }

                tile_regs_release();
                in1_index_subblock_offset += shape.out_subblock_w;
            }

            if constexpr (!caller_owns_pack_target) {
                if (last_out) {
                    pack_target_buf.push_back(row_group_tiles);
                } else {
                    interm_buf.push_back(row_group_tiles);
                }
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        // Drain / reload bookkeeping: identical to the helper (spill_row_grouped == true).
        const uint32_t drain_step = row_group_tiles;
        if constexpr (pack_last_to_interm) {
            if constexpr (!caller_owns_pack_target) {
                if (block < shape.num_k_blocks - 1) {
                    for (uint32_t s = 0; s < out_block_num_tiles; s += drain_step) {
                        interm_buf.wait_front(drain_step);
                        interm_buf.pop_front(drain_step);
                    }
                }
            }
            enable_reload = false;
        } else {
            if (shape.num_k_blocks >= 2 && block < shape.num_k_blocks - 2) {
                for (uint32_t s = 0; s < out_block_num_tiles; s += drain_step) {
                    interm_buf.wait_front(drain_step);
                    interm_buf.pop_front(drain_step);
                }
            }
            if (block == shape.num_k_blocks - 2) {
                enable_reload = true;
            }
        }

        in0_buf.pop_front(in0_block_num_tiles);
        in1_buf.pop_front(in1_block_num_tiles);
    }

    if constexpr (pack_path == PackPath::RowMop) {
        // Leave the packer as the helper does: the 1-tile default MOP, so later pack_tile calls are unaffected.
        PACK((llk_pack_init<PackMode::Default, false, true, true>(blk_pack_id, 1)));
    }
}

}  // namespace kwalk
