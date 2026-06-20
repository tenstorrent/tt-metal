// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"

namespace compute_kernel_lib {

template <
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    reblock_untilize_config::InitUninitMode init_uninit_mode,
    reblock_untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode,
    OutputCBLayout layout,
    typename Buf>
inline void reblock_and_untilize(
    uint32_t in0_num_subblocks,
    uint32_t num_subblocks_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    Buf& interm_buf,
    Buf& out_buf) {
    static_assert(
        layout == OutputCBLayout::SubblockMajor,
        "reblock_and_untilize requires SubblockMajor interm input. The tile addressing "
        "below assumes tiles are grouped per-subblock; TileRowMajor input is already in "
        "tile-row order, so callers should use the standard untilize helper instead.");

    const uint32_t interm_cb_id = buf_id(interm_buf);
    const uint32_t out_cb_id = buf_id(out_buf);

    // Data-format reconfig — gated on reconfig_mode, INDEPENDENT of init_uninit_mode
    // (mirrors untilize). srcA reads interm via copy_tile; pack targets out. Done once
    // before the in0_subblock loop. Without this the helper would untilize using
    // whatever formats the previous op left — correct only when they coincide with
    // interm/out.
    if constexpr (
        reconfig_mode == reblock_untilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure ||
        reconfig_mode == reblock_untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure) {
        reconfig_data_format_srca(interm_cb_id);
    }
    if constexpr (
        reconfig_mode == reblock_untilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure ||
        reconfig_mode == reblock_untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure) {
        pack_reconfig_data_format(out_cb_id);
    }

    // Init once before the loop (gated).
    if constexpr (
        init_uninit_mode == reblock_untilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == reblock_untilize_config::InitUninitMode::InitOnly) {
        pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb_id);
        copy_tile_to_dst_init_short(interm_cb_id);
    }

    const uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_subblocks_w);

    // Internalized in0_subblock loop: one call untilizes the whole output block
    // (mirroring untilize(num_blocks)) — a single InitAndUninit call covers reconfig +
    // init + every row-group + uninit.
    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
        interm_buf.wait_front(num_tiles_in_row_of_subblocks);

        uint32_t within_block_index = 0;
        for (uint32_t h = 0; h < out_subblock_h; h++) {
            uint32_t block_offset = 0;
            out_buf.reserve_back(out_block_w);
            for (uint32_t n = 0; n < num_subblocks_w; n++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < out_subblock_w; w++) {
                    copy_tile(interm_cb_id, block_offset + within_block_index + w, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<out_subblock_w, out_block_w>(out_cb_id, 1, n);
                tile_regs_release();
                block_offset += out_subblock_num_tiles;
            }
            out_buf.push_back(out_block_w);
            within_block_index += out_subblock_w;
        }
        interm_buf.pop_front(num_tiles_in_row_of_subblocks);
    }

    if constexpr (
        init_uninit_mode == reblock_untilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == reblock_untilize_config::InitUninitMode::UninitOnly) {
        pack_untilize_uninit(interm_cb_id);
    }
}

}  // namespace compute_kernel_lib
