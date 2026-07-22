// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// per_token_cast_back: out = cast(input_e4m3) * scale, where scale is one fp32
// scalar per token (row) per 128-wide block. A 128-element block spans
// 4 consecutive 32x32 tiles; within any tile the scale is a per-row scalar (constant across columns),
// so we broadcast it with mul_tiles_bcast_cols (BroadcastType::COL = filled column 0, per notes §6).
//
// The reader builds one bcast operand tile in cb_scale_bcast_fp32 with column 0 = scale[:, block_idx]
// (face-aware). There is no in-kernel column-shift LLK on Blackhole (notes §6), so the per-block
// column selection lives in the reader's data layout, not a shift here.
//
// CB naming: *_fp32 buffers always hold fp32, *_bf16 buffers always hold bf16; buffers whose format
// follows the datapath/output (cb_in_tile, cb_out) carry no suffix and are reused across both paths.
//
// Per block = tile_h rows x 128 cols = 4 tiles for default 32-wide tiles.
// fp32 datapath:
//   Phase 1: cast fp8 (e4m3) rm -> fp32 rm
//   Phase 2: tilize fp32 rm -> fp32 tiles
//   Phase 3: multiply by fp32 broadcast scale and untilize to output
// bf16 datapath:
//   Phase 1: tilize casts fp8 (e4m3) rm -> bf16 tiles
//   Phase 2: cast fp32 scale -> bf16 (packer) so SrcB matches the bf16 SrcA
//   Phase 3: multiply by bf16 broadcast scale and untilize to output

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_e4m3_id = get_compile_time_arg_val(0);
    CircularBuffer cb_input_e4m3(cb_input_e4m3_id);
    // fp32-only: copy-decoded e4m3 -> fp32 row-major block (used by the fp32 datapath only).
    constexpr uint32_t cb_in_rm_fp32_id = get_compile_time_arg_val(1);
    CircularBuffer cb_in_rm_fp32(cb_in_rm_fp32_id);
    // reused: tilized input feeding the multiply; format follows the datapath (fp32 or bf16).
    constexpr uint32_t cb_in_tile_id = get_compile_time_arg_val(2);
    CircularBuffer cb_in_tile(cb_in_tile_id);
    // fp32: raw scale (column 0) from the reader, always fp32.
    constexpr uint32_t cb_scale_bcast_fp32_id = get_compile_time_arg_val(3);
    CircularBuffer cb_scale_bcast_fp32(cb_scale_bcast_fp32_id);
    // reused: row-major output; format follows output_dtype (bf16 or fp32).
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(4);
    CircularBuffer cb_out(cb_out_id);
    // Tile dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(5);
    constexpr uint32_t tile_w = get_compile_time_arg_val(6);
    // 1 => run the datapath in bf16 (HiFi2): tilize decodes e4m3 straight into bf16 tiles and the fp32
    // scale is narrowed to bf16 (packer -> cb_scale_bcast_bf16) so the multiply's SrcB matches SrcA.
    constexpr uint32_t narrow_scales_to_bf16 = get_compile_time_arg_val(7);
    // bf16: scale narrowed from fp32 to bf16 on-device (used by the bf16 datapath only).
    constexpr uint32_t cb_scale_bcast_bf16_id = get_compile_time_arg_val(8);
    CircularBuffer cb_scale_bcast_bf16(cb_scale_bcast_bf16_id);
    // reused selector: the multiply reads whichever scale matches the datapath format.
    constexpr uint32_t cb_scale_mul_id = narrow_scales_to_bf16 ? cb_scale_bcast_bf16_id : cb_scale_bcast_fp32_id;
    constexpr uint32_t block_w = 128;                // BlockW
    constexpr uint32_t block_wt = block_w / tile_w;  // BlockWt
    constexpr uint32_t block_ht = 1;                 // BlockHt
    constexpr uint32_t tiles_per_block = block_ht * block_wt;

    constexpr uint32_t IDST0 = 0;

    uint32_t num_blocks = get_arg_val<uint32_t>(0);  // tile_h x 128 blocks for this core

    compute_kernel_hw_startup(cb_input_e4m3_id, cb_out_id);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        {
            if constexpr (narrow_scales_to_bf16) {
                // ----- Phase 1 (bf16): tilize casts e4m3 straight into bf16 tiles -----
                compute_kernel_lib::tilize<tiles_per_block, cb_input_e4m3_id, cb_in_tile_id>(block_ht);

                // ----- Phase 2 (bf16): cast fp32 scale -> bf16 -----
                // Packer narrows the fp32 scale to bf16 so the multiply's SrcB matches SrcA.
                reconfig_data_format_srca(cb_scale_bcast_fp32_id);
                pack_reconfig_data_format(cb_scale_bcast_bf16_id);
                copy_tile_init(cb_scale_bcast_fp32_id);
                cb_scale_bcast_fp32.wait_front(block_ht);
                cb_scale_bcast_bf16.reserve_back(block_ht);
                tile_regs_acquire();
                copy_tile(cb_scale_bcast_fp32_id, 0, IDST0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(IDST0, cb_scale_bcast_bf16_id);
                tile_regs_release();
                cb_scale_bcast_bf16.push_back(block_ht);
                cb_scale_bcast_fp32.pop_front(block_ht);
            } else {
                // ----- Phase 1: cast e4m3 -> fp32 -----
                // tilize cannot cast e4m3 into a fp32 rm as done for bf16 datapath
                reconfig_data_format_srca(cb_input_e4m3_id);
                pack_reconfig_data_format(cb_in_rm_fp32_id);
                copy_tile_init(cb_input_e4m3_id);
                // One input_e4m3 page is one 32x32 row-major tile. A 128-wide block contains
                // tiles_per_block such pages.
                for (uint32_t s = 0; s < tiles_per_block; ++s) {
                    cb_input_e4m3.wait_front(1);
                    cb_in_rm_fp32.reserve_back(1);
                    tile_regs_acquire();
                    copy_tile(cb_input_e4m3_id, 0, IDST0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(IDST0, cb_in_rm_fp32_id);
                    tile_regs_release();
                    cb_in_rm_fp32.push_back(1);
                    cb_input_e4m3.pop_front(1);
                }

                // ----- Phase 2: tilize -----
                compute_kernel_lib::tilize<tiles_per_block, cb_in_rm_fp32_id, cb_in_tile_id>(block_ht);
            }

            // ----- Phase 3: multiply by broadcast scale and untilize to output -----
            // mul writes all tiles_per_block(=4) tiles into DEST, and pack_untilize_dest untilizes them
            // to the row-major output in the packer, avoiding a separate output-tile L1 round-trip. In
            // 32-bit DEST (fp32_dest_acc), the half-sync pack-untilize block cap is 4 tiles = one block.
            reconfig_data_format(cb_in_tile_id, cb_scale_mul_id);
            mul_bcast_cols_init_short(cb_in_tile_id, cb_scale_mul_id);
            pack_untilize_dest_init<tiles_per_block, tiles_per_block>(cb_out_id);
            cb_in_tile.wait_front(tiles_per_block);
            if constexpr (narrow_scales_to_bf16) {
                cb_scale_bcast_bf16.wait_front(block_ht);
            } else {
                cb_scale_bcast_fp32.wait_front(block_ht);
            }
            cb_out.reserve_back(tiles_per_block);
            tile_regs_acquire();
            for (uint32_t k = 0; k < block_wt; ++k) {
                mul_tiles_bcast_cols(cb_in_tile_id, cb_scale_mul_id, k, 0, k);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<tiles_per_block>(cb_out_id);
            tile_regs_release();
            cb_out.push_back(tiles_per_block);
            cb_in_tile.pop_front(tiles_per_block);
            if constexpr (narrow_scales_to_bf16) {
                cb_scale_bcast_bf16.pop_front(block_ht);
            } else {
                cb_scale_bcast_fp32.pop_front(block_ht);
            }
            pack_untilize_uninit(cb_out_id);
        }
    }
}
