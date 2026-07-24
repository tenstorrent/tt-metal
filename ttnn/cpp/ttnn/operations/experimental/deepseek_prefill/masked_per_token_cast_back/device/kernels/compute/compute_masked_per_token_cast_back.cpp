// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// masked_per_token_cast_back: same math as per_token_cast_back (out = decode(input_e4m3) * scale, with
// scale broadcast over each 128-wide block's columns), but the per-core num_blocks loop count is
// computed by the reader (from the device-resident per-expert counts) and delivered here via the
// cb_control mailbox (read_tile_value distributes the UNPACK read to MATH/PACK so all three threads
// agree on the loop bound).
//
// Per block = tile_h rows x 128 cols = 4 tiles (default 32-wide tiles):
//   Phase 1: e4m3 -> compute-df tiles (bf16: tilize decodes directly; fp32: copy-decode then tilize)
//   Phase 2: broadcast-multiply by scale, untilize straight from DEST to the row-major output

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_input_e4m3_id = get_compile_time_arg_val(0);
    CircularBuffer cb_input_e4m3(cb_input_e4m3_id);
    constexpr uint32_t cb_in_rm_id = get_compile_time_arg_val(1);
    CircularBuffer cb_in_rm(cb_in_rm_id);
    constexpr uint32_t cb_in_tile_id = get_compile_time_arg_val(2);
    CircularBuffer cb_in_tile(cb_in_tile_id);
    constexpr uint32_t cb_scale_bcast_id = get_compile_time_arg_val(3);
    CircularBuffer cb_scale_bcast(cb_scale_bcast_id);
    constexpr uint32_t cb_out_fp32_id = get_compile_time_arg_val(4);
    CircularBuffer cb_out_fp32(cb_out_fp32_id);
    // Tile dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(5);
    constexpr uint32_t tile_w = get_compile_time_arg_val(6);
    constexpr uint32_t cb_control_id = get_compile_time_arg_val(7);
    CircularBuffer cb_control(cb_control_id);
    // 1 when the compute datapath is bf16 (compute_df != Float32). Only then can tilize decode e4m3
    // directly: llk_unpack_tilize's Float32-dest "lossless" mode mishandles an 8-bit source, so an fp32
    // compute datapath must decode e4m3 via a separate copy first.
    constexpr uint32_t compute_is_bf16 = get_compile_time_arg_val(8);
    // 1 => the fp32 scale must be narrowed to bf16 on-device: the packer converts it (into
    // cb_scale_bcast_bf16) before the bcast multiply. 0 => the scale is consumed directly from cb_scale_bcast.
    constexpr uint32_t convert_scale = get_compile_time_arg_val(9);
    constexpr uint32_t cb_scale_bcast_bf16_id = get_compile_time_arg_val(10);
    CircularBuffer cb_scale_bcast_bf16(cb_scale_bcast_bf16_id);
    constexpr uint32_t cb_scale_mul_id = convert_scale ? cb_scale_bcast_bf16_id : cb_scale_bcast_id;
    constexpr uint32_t block_w = 128;                // BlockW
    constexpr uint32_t block_wt = block_w / tile_w;  // BlockWt
    constexpr uint32_t block_ht = 1;                 // BlockHt
    constexpr uint32_t tiles_per_block = block_ht * block_wt;

    constexpr uint32_t IDST0 = 0;

    compute_kernel_hw_startup(cb_input_e4m3_id, cb_out_fp32_id);

    // num_blocks is computed by the reader and delivered via the control mailbox.
    cb_control.wait_front(1);
    uint32_t num_blocks = cb_control.read_tile_value(0, 0);
    cb_control.pop_front(1);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        {
            if constexpr (compute_is_bf16) {
                // Phase 1 (bf16): tilize decodes e4m3 and reshapes to tiles in one pass.
                // Only valid for a non-Float32 dest (Float32 "lossless" tilize mishandles an 8-bit source).
                reconfig_data_format_srca(cb_input_e4m3_id);
                pack_reconfig_data_format(cb_in_tile_id);
                tilize_init(cb_input_e4m3_id, tiles_per_block, cb_in_tile_id);
                cb_input_e4m3.wait_front(tiles_per_block);
                cb_in_tile.reserve_back(tiles_per_block);
                tilize_block(cb_input_e4m3_id, tiles_per_block, cb_in_tile_id);
                cb_in_tile.push_back(tiles_per_block);
                cb_input_e4m3.pop_front(tiles_per_block);
                tilize_uninit(cb_input_e4m3_id, cb_in_tile_id);
            } else {
                // Phase 1 (fp32): copy-decode e4m3 -> fp32 row-major, then tilize.
                // tilize cannot decode e4m3 into a Float32 dest, so decode first via copy_tile.
                reconfig_data_format_srca(cb_input_e4m3_id);
                pack_reconfig_data_format(cb_in_rm_id);
                copy_tile_init(cb_input_e4m3_id);
                for (uint32_t s = 0; s < tiles_per_block; ++s) {
                    cb_input_e4m3.wait_front(1);
                    cb_in_rm.reserve_back(1);
                    tile_regs_acquire();
                    copy_tile(cb_input_e4m3_id, 0, IDST0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(IDST0, cb_in_rm_id);
                    tile_regs_release();
                    cb_in_rm.push_back(1);
                    cb_input_e4m3.pop_front(1);
                }

                reconfig_data_format_srca(cb_in_rm_id);
                pack_reconfig_data_format(cb_in_tile_id);
                tilize_init(cb_in_rm_id, tiles_per_block, cb_in_tile_id);
                cb_in_rm.wait_front(tiles_per_block);
                cb_in_tile.reserve_back(tiles_per_block);
                tilize_block(cb_in_rm_id, tiles_per_block, cb_in_tile_id);
                cb_in_tile.push_back(tiles_per_block);
                cb_in_rm.pop_front(tiles_per_block);
                tilize_uninit(cb_in_rm_id, cb_in_tile_id);
            }

            // Scale convert (bf16): packer narrows the fp32 scale to bf16 so the multiply's SrcB matches SrcA.
            if constexpr (convert_scale) {
                reconfig_data_format_srca(cb_scale_bcast_id);
                pack_reconfig_data_format(cb_scale_bcast_bf16_id);
                copy_tile_init(cb_scale_bcast_id);
                cb_scale_bcast.wait_front(block_ht);
                cb_scale_bcast_bf16.reserve_back(block_ht);
                tile_regs_acquire();
                copy_tile(cb_scale_bcast_id, 0, IDST0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(IDST0, cb_scale_bcast_bf16_id);
                tile_regs_release();
                cb_scale_bcast_bf16.push_back(block_ht);
                cb_scale_bcast.pop_front(block_ht);
            }

            // Phase 2: broadcast-multiply by scale into DEST, then pack_untilize_dest straight to output
            // (no intermediate output-tile CB). fp32 DEST caps the half-sync pack-untilize at 4 tiles = one block.
            reconfig_data_format(cb_in_tile_id, cb_scale_mul_id);
            mul_bcast_cols_init_short(cb_in_tile_id, cb_scale_mul_id);
            pack_untilize_dest_init<tiles_per_block, tiles_per_block>(cb_out_fp32_id);
            cb_in_tile.wait_front(tiles_per_block);
            if constexpr (convert_scale) {
                cb_scale_bcast_bf16.wait_front(block_ht);
            } else {
                cb_scale_bcast.wait_front(block_ht);
            }
            cb_out_fp32.reserve_back(tiles_per_block);
            tile_regs_acquire();
            for (uint32_t k = 0; k < block_wt; ++k) {
                mul_tiles_bcast_cols(cb_in_tile_id, cb_scale_mul_id, k, 0, k);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<tiles_per_block>(cb_out_fp32_id);
            tile_regs_release();
            cb_out_fp32.push_back(tiles_per_block);
            cb_in_tile.pop_front(tiles_per_block);
            if constexpr (convert_scale) {
                cb_scale_bcast_bf16.pop_front(block_ht);
            } else {
                cb_scale_bcast.pop_front(block_ht);
            }
            pack_untilize_uninit(cb_out_fp32_id);
        }
    }
}
