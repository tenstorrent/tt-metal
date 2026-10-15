// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for the 1x32 / 1x64 / 1x128 scaled-fp8 post_combine_reduce (fused DQ2, sparse SDPA style), all on the FPU.
// Per 32-token block and stage s (the reader's compacted s-th local slot of every token):
//   1. tilize c_0 (32 rows of e4m3, part_wt tiles wide) -> bf16 tiles in c_6: tile row t = token t, so each
//      row of a tile lies in one scale block;
//   2. out (+)= c_6 * c_4 column-broadcast, c_4 holding the reader's weight * scale per row, one tile per scale
//      block; packed into the block's output tiles with L1 accumulation over the stages.
// The output is already tiled; no row-major scratch or final tilize.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

#ifndef TILES_PER_SCALE
#define TILES_PER_SCALE 1
#endif

constexpr uint32_t cb_rm_id = tt::CBIndex::c_0;
constexpr uint32_t cb_wscale_id = tt::CBIndex::c_4;
constexpr uint32_t cb_tiles_id = tt::CBIndex::c_6;
constexpr uint32_t cb_stage_count_id = tt::CBIndex::c_7;
constexpr uint32_t cb_output_id = tt::CBIndex::c_16;

constexpr uint32_t part_wt = get_compile_time_arg_val(0);
constexpr uint32_t num_experts = get_compile_time_arg_val(1);  // stages c_0 holds per block
constexpr uint32_t DST_TILES = 4;                              // 32-bit DEST, half sync
constexpr uint32_t part_scales = part_wt / TILES_PER_SCALE;
static_assert(part_wt % DST_TILES == 0, "part width must be a multiple of the DEST batch");
static_assert(part_wt % TILES_PER_SCALE == 0, "part width must be whole scale blocks");

void kernel_main() {
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);

    CircularBuffer cb_rm(cb_rm_id);
    CircularBuffer cb_wscale(cb_wscale_id);
    CircularBuffer cb_tiles(cb_tiles_id);
    CircularBuffer cb_stage_count(cb_stage_count_id);
    CircularBuffer cb_output(cb_output_id);

    compute_kernel_hw_startup(cb_rm_id, cb_tiles_id);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        cb_stage_count.wait_front(1);
        const uint32_t num_stages = cb_stage_count.read_tile_value(0, 0);
        cb_stage_count.pop_front(1);

        cb_output.reserve_back(part_wt);
        for (uint32_t s = 0; s < num_stages; ++s) {
            // 1. e4m3 rows -> bf16 tiles (unpack_tilize; Fp8 fast path).
            compute_kernel_lib::tilize<part_wt, cb_rm_id, cb_tiles_id>(1);

            // 2. out (+)= data * (weight * scale), accumulated in L1 over the stages.
            reconfig_data_format(cb_tiles_id, cb_wscale_id);
            pack_reconfig_data_format(cb_output_id);
            pack_reconfig_l1_acc(s > 0 ? 1 : 0);
            mul_bcast_cols_init(cb_tiles_id, cb_wscale_id);
            cb_tiles.wait_front(part_wt);
            cb_wscale.wait_front(part_scales);
            for (uint32_t j = 0; j < part_wt; j += DST_TILES) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < DST_TILES; ++k) {
                    mul_tiles_bcast_cols(cb_tiles_id, cb_wscale_id, j + k, (j + k) / TILES_PER_SCALE, k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < DST_TILES; ++k) {
                    pack_tile<true>(k, cb_output_id, j + k);
                }
                tile_regs_release();
            }
            pack_reconfig_l1_acc(0);
            cb_tiles.pop_front(part_wt);
            cb_wscale.pop_front(part_scales);
        }
        // The reader hands over c_0 whole, unused stages included.
        if (num_stages < num_experts) {
            cb_rm.wait_front((num_experts - num_stages) * part_wt);
            cb_rm.pop_front((num_experts - num_stages) * part_wt);
        }
        cb_output.push_back(part_wt);
    }
}
