// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for the 1x32 scaled-fp8 post_combine_reduce (fused DQ2, sparse SDPA style), all on the FPU.
// Per 32-token block and stage s (the reader's compacted s-th local slot of every token):
//   1. tilize c_0 (32 rows of e4m3, part_wt tiles wide) -> bf16 tiles in c_6: tile row t = token t, so each
//      tile row is one 1x32 scale block;
//   2. c_5 = c_4 (scale, column 0) * c_1 (routing weight, column 0), column-broadcast: per-row weight * scale;
//   3. out (+)= c_6 * c_5 column-broadcast, packed into the block's output tiles with L1 accumulation over
//      the stages.
// The output is already tiled; no row-major scratch or final tilize.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

constexpr uint32_t cb_rm_id = tt::CBIndex::c_0;
constexpr uint32_t cb_weight_col_id = tt::CBIndex::c_1;
constexpr uint32_t cb_scale_id = tt::CBIndex::c_4;
constexpr uint32_t cb_wscale_id = tt::CBIndex::c_5;
constexpr uint32_t cb_tiles_id = tt::CBIndex::c_6;
constexpr uint32_t cb_stage_count_id = tt::CBIndex::c_7;
constexpr uint32_t cb_output_id = tt::CBIndex::c_16;

constexpr uint32_t part_wt = get_compile_time_arg_val(0);
constexpr uint32_t DST_TILES = 4;  // 32-bit DEST, half sync
static_assert(part_wt % DST_TILES == 0, "part width must be a multiple of the DEST batch");

void kernel_main() {
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);

    CircularBuffer cb_weight_col(cb_weight_col_id);
    CircularBuffer cb_scale(cb_scale_id);
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

            // 2. weight * scale per row (column 0 is all the bcast operand reads).
            reconfig_data_format(cb_scale_id, cb_weight_col_id);
            pack_reconfig_data_format(cb_wscale_id);
            mul_bcast_cols_init(cb_scale_id, cb_weight_col_id);
            cb_scale.wait_front(part_wt);
            cb_weight_col.wait_front(1);
            cb_wscale.reserve_back(part_wt);
            for (uint32_t j = 0; j < part_wt; j += DST_TILES) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < DST_TILES; ++k) {
                    mul_tiles_bcast_cols(cb_scale_id, cb_weight_col_id, j + k, 0, k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < DST_TILES; ++k) {
                    pack_tile(k, cb_wscale_id);
                }
                tile_regs_release();
            }
            cb_wscale.push_back(part_wt);
            cb_scale.pop_front(part_wt);
            cb_weight_col.pop_front(1);

            // 3. out (+)= data * (weight * scale), accumulated in L1 over the stages.
            reconfig_data_format(cb_tiles_id, cb_wscale_id);
            pack_reconfig_data_format(cb_output_id);
            pack_reconfig_l1_acc(s > 0 ? 1 : 0);
            mul_bcast_cols_init(cb_tiles_id, cb_wscale_id);
            cb_tiles.wait_front(part_wt);
            cb_wscale.wait_front(part_wt);
            for (uint32_t j = 0; j < part_wt; j += DST_TILES) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < DST_TILES; ++k) {
                    mul_tiles_bcast_cols(cb_tiles_id, cb_wscale_id, j + k, j + k, k);
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
            cb_wscale.pop_front(part_wt);
        }
        cb_output.push_back(part_wt);
    }
}
