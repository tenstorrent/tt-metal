// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bias_seed: raw K-loop for the op's one matmul_block configuration (TileRowMajor + packer L1-acc + Interm +
// caller-owned pack target, Short init, INPUT_AND_OUTPUT reconfig, WaitAndPopPerKBlock on both inputs). It
// emits the same LLK sequence as compute_kernel_lib::matmul_block for that configuration, plus the two
// capabilities the helper lacks:
//   - seeded_l1_acc: K-block 0 packs with L1-acc ON, accumulating onto values already in the pack target
//     (the helper hard-codes llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)).
//   - dest_seed: on K-block 0, after tile_regs_acquire and before the first matmul step, each subblock's DEST
//     is loaded with the row-broadcast bias (unary_bcast ROW); the matmul accumulates onto it. The helper has
//     no hook between acquire and the first matmul_block LLK call.

#pragma once

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

namespace bias_seed {

template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t partials_cb,
    uint32_t seed_cb,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t sb_h,
    uint32_t sb_w,
    uint32_t k_block,
    uint32_t num_k_blocks,
    uint32_t k_tiles,
    bool seeded_l1_acc,
    bool dest_seed,
    // dest_seed: the K-block whose DEST is pre-loaded, and the tile index in seed_cb of bias column 0
    // (seed_cb = cb_bias at block 0, or in1_cb itself at the last block, where the bias tile row rides in
    // the slot's spare K row).
    uint32_t seed_block = 0,
    uint32_t seed_tile_base = 0>
ALWI void matmul_k_loop() {
    constexpr uint32_t n_tiles = sb_w * in1_num_subblocks;
    constexpr uint32_t in0_block_tiles = sb_h * k_block * in0_num_subblocks;
    constexpr uint32_t in1_block_tiles = n_tiles * k_block;
    constexpr uint32_t row_group_tiles = sb_h * n_tiles;

    reconfig_data_format(in1_cb, in0_cb);
    PACK((pack_reconfig_data_format(partials_cb)));
    ckernel::matmul_block_init(in0_cb, in1_cb, false, sb_w, sb_h, k_block);

    for (uint32_t block = 0; block < num_k_blocks; ++block) {
        cb_wait_front(in0_cb, in0_block_tiles);
        cb_wait_front(in1_cb, in1_block_tiles);
        const bool seed_now = dest_seed && block == seed_block;
        // Ragged last K-block: only the K tiles that exist (the slot's trailing columns / rows are stale).
        const uint32_t inner_steps = (block + 1) * k_block > k_tiles ? k_tiles - block * k_block : k_block;
        for (uint32_t in0_sb = 0; in0_sb < in0_num_subblocks; ++in0_sb) {
            for (uint32_t in1_sb = 0; in1_sb < in1_num_subblocks; ++in1_sb) {
                tile_regs_acquire();
                if constexpr (dest_seed) {
                    if (seed_now) {
                        // seed_cb shares in1's bf16 format, so the unpacker source format is already right.
                        // ROW broadcast reads row 0 only: the bias tile's padding rows need no zeroing.
                        unary_bcast_init<BroadcastType::ROW>(seed_cb);
                        for (uint32_t r = 0; r < sb_h; ++r) {
                            for (uint32_t c = 0; c < sb_w; ++c) {
                                unary_bcast<BroadcastType::ROW>(
                                    seed_cb, seed_tile_base + in1_sb * sb_w + c, r * sb_w + c);
                            }
                        }
                        ckernel::matmul_block_init(in0_cb, in1_cb, false, sb_w, sb_h, k_block);
                    }
                }
                uint32_t in0_index = in0_sb * sb_h * k_block;
                uint32_t in1_index = in1_sb * sb_w;
                for (uint32_t inner = 0; inner < inner_steps; ++inner) {
                    ckernel::matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, false, sb_w, sb_h, k_block);
                    ++in0_index;
                    in1_index += n_tiles;
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((pack_reconfig_data_format(partials_cb)));
                PACK((llk_pack_reconfig_l1_acc((block == 0 && !seeded_l1_acc) ? 0 : 1)));
                const uint32_t base = in0_sb * row_group_tiles + in1_sb * sb_w;
                for (uint32_t r = 0; r < sb_h; ++r) {
                    for (uint32_t c = 0; c < sb_w; ++c) {
                        pack_tile<true>(r * sb_w + c, partials_cb, base + r * n_tiles + c);
                    }
                }
                tile_regs_release();
            }
        }
        cb_pop_front(in0_cb, in0_block_tiles);
        cb_pop_front(in1_cb, in1_block_tiles);
    }
}

// partials_seed: write the row-broadcast bias (fp32) into every tile of the reserved partials block, so the
// seeded K-loop accumulates onto it. One unary_bcast per bias tile; each DEST tile is packed to all rows.
template <uint32_t partials_cb, uint32_t bias_cb, uint32_t rows, uint32_t n_tiles, uint32_t dest_tiles>
ALWI void seed_partials() {
    static_assert(n_tiles % dest_tiles == 0);
    reconfig_data_format_srca(bias_cb);
    unary_bcast_init<BroadcastType::ROW>(bias_cb);
    PACK((pack_reconfig_data_format(partials_cb)));
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t n0 = 0; n0 < n_tiles; n0 += dest_tiles) {
        tile_regs_acquire();
        for (uint32_t c = 0; c < dest_tiles; ++c) {
            unary_bcast<BroadcastType::ROW>(bias_cb, n0 + c, c);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < dest_tiles; ++c) {
                pack_tile<true>(c, partials_cb, r * n_tiles + n0 + c);
            }
        }
        tile_regs_release();
    }
}

}  // namespace bias_seed
