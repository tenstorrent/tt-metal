// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Index score of every query row for every key this core owns:
//     scores[j][t] = sum_h ReLU(q_jh . k_t) * w_jh
//
// The query is first tilized into 8x32 tiles, eight heads of one query row per tile. Keys are then
// processed one 32-token tile row at a time, and each chunk is scored against every query row while
// it is resident: for each 8-head group custom_mm forms the 8x32 block q_g . K_chunk^T in Dst, and
// the SFPU adds its weighted ReLU into partial sums held in Dst. After the row's last group the
// partials are reduced to the 1x32 score row, which is packed once per chunk and query row, in
// (chunk, query row) order.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "lightning_score_sfpu.h"

void kernel_main() {
    // ---- Compile-time args ----
    [[maybe_unused]] constexpr uint32_t k = get_arg(args::k);
    constexpr uint32_t num_queries = get_arg(args::num_queries);            // Sq
    constexpr uint32_t num_heads = get_arg(args::num_heads);                // Hi
    constexpr uint32_t d_tiles = get_arg(args::d_tiles);                    // D / 32
    constexpr uint32_t num_weight_tiles = get_arg(args::num_weight_tiles);  // Sq * Hi / 32
    constexpr uint32_t chunks_per_block = get_arg(args::chunks_per_block);  // block_size / 32
    constexpr uint32_t key_tiles_per_block = chunks_per_block * d_tiles;
    constexpr uint32_t num_head_groups = num_heads / 8;
    constexpr uint32_t num_query_group_tiles = num_queries * num_head_groups * d_tiles;
    static_assert(num_heads % 8 == 0, "heads are scored 8 per query tile");
    static_assert(d_tiles >= 2 && d_tiles <= 256 && d_tiles % 2 == 0, "custom_mm needs an even kt_dim in [2, 256]");

    constexpr bool kTransposeKey = true;
    constexpr bool kSplitAcc = true;

    // ---- Dataflow buffers ----
    DataflowBuffer query_rm_dfb(dfb::query_rm);        // consumer <- reader, [Sq * Hi, D] as 8-row row-major strips
    DataflowBuffer query_tiled_dfb(dfb::query_tiled);  // self-loop, [Sq * Hi, D] as 8x32 tiles
    DataflowBuffer key_dfb(dfb::key);                  // consumer <- reader, one block of 32x32 tiles
    DataflowBuffer weights_dfb(dfb::weights);          // consumer <- reader, [Sq, Hi] as 1x32 tiles
    DataflowBuffer ctrl_dfb(dfb::ctrl);                // consumer <- reader, word 0 = blocks on this core
    DataflowBuffer scores_dfb(dfb::scores);            // producer -> writer, one fp32 1x32 tile per 32 keys and row
    DataflowBuffer indices_dfb(dfb::indices);          // producer -> reader

    // Configure for tilize first: tilize_init does not reprogram the packer output format, so the
    // packer must already target query_tiled's bf16 rather than the fp32 scores.
    compute_kernel_hw_startup(query_rm_dfb.get_id(), query_tiled_dfb.get_id());

    // Tilize the row-major query into 8x32 tiles, one strip of 8 heads at a time: tile
    // (j * num_head_groups + g) * d_tiles + i holds query row j's heads [8g, 8g + 8) x D columns
    // [32i, 32i + 32).
    tilize_init(query_rm_dfb.get_id(), d_tiles, query_tiled_dfb.get_id());
    for (uint32_t g = 0; g < num_queries * num_head_groups; ++g) {
        query_rm_dfb.wait_front(d_tiles);
        query_tiled_dfb.reserve_back(d_tiles);
        tilize_block(query_rm_dfb.get_id(), d_tiles, query_tiled_dfb.get_id());
        query_tiled_dfb.push_back(d_tiles);
        query_rm_dfb.pop_front(d_tiles);
    }
    tilize_uninit(query_rm_dfb.get_id(), query_tiled_dfb.get_id());
    query_tiled_dfb.wait_front(num_query_group_tiles);
    // Tilize reprogrammed unpack, math and pack; restore the full custom_mm configuration. The SFPU
    // passes below need no init of their own (they only use ADDR_MOD_7, which custom_mm leaves as a
    // no-increment mode), so this is the only init in the kernel after tilize.
    custom_mm_block_init<kTransposeKey, kSplitAcc>(query_tiled_dfb.get_id(), key_dfb.get_id(), scores_dfb.get_id());

    // Head weights stay in their bf16 shard: query row j's head h is element j * Hi + h counted from
    // the first tile (the 1x32 tiles are contiguous), read by the SFPU pass as it needs it.
    weights_dfb.wait_front(num_weight_tiles);
    [[maybe_unused]] const volatile uint16_t* weights =
        reinterpret_cast<const volatile uint16_t*>(weights_dfb.get_tile_address(0));
    // The math thread reads the weights straight from L1; its data cache can still hold an earlier
    // program's lines at these addresses.
    invalidate_l1_cache();

    ctrl_dfb.wait_front(1);
    const uint32_t num_blocks = ctrl_dfb.read_tile_value(0, 0);
    ctrl_dfb.pop_front(1);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        key_dfb.wait_front(key_tiles_per_block);
        for (uint32_t chunk = 0; chunk < chunks_per_block; ++chunk) {
            for (uint32_t row = 0; row < num_queries; ++row) {
                const uint32_t first_group_tile = row * num_head_groups * d_tiles;
                [[maybe_unused]] const volatile uint16_t* row_weights = weights + row * num_heads;
                scores_dfb.reserve_back(1);
                tile_regs_acquire();
                for (uint32_t g = 0; g < num_head_groups; ++g) {
                    // One full D reduction: 8 heads' d_tiles 8x32 tiles against the chunk's d_tiles key
                    // tiles, each key tile transposed so the output columns are indexed by token.
                    // clear_src must stay off: on some cores the SrcB clear corrupts the scores of
                    // tokens 0-3 of the chunk (same cores every run, never with it off).
                    custom_mm_block</*finalize=*/true, /*read_transposed=*/false, /*clear_src=*/false>(
                        query_tiled_dfb.get_id(),
                        key_dfb.get_id(),
                        first_group_tile + g * d_tiles,
                        chunk * d_tiles,
                        ckernel::sfpu::kLightningScoreTile,
                        d_tiles);
                    if (g == 0) {
                        MATH((ckernel::sfpu::lightning_weighted_relu_accumulate<true>(row_weights)));
                    } else {
                        MATH((ckernel::sfpu::lightning_weighted_relu_accumulate<false>(row_weights + g * 8)));
                    }
                }
                MATH((ckernel::sfpu::lightning_reduce_partials_to_row0()));
                tile_regs_commit();

                tile_regs_wait();
                pack_tile<true>(ckernel::sfpu::kLightningAccTile, scores_dfb.get_id(), 0);
                tile_regs_release();
                scores_dfb.push_back(1);
            }
        }
        key_dfb.pop_front(key_tiles_per_block);
    }
    custom_mm_block_uninit();

    indices_dfb.reserve_back(1);
    indices_dfb.push_back(1);

    query_tiled_dfb.pop_front(num_query_group_tiles);
    weights_dfb.pop_front(num_weight_tiles);
}
