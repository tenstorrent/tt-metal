// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

// Per-core gate_up matmul + SwiGLU gate + down matmul (runs on every compute core).
//
// The B (<= 32) tokens of the batch are the rows of a single tile row, so every matmul below already
// computes all B tokens at once and nothing here loops over tokens: an expert's weights are unpacked
// once and serve every token that selected it. The tokens are separated only at the weighted
// accumulation, by the per-token routing-weight tile (phase 2).
//
// The selected experts are processed in BLOCKS of `experts_block`, and both phases below run once per
// block: only one block's activations are resident in cb_act, which is what allows far more experts
// than L1 could hold at once. The accumulator spans all blocks, so the weighted sum is unaffected --
// only the first expert of the first block starts it and only the last expert of the last block
// finishes it.
//
// PHASE 1 (SwiGLU cores only -- those owning a slice of the I dim):
//   Each SwiGLU core owns a 2-tile (64 column) slice of the SwiGLU output I dimension. Its
//   gate_up weight shard ([K, 128] == [k_tiles, 4] tiles) holds the gate columns (tile cols
//   0,1) and paired up columns (tile cols 2,3) for that slice. For every selected expert:
//       gate = x @ gate_w   (cb_weights tile cols 0,1) -> [32, 64]
//       up   = x @ up_w     (cb_weights tile cols 2,3) -> [32, 64]
//       cb_out = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit) -> [32, 64]
//   where x (cb_input) is resident as Kt == H/32 activation tiles, each holding all B tokens, and is
//   reused by every block. cb_out (this core's 2-tile slice of act[B, I]) is scattered by the writer
//   to core {0,0}, gathered into the block's activations, and broadcast back into every core's cb_act.
//
// PHASE 2 (all cores): the down matmul, scaled by each token's routing weight for the expert and
//   accumulated into a single output tile row. cb_act holds the block's activations, act[B, I] per
//   expert (i_tiles tiles each, K order). For each expert, each core multiplies its down weight shard
//   ([I, H/64] == [i_tiles, 2] tiles) to produce its 2-tile (64 column) slice of down_e[B, H], scales
//   it by the per-token routing-weight tile (cb_rscalar, row b = token b's weight for this expert),
//   and accumulates:
//       down_e = act @ down_w   (cb_down_w tile (k, n) at k*2 + n) -> [32, 64]
//       out   += routing_w[:, e] * down_e   (summed over all active experts)
//   The running sum ping-pongs through cb_acc; the final expert writes cb_down_out, which the
//   writer drains once into the [1, B, H] DRAM output tile row.
//
// Compile-time args:
//   0: num_active   (routing-selected experts to run)
//   1: k_tiles      (H / 32, gate_up contraction)
//   2: i_tiles      (I / 32; SwiGLU output cols AND down contraction (act K-tiles))
//   3: cb_input     (activation tiles)
//   4: cb_weights   (this core's per-expert [K, 128] gate+up slice)
//   5: cb_mm        (gate_up matmul staging: 4 tiles = gate 0,1 | up 2,3; reused for down)
//   6: cb_out       (this core's 2 SwiGLU output tiles per expert)
//   7: limit_bits   (SwiGLU clamp limit as a float bit pattern)
//   8: cb_act       (one block of gathered activations, act[B, I] per expert)
//   9: cb_down_w    (this core's per-expert [I, 64] down slice = i_tiles*2 tiles)
//  10: cb_down_out  (this core's 2 accumulated output tiles, written once)
//  11: cb_rscalar   (routing-weight tiles for the current block; row b = token b's weight)
//  12: cb_acc       (running weighted-sum accumulator, ping-ponged across experts)
//  13: cb_wtmp      (staging for one expert's weighted down output before the accumulate)
//  14: num_producers (number of SwiGLU cores; each owns i_tiles/num_producers of the I dim)
//  15: experts_block (experts per block; the activation block held in cb_act at once)
//  16: gate_up_reserve_tiles (pages a gate_up slice occupies in cb_weights, >= the slice itself)
//  17: down_reserve_tiles    (pages a down slice occupies in cb_weights, >= the slice itself)
//
// Runtime args:
//   0: core_index (this core's flat grid index, x*8 + y)
void kernel_main() {
    constexpr uint32_t num_active = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_mm_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(6);
    constexpr uint32_t limit_bits = get_compile_time_arg_val(7);
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_down_w_id = get_compile_time_arg_val(9);
    constexpr uint32_t cb_down_out_id = get_compile_time_arg_val(10);
    constexpr uint32_t cb_rscalar_id = get_compile_time_arg_val(11);
    constexpr uint32_t cb_acc_id = get_compile_time_arg_val(12);
    constexpr uint32_t cb_wtmp_id = get_compile_time_arg_val(13);
    constexpr uint32_t num_producers = get_compile_time_arg_val(14);
    constexpr uint32_t experts_block = get_compile_time_arg_val(15);
    constexpr uint32_t gate_up_reserve_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t down_reserve_tiles = get_compile_time_arg_val(17);

    const uint32_t core_index = get_arg_val<uint32_t>(0);
    const bool swiglu_core = core_index < num_producers;

    constexpr uint32_t kOutTilesPerCore = 2;
    // This core's share of the SwiGLU I dim, and the gate|up tile-column width of its gate_up
    // shard (the first swiglu_tiles cols are gate, the next swiglu_tiles are the paired up).
    constexpr uint32_t swiglu_tiles = i_tiles / num_producers;
    constexpr uint32_t kShardTileCols = 2 * swiglu_tiles;
    // The gate_up ([K, 128]) and down ([I, 64] == [i_tiles, 2]) weight slots in cb_weights are waited
    // on and popped by their *_reserve_tiles page counts, which the host may have padded past the
    // slice itself so both phases advance the shared CB by the same stride (see the program factory).
    // The matmuls index tiles from the front of a slot, so any pad is simply never read.
    constexpr uint32_t kNumBlocks = (num_active + experts_block - 1) / experts_block;

    // gate: clamp(min = -inf, max = limit); up: clamp(min = -limit, max = limit).
    constexpr uint32_t kNegInfBits = 0xFF800000u;
    constexpr uint32_t neg_limit_bits = limit_bits ^ 0x80000000u;

    CircularBuffer in_cb(cb_input_id);
    CircularBuffer w_cb(cb_weights_id);
    CircularBuffer mm_cb(cb_mm_id);
    CircularBuffer out_cb(cb_out_id);
    CircularBuffer act_cb(cb_act_id);
    CircularBuffer down_w_cb(cb_down_w_id);
    CircularBuffer down_out_cb(cb_down_out_id);
    CircularBuffer rscalar_cb(cb_rscalar_id);
    CircularBuffer acc_cb(cb_acc_id);
    CircularBuffer wtmp_cb(cb_wtmp_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_input_id, cb_weights_id, cb_mm_id);
    matmul_init(cb_input_id, cb_weights_id);

    // Activation x is broadcast once and reused for every expert's gate_up matmul, in every block.
    if (swiglu_core) {
        in_cb.wait_front(k_tiles);
    }

    for (uint32_t blk = 0; blk < kNumBlocks; ++blk) {
        const uint32_t first_expert = blk * experts_block;
        const uint32_t remaining = num_active - first_expert;
        const uint32_t block_experts = remaining < experts_block ? remaining : experts_block;

        // ===============================================================================
        // PHASE 1: gate_up matmul + SwiGLU for the block's experts (SwiGLU cores only). Each
        // expert's 2-tile activation slice is pushed to cb_out for the writer to scatter to the
        // leader.
        // ===============================================================================
        if (swiglu_core) {
            for (uint32_t j = 0; j < block_experts; ++j) {
                w_cb.wait_front(gate_up_reserve_tiles);

                // ---- gate + up matmul -> cb_mm (gate tiles 0,1; up tiles 2,3). ----
                matmul_init(cb_input_id, cb_weights_id);
                reconfig_data_format(cb_weights_id, cb_input_id);
                pack_reconfig_data_format(cb_mm_id);
                mm_cb.reserve_back(kShardTileCols);

                // gate (weight tile (k, n) at k*kShardTileCols + n) -> dst n -> cb_mm n
                tile_regs_acquire();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    for (uint32_t k = 0; k < k_tiles; ++k) {
                        matmul_tiles(cb_input_id, cb_weights_id, k, k * kShardTileCols + n, n);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    pack_tile<true>(n, cb_mm_id, n);
                }
                tile_regs_release();

                // up (weight tile (k, n) at k*kShardTileCols + swiglu_tiles + n) -> dst n
                //    -> cb_mm swiglu_tiles + n
                tile_regs_acquire();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    for (uint32_t k = 0; k < k_tiles; ++k) {
                        matmul_tiles(cb_input_id, cb_weights_id, k, k * kShardTileCols + swiglu_tiles + n, n);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    pack_tile<true>(n, cb_mm_id, swiglu_tiles + n);
                }
                tile_regs_release();

                mm_cb.push_back(kShardTileCols);
                w_cb.pop_front(gate_up_reserve_tiles);

                // ---- SwiGLU: cb_mm (gate | up) -> cb_out (swiglu_tiles tiles). ----
                mm_cb.wait_front(kShardTileCols);
                copy_tile_to_dst_init_short(cb_mm_id);
                reconfig_data_format_srca(cb_mm_id);
                pack_reconfig_data_format(cb_out_id);
                out_cb.reserve_back(swiglu_tiles);

                tile_regs_acquire();
                // gate -> dst [0, swiglu_tiles), up -> dst [swiglu_tiles, 2*swiglu_tiles).
                for (uint32_t n = 0; n < kShardTileCols; ++n) {
                    copy_tile(cb_mm_id, n, n);
                }

                // gate = silu(clamp(gate, max = limit))
                clamp_tile_init();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    clamp_tile(n, kNegInfBits, limit_bits);
                }
                silu_tile_init();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    silu_tile(n);
                }

                // up = clamp(up, -limit, limit)
                clamp_tile_init();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    clamp_tile(swiglu_tiles + n, neg_limit_bits, limit_bits);
                }

                // out = gate * up
                mul_binary_tile_init();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    mul_binary_tile(n, swiglu_tiles + n, n);
                }

                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t n = 0; n < swiglu_tiles; ++n) {
                    pack_tile(n, cb_out_id);
                }
                tile_regs_release();

                mm_cb.pop_front(kShardTileCols);
                out_cb.push_back(swiglu_tiles);
            }
        }

        // ===============================================================================
        // PHASE 2: down matmul for the block's experts (all cores), each scaled by the tokens'
        // routing weights for that expert and accumulated into a single output tile row. The block's
        // gather + broadcast has made its activations resident in cb_act; the block's expert j
        // activation for all B tokens occupies tiles [j*i_tiles, (j+1)*i_tiles). For each expert:
        //     down_e = act_e @ down_w_e                      -> cb_mm staging (2 tiles, fp32)
        //     out   += routing_w[:, e] * down_e              (per-token multiply + add)
        // The loop is over distinct experts only (num_active is the size of the union of the tokens'
        // selections), so a shared expert's down weights are unpacked and multiplied once for the
        // batch. The running sum ping-pongs through cb_acc ACROSS blocks; the last expert of the last
        // block writes cb_down_out, which the writer drains once into the [1, B, H] DRAM output.
        // ===============================================================================
        const uint32_t block_act_tiles = block_experts * i_tiles;
        act_cb.wait_front(block_act_tiles);
        rscalar_cb.wait_front(block_experts);

        for (uint32_t j = 0; j < block_experts; ++j) {
            const uint32_t act_base = j * i_tiles;  // first activation tile of this block's expert j
            down_w_cb.wait_front(down_reserve_tiles);

            // ---- down matmul -> cb_mm staging (reuses the dead Phase-1 gate_up staging buffer). ----
            matmul_init(cb_act_id, cb_down_w_id);
            reconfig_data_format(cb_down_w_id, cb_act_id);
            pack_reconfig_data_format(cb_mm_id);
            mm_cb.reserve_back(kOutTilesPerCore);

            tile_regs_acquire();
            for (uint32_t n = 0; n < kOutTilesPerCore; ++n) {
                for (uint32_t k = 0; k < i_tiles; ++k) {
                    matmul_tiles(cb_act_id, cb_down_w_id, act_base + k, k * kOutTilesPerCore + n, n);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_mm_id);
            pack_tile(1, cb_mm_id);
            tile_regs_release();

            mm_cb.push_back(kOutTilesPerCore);
            down_w_cb.pop_front(down_reserve_tiles);

            // The accumulation spans every block, so what matters is an expert's position in the whole
            // selection, not in its block.
            const bool first = (first_expert + j == 0);
            const bool last = (first_expert + j == num_active - 1);

            // ---- multiply: weighted_e = routing_w[:, e] * down_e (per-token, elementwise). ----
            // cb_rscalar tile j holds this block's expert j routing weight for token b in tile row b
            // (splatted across the row), so one elementwise multiply scales every token of the tile row
            // by its own weight -- and scales away the tokens that did not select this expert, whose
            // weight is 0. This is why the shared fetch + shared matmul above are legal for a batch.
            //
            // For the very first expert there is nothing to accumulate yet, so the product goes
            // straight to the running accumulator (or the final output if it is the only expert).
            // Otherwise it is staged in cb_wtmp and added to the accumulator below.
            const uint32_t mul_dst_id = first ? (last ? cb_down_out_id : cb_acc_id) : cb_wtmp_id;
            CircularBuffer mul_dst_cb(mul_dst_id);
            mm_cb.wait_front(kOutTilesPerCore);
            mul_tiles_init(cb_mm_id, cb_rscalar_id);
            reconfig_data_format(cb_mm_id, cb_rscalar_id);
            pack_reconfig_data_format(mul_dst_id);
            mul_dst_cb.reserve_back(kOutTilesPerCore);

            tile_regs_acquire();
            mul_tiles(cb_mm_id, cb_rscalar_id, 0, j, 0);
            mul_tiles(cb_mm_id, cb_rscalar_id, 1, j, 1);
            tile_regs_commit();
            mm_cb.pop_front(kOutTilesPerCore);
            tile_regs_wait();
            pack_tile(0, mul_dst_id);
            pack_tile(1, mul_dst_id);
            tile_regs_release();
            mul_dst_cb.push_back(kOutTilesPerCore);

            // ---- accumulate: out = acc + weighted_e (only once there is a prior partial sum). ----
            if (!first) {
                const uint32_t add_dst_id = last ? cb_down_out_id : cb_acc_id;
                CircularBuffer add_dst_cb(add_dst_id);
                acc_cb.wait_front(kOutTilesPerCore);
                wtmp_cb.wait_front(kOutTilesPerCore);
                add_tiles_init(cb_acc_id, cb_wtmp_id);
                reconfig_data_format(cb_acc_id, cb_wtmp_id);
                pack_reconfig_data_format(add_dst_id);
                add_dst_cb.reserve_back(kOutTilesPerCore);

                tile_regs_acquire();
                add_tiles(cb_acc_id, cb_wtmp_id, 0, 0, 0);
                add_tiles(cb_acc_id, cb_wtmp_id, 1, 1, 1);
                tile_regs_commit();
                acc_cb.pop_front(kOutTilesPerCore);
                wtmp_cb.pop_front(kOutTilesPerCore);
                tile_regs_wait();
                pack_tile(0, add_dst_id);
                pack_tile(1, add_dst_id);
                tile_regs_release();
                add_dst_cb.push_back(kOutTilesPerCore);
            }
        }

        // Release the block's activations and routing tiles. On the leader this is what frees the
        // slot the next block will be gathered into, so it must happen before that block's sync.
        rscalar_cb.pop_front(block_experts);
        act_cb.pop_front(block_act_tiles);
    }

    if (swiglu_core) {
        in_cb.pop_front(k_tiles);
    }
}
