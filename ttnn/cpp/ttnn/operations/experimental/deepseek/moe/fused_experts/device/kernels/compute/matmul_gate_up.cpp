// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

// Per-core gate_up matmul + SwiGLU gate + down matmul (runs on every compute core).
//
// PHASE 1 (SwiGLU cores only -- those owning a slice of the I dim):
//   Each SwiGLU core owns a 2-tile (64 column) slice of the SwiGLU output I dimension. Its
//   gate_up weight shard ([K, 128] == [k_tiles, 4] tiles) holds the gate columns (tile cols
//   0,1) and paired up columns (tile cols 2,3) for that slice. For every selected expert:
//       gate = x @ gate_w   (cb_weights tile cols 0,1) -> [32, 64]
//       up   = x @ up_w     (cb_weights tile cols 2,3) -> [32, 64]
//       cb_out = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit) -> [32, 64]
//   where x (cb_input) is resident as Kt == H/32 activation tiles. cb_out (this core's 2-tile
//   slice of act[1, I]) is scattered by the writer to core {0,0}, gathered into the full
//   activation, and broadcast back into every core's cb_act.
//
// PHASE 2 (all cores): the down matmul, scaled by each expert's routing weight and accumulated
//   into a single output row. cb_act holds the full activation act[1, I] (i_tiles tiles, K
//   order). For each expert, each core multiplies its down weight shard ([I, H/64] ==
//   [i_tiles, 2] tiles) to produce its 2-tile (64 column) slice of down_e[1, H], scales it by
//   the expert's routing weight via a SCALAR broadcast (cb_rscalar), and accumulates:
//       down_e = act @ down_w   (cb_down_w tile (k, n) at k*2 + n) -> [32, 64]
//       out   += routing_w[e] * down_e   (summed over all active experts)
//   The running sum ping-pongs through cb_acc; the last expert writes cb_down_out, which the
//   writer drains once into the [1, 1, H] DRAM output row.
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
//   8: cb_act       (full gathered activation act[1, I], i_tiles tiles)
//   9: cb_down_w    (this core's per-expert [I, 64] down slice = i_tiles*2 tiles)
//  10: cb_down_out  (this core's 2 accumulated output tiles, written once)
//  11: cb_rscalar   (per-active-expert routing-weight scalar tiles for the SCALAR broadcast)
//  12: cb_acc       (running weighted-sum accumulator, ping-ponged across experts)
//  13: cb_wtmp      (staging for one expert's weighted down output before the accumulate)
//
// Runtime args:
//   0: col_start_tile (this core's first SwiGLU output tile = compute_index * 2)
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

    const uint32_t col_start_tile = get_arg_val<uint32_t>(0);
    const bool swiglu_core = col_start_tile < i_tiles;

    constexpr uint32_t kOutTilesPerCore = 2;
    constexpr uint32_t kShardTileCols = 2 * kOutTilesPerCore;  // gate 2 | up 2
    const uint32_t slice_tiles = k_tiles * kShardTileCols;
    // down weight shard: [I, 64] == [i_tiles, 2] tiles (full K = I, this core's 2 H-cols).
    const uint32_t down_slice_tiles = i_tiles * kOutTilesPerCore;

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

    mm_init(cb_input_id, cb_weights_id, cb_mm_id);

    // ===================================================================================
    // PHASE 1: gate_up matmul + SwiGLU for ALL experts (SwiGLU cores only). Each expert's
    // 2-tile activation slice is pushed to cb_out for the writer to scatter to the leader.
    // ===================================================================================
    if (swiglu_core) {
        // Activation x is broadcast once and reused for every expert's gate_up matmul.
        in_cb.wait_front(k_tiles);

        for (uint32_t e = 0; e < num_active; ++e) {
            w_cb.wait_front(slice_tiles);

            // ---- gate + up matmul -> cb_mm (gate tiles 0,1; up tiles 2,3). ----
            mm_init_short(cb_input_id, cb_weights_id);
            reconfig_data_format(cb_weights_id, cb_input_id);
            pack_reconfig_data_format(cb_mm_id);
            mm_cb.reserve_back(2 * kOutTilesPerCore);

            // gate (weight tile (k, n) at k*4 + n) -> dst 0,1 -> cb_mm 0,1
            tile_regs_acquire();
            for (uint32_t n = 0; n < kOutTilesPerCore; ++n) {
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    matmul_tiles(cb_input_id, cb_weights_id, k, k * kShardTileCols + n, n);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_mm_id, 0);
            pack_tile<true>(1, cb_mm_id, 1);
            tile_regs_release();

            // up (weight tile (k, n) at k*4 + 2 + n) -> dst 0,1 -> cb_mm 2,3
            tile_regs_acquire();
            for (uint32_t n = 0; n < kOutTilesPerCore; ++n) {
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    matmul_tiles(cb_input_id, cb_weights_id, k, k * kShardTileCols + kOutTilesPerCore + n, n);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_mm_id, 2);
            pack_tile<true>(1, cb_mm_id, 3);
            tile_regs_release();

            mm_cb.push_back(2 * kOutTilesPerCore);
            w_cb.pop_front(slice_tiles);

            // ---- SwiGLU: cb_mm (gate 0,1 | up 2,3) -> cb_out (2 tiles). ----
            mm_cb.wait_front(2 * kOutTilesPerCore);
            copy_tile_to_dst_init_short(cb_mm_id);
            reconfig_data_format_srca(cb_mm_id);
            pack_reconfig_data_format(cb_out_id);
            out_cb.reserve_back(kOutTilesPerCore);

            tile_regs_acquire();
            copy_tile(cb_mm_id, 0, 0);  // gate 0 -> dst 0
            copy_tile(cb_mm_id, 1, 1);  // gate 1 -> dst 1
            copy_tile(cb_mm_id, 2, 2);  // up 0   -> dst 2
            copy_tile(cb_mm_id, 3, 3);  // up 1   -> dst 3

            // gate = silu(clamp(gate, max = limit))
            clamp_tile_init();
            clamp_tile(0, kNegInfBits, limit_bits);
            clamp_tile(1, kNegInfBits, limit_bits);
            silu_tile_init();
            silu_tile(0);
            silu_tile(1);

            // up = clamp(up, -limit, limit)
            clamp_tile_init();
            clamp_tile(2, neg_limit_bits, limit_bits);
            clamp_tile(3, neg_limit_bits, limit_bits);

            // out = gate * up
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            mul_binary_tile(1, 3, 1);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out_id);
            pack_tile(1, cb_out_id);
            tile_regs_release();

            mm_cb.pop_front(2 * kOutTilesPerCore);
            out_cb.push_back(kOutTilesPerCore);
        }

        in_cb.pop_front(k_tiles);
    }

    // ===================================================================================
    // PHASE 2: down matmul for ALL experts (all cores), each scaled by its routing weight and
    // accumulated into a single output row. The single gather + broadcast has made the whole
    // [num_active, I] activation block resident in cb_act; expert e's activation occupies tiles
    // [e*i_tiles, (e+1)*i_tiles). For each expert:
    //     down_e = act_e @ down_w_e                      -> cb_mm staging (2 tiles, fp32)
    //     out   += routing_w[e] * down_e                 (SCALAR broadcast multiply + add)
    // The running sum ping-pongs through cb_acc; the final expert writes cb_down_out, which the
    // writer drains once into the [1, 1, H] DRAM output.
    // ===================================================================================
    const uint32_t act_total_tiles = num_active * i_tiles;
    act_cb.wait_front(act_total_tiles);
    rscalar_cb.wait_front(num_active);

    for (uint32_t e = 0; e < num_active; ++e) {
        const uint32_t act_base = e * i_tiles;  // first activation tile for expert e
        down_w_cb.wait_front(down_slice_tiles);

        // ---- down matmul -> cb_mm staging (reuses the dead Phase-1 gate_up staging buffer). ----
        mm_init_short(cb_act_id, cb_down_w_id);
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
        down_w_cb.pop_front(down_slice_tiles);

        const bool last = (e == num_active - 1);

        // ---- multiply: weighted_e = routing_w[e] * down_e (SCALAR broadcast). ----
        // For the first expert there is nothing to accumulate yet, so the product goes straight
        // to the running accumulator (or the final output if it is the only expert). Otherwise
        // it is staged in cb_wtmp and added to the accumulator below.
        const uint32_t mul_dst_id = (e == 0) ? (last ? cb_down_out_id : cb_acc_id) : cb_wtmp_id;
        CircularBuffer mul_dst_cb(mul_dst_id);
        mm_cb.wait_front(kOutTilesPerCore);
        mul_tiles_bcast_scalar_init_short(cb_mm_id, cb_rscalar_id);
        reconfig_data_format(cb_mm_id, cb_rscalar_id);
        pack_reconfig_data_format(mul_dst_id);
        mul_dst_cb.reserve_back(kOutTilesPerCore);

        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_mm_id, cb_rscalar_id, 0, e, 0);
        mul_tiles_bcast_scalar(cb_mm_id, cb_rscalar_id, 1, e, 1);
        tile_regs_commit();
        mm_cb.pop_front(kOutTilesPerCore);
        tile_regs_wait();
        pack_tile(0, mul_dst_id);
        pack_tile(1, mul_dst_id);
        tile_regs_release();
        mul_dst_cb.push_back(kOutTilesPerCore);

        // ---- accumulate: out = acc + weighted_e (only once there is a prior partial sum). ----
        if (e > 0) {
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

    rscalar_cb.pop_front(num_active);
    act_cb.pop_front(act_total_tiles);
}
