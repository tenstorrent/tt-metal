// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "partial_phases.hpp"  // phase1_partial (shared with the gate_up compute)

using std::uint32_t;

// Partial-width-sharded matmul compute.
//
// Phase 1 (every B core): compute a partial product over this core's K-slice.
//   full_in0_cb (c_3): full gathered A  [M_tiles x K_tiles]   (published by reader)
//   in1_cb (c_1):      this core's B block [Kc_tiles x Nc_tiles] (resident in L1)
//   -> partial_cb (c_4): partial [M_tiles x Nc_tiles]
//
// Phase 2 (base cores only, k_idx == 0): sum the K_blocks partials gathered by the
// writer into reduce_cb (c_5) and write the final output shard.
//   reduce_cb (c_5): K_blocks * [M_tiles x Nc_tiles] partials (block k == k_idx)
//   -> out_cb (c_2): output shard [M_tiles x Nc_tiles]
//
// full A (full_in0) layout (built by reader): A is width(K)-sharded across
// `num_senders = K_tiles / inA_K_tiles_per_core` sender cores, each holding a
// contiguous [M_tiles*32, inA_K_tiles_per_core*32] slice in TILE row-major order,
// and the reader copies each sender's whole slice into full_in0 at offset
// sender*M_tiles*inA_K_tiles_per_core. So full_in0 is SENDER-MAJOR: global tile
// (m, k_global) lives at
//   sender * (M_tiles * inA_K_tiles_per_core) + m * inA_K_tiles_per_core + kc_local
// with sender = k_global / inA_K_tiles_per_core, kc_local = k_global % ...
// (For M_tiles==1 this collapses to k_global, matching the old behaviour.)
//
// M_tiles>1: phase 1 computes the entire M dimension of the partial in a single
// DST block (out_block_h = M_tiles), so each matmul_block produces all M-tiles of
// a given N-tile at once. The program factory asserts M < 256, keeping M_tiles <= 8
// so the block fits in DST. Phase 2 reduces per (mt,nc) across the K_blocks slabs
// in reduce_cb (each slab is one [M_tiles x Nc_tiles] block).
//
// matmul_block accumulation: matmul_block does NOT internally reduce over kt_dim.
// kt_dim (= inA_K_tiles_per_core) is only the in0 *row stride* (K-tiles per M-row
// in full_in0's per-sender slice). Each call multiplies one K-column slice
// (rt_dim=M_tiles in0 tiles x ct_dim=1 in1 tile) into DST; the reduction over this
// core's K-slice is done by the explicit kc loop, accumulating into the same DST.

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Kc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t fused_gelu = get_compile_time_arg_val(6);
    constexpr uint32_t fused_gelu_approx = get_compile_time_arg_val(7);
    // Gated-residual epilogue: out = residual + gate * (A @ B). gate is per-channel over N
    // (replicated down the rows so a plain mul_tiles works); residual is this base core's N-slice.
    constexpr uint32_t fused_residual = get_compile_time_arg_val(8);
    constexpr uint32_t residual_cb_id = get_compile_time_arg_val(9);
    constexpr uint32_t gate_cb_id = get_compile_time_arg_val(10);
    constexpr uint32_t mm_cb_id = get_compile_time_arg_val(11);
    constexpr uint32_t mmg_cb_id = get_compile_time_arg_val(12);  // scratch for gate * mm

    const uint32_t k_idx = get_arg_val<uint32_t>(0);
    const uint32_t is_base = get_arg_val<uint32_t>(1);

    constexpr uint32_t full_in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t partial_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t reduce_cb_id = tt::CBIndex::c_5;

    constexpr uint32_t full_in0_num_tiles = M_tiles * K_tiles;
    constexpr uint32_t in1_num_tiles = Kc_tiles * Nc_tiles;
    constexpr uint32_t block_num_tiles = M_tiles * Nc_tiles;
    constexpr uint32_t reduce_num_tiles = K_blocks * block_num_tiles;
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    // The whole M dimension of the partial is computed in one DST block.
    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t out_block_w = 1;
    // in0 row stride within a sender slice (K-tiles per M-row) == matmul kt_dim.
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    // ---- Phase 1: partial matmul (shared with the gate_up compute) ----
    cb_wait_front(full_in0_cb_id, full_in0_num_tiles);
    cb_wait_front(in1_cb_id, in1_num_tiles);
    const uint32_t k_offset = k_idx * Kc_tiles;  // this core's K-slice start (global K-tile)
    phase1_partial<
        M_tiles,
        Kc_tiles,
        Nc_tiles,
        inA_K_tiles_per_core,
        out_block_w,
        out_block_h,
        in0_block_w,
        block_num_tiles,
        sender_slice_tiles>(full_in0_cb_id, in1_cb_id, partial_cb_id, k_offset);
    cb_pop_front(full_in0_cb_id, full_in0_num_tiles);

    if (is_base == 0) {
        return;
    }
    // ---- Phase 2: reduce K_blocks partials (base cores only) ----
    // reduce_cb holds K_blocks contiguous [M_tiles x Nc_tiles] slabs; for each
    // (mt,nc) pairwise accumulate the matching tile across all K_blocks slabs.
    cb_wait_front(reduce_cb_id, reduce_num_tiles);

    binary_op_init_common(reduce_cb_id, reduce_cb_id, out_cb_id);
    add_tiles_init(reduce_cb_id, reduce_cb_id, true /* acc_to_dest */);
    if (fused_gelu) {
        gelu_tile_init<(bool)fused_gelu_approx>();
    }

    if (fused_residual) {
        // Gated-residual epilogue: out = residual + gate * (A @ B). The reader has staged this base
        // core's gate (resident, replicated down the rows) and residual N-slice into their CBs.
        cb_wait_front(gate_cb_id, Nc_tiles);
        cb_wait_front(residual_cb_id, block_num_tiles);
    }

    cb_reserve_back(out_cb_id, block_num_tiles);
    if (fused_residual) {
        cb_reserve_back(mm_cb_id, block_num_tiles);
    }
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
            const uint32_t tile_in_block = mt * Nc_tiles + nc;
            tile_regs_acquire();
            for (uint32_t block = 0; block < K_blocks; block += 2) {
                add_tiles(
                    reduce_cb_id,
                    reduce_cb_id,
                    block * block_num_tiles + tile_in_block,
                    (block + 1) * block_num_tiles + tile_in_block,
                    0);
            }
            // Fuse the GeGLU gate activation directly onto the reduced output tile in DST,
            // so the gate projection needs no separate elementwise gelu op afterwards.
            if (fused_gelu) {
                gelu_tile<(bool)fused_gelu_approx>(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            // Without the residual epilogue: pack the reduced (+gelu) tile straight to the output.
            // With it: stash the reduced tile in mm_cb, then apply gate*mul + residual add below.
            pack_tile<true>(0, fused_residual ? mm_cb_id : out_cb_id, tile_in_block);
            tile_regs_release();
        }
    }
    if (!fused_residual) {
        cb_push_back(out_cb_id, block_num_tiles);
        cb_pop_front(reduce_cb_id, reduce_num_tiles);
        return;
    }
    cb_push_back(mm_cb_id, block_num_tiles);
    cb_pop_front(reduce_cb_id, reduce_num_tiles);

    // ---- Gated-residual epilogue ----
    // mm_cb holds the reduced matmul result. gate_cb holds Nc_tiles per-channel gate tiles (the
    // gate for N-column nc is replicated down all TILE_H rows, so a plain mul_tiles applies it per
    // channel without a broadcast). residual_cb holds this core's interleaved [M_tiles x Nc_tiles]
    // N-slice. Compute out[mt,nc] = residual[mt,nc] + gate[nc] * mm[mt,nc].
    cb_wait_front(mm_cb_id, block_num_tiles);
    cb_reserve_back(mmg_cb_id, block_num_tiles);
    // gate * mm -> mmg (gate tile index is nc -- one gate tile per N-column, replicated down rows).
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
            const uint32_t tile_in_block = mt * Nc_tiles + nc;
            tile_regs_acquire();
            mul_tiles_init(mm_cb_id, gate_cb_id);
            mul_tiles(mm_cb_id, gate_cb_id, tile_in_block, nc, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, mmg_cb_id, tile_in_block);
            tile_regs_release();
        }
    }
    cb_push_back(mmg_cb_id, block_num_tiles);
    cb_pop_front(mm_cb_id, block_num_tiles);
    // residual + (gate*mm) -> out.
    cb_wait_front(mmg_cb_id, block_num_tiles);
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
            const uint32_t tile_in_block = mt * Nc_tiles + nc;
            tile_regs_acquire();
            add_tiles_init(residual_cb_id, mmg_cb_id);
            add_tiles(residual_cb_id, mmg_cb_id, tile_in_block, tile_in_block, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_cb_id, tile_in_block);
            tile_regs_release();
        }
    }
    cb_push_back(out_cb_id, block_num_tiles);
    cb_pop_front(mmg_cb_id, block_num_tiles);
    cb_pop_front(gate_cb_id, Nc_tiles);
    cb_pop_front(residual_cb_id, block_num_tiles);
}
