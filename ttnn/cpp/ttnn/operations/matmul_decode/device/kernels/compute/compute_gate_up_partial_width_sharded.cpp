// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/gelu.h"

using std::uint32_t;

// Fused gate+up+GeGLU partial-width-sharded matmul compute.
//
// Mirrors compute_partial_width_sharded.cpp but runs the partial matmul (phase 1) and the
// base-core K-reduction (phase 2) TWICE over the SINGLE gathered A: once for gate_b (in1, fused
// gelu) and once for up_b (in1b, no activation). Phase 3 then multiplies the two fully-reduced
// results into the single GeGLU output hid = gelu(A @ gate_w) * (A @ up_w) -- folding the
// downstream eltwise multiply into the op so the MLP emits one tensor, not two.
//
//   full_in0_cb (c_3): full gathered A  [M_tiles x K_tiles]   (published by reader, shared)
//   in1_cb  (c_1): this core's gate_b block [Kc_tiles x Nc_tiles]  -> gate_partial (c_4)
//   in1b_cb (c_6): this core's up_b   block [Kc_tiles x Nc_tiles]  -> up_partial   (c_7)
//   gate_reduce (c_5) / up_reduce (c_8): K_blocks partials gathered by the writer (base cores)
//   -> gate_out (c_2)  (gelu fused) , up_out (c_9)  [both internal scratch on base cores]
//   -> hid_out (c_10): gate_out * up_out  (the single device output shard)
//
// full_in0 is consumed (cb_pop_front) ONCE after BOTH phase-1 matmuls read it.

namespace {

// Phase 1: partial matmul of the gathered A with `in1_cb` -> `partial_cb`. (No activation here;
// gelu is fused into phase 2's pack so it lands on the reduced result.)
template <
    uint32_t M_tiles,
    uint32_t Kc_tiles,
    uint32_t Nc_tiles,
    uint32_t inA_K_tiles_per_core,
    uint32_t out_block_w,
    uint32_t out_block_h,
    uint32_t in0_block_w,
    uint32_t block_num_tiles,
    uint32_t sender_slice_tiles>
inline void phase1_partial(uint32_t full_in0_cb_id, uint32_t in1_cb_id, uint32_t partial_cb_id, uint32_t k_offset) {
    using namespace ckernel;
    mm_block_init(full_in0_cb_id, in1_cb_id, partial_cb_id, false, out_block_w, out_block_h, in0_block_w);
    cb_reserve_back(partial_cb_id, block_num_tiles);
    for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
        tile_regs_acquire();
        for (uint32_t kc = 0; kc < Kc_tiles; ++kc) {
            const uint32_t k_global = k_offset + kc;
            const uint32_t sender = k_global / inA_K_tiles_per_core;
            const uint32_t kc_local = k_global - sender * inA_K_tiles_per_core;
            const uint32_t in0_tile = sender * sender_slice_tiles + kc_local;
            const uint32_t in1_tile = kc * Nc_tiles + nc;
            matmul_block(
                full_in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t mt = 0; mt < out_block_h; ++mt) {
            pack_tile<true>(mt, partial_cb_id, mt * Nc_tiles + nc);
        }
        tile_regs_release();
    }
    cb_push_back(partial_cb_id, block_num_tiles);
}

// Phase 2 (base cores only): reduce K_blocks partials in reduce_cb (pairwise add) into out_cb,
// optionally fusing gelu (gate) onto the reduced tile.
template <
    uint32_t M_tiles,
    uint32_t Nc_tiles,
    uint32_t K_blocks,
    uint32_t block_num_tiles,
    bool do_gelu,
    bool gelu_approx>
inline void phase2_reduce(uint32_t reduce_cb_id, uint32_t out_cb_id) {
    using namespace ckernel;
    constexpr uint32_t reduce_num_tiles = K_blocks * block_num_tiles;
    cb_wait_front(reduce_cb_id, reduce_num_tiles);

    binary_op_init_common(reduce_cb_id, reduce_cb_id, out_cb_id);
    add_tiles_init(reduce_cb_id, reduce_cb_id, true /* acc_to_dest */);
    if (do_gelu) {
        gelu_tile_init<gelu_approx>();
    }
    cb_reserve_back(out_cb_id, block_num_tiles);
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
            if (do_gelu) {
                gelu_tile<gelu_approx>(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_cb_id, tile_in_block);
            tile_regs_release();
        }
    }
    cb_push_back(out_cb_id, block_num_tiles);
    cb_pop_front(reduce_cb_id, reduce_num_tiles);
}

// Phase 3 (base cores only): elementwise multiply of the two fully-reduced results
// gate_out (gelu fused) * up_out -> hid_out (the single device output shard). mul_tiles takes CB
// inputs, so the two reduced operands round-trip through their L1 CBs (block_num_tiles is small).
template <uint32_t block_num_tiles>
inline void phase3_multiply(uint32_t gate_out_cb_id, uint32_t up_out_cb_id, uint32_t hid_out_cb_id) {
    using namespace ckernel;
    cb_wait_front(gate_out_cb_id, block_num_tiles);
    cb_wait_front(up_out_cb_id, block_num_tiles);

    binary_op_init_common(gate_out_cb_id, up_out_cb_id, hid_out_cb_id);
    mul_tiles_init(gate_out_cb_id, up_out_cb_id);
    cb_reserve_back(hid_out_cb_id, block_num_tiles);
    for (uint32_t t = 0; t < block_num_tiles; ++t) {
        tile_regs_acquire();
        mul_tiles(gate_out_cb_id, up_out_cb_id, t, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, hid_out_cb_id, t);
        tile_regs_release();
    }
    cb_push_back(hid_out_cb_id, block_num_tiles);
    cb_pop_front(gate_out_cb_id, block_num_tiles);
    cb_pop_front(up_out_cb_id, block_num_tiles);
}

}  // namespace

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Kc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t fused_gelu_approx = get_compile_time_arg_val(6);

    const uint32_t k_idx = get_arg_val<uint32_t>(0);
    const uint32_t is_base = get_arg_val<uint32_t>(1);

    constexpr uint32_t full_in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;  // gate_b
    constexpr uint32_t gate_out_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t gate_partial_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t gate_reduce_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t in1b_cb_id = tt::CBIndex::c_6;  // up_b
    constexpr uint32_t up_partial_cb_id = tt::CBIndex::c_7;
    constexpr uint32_t up_reduce_cb_id = tt::CBIndex::c_8;
    constexpr uint32_t up_out_cb_id = tt::CBIndex::c_9;
    constexpr uint32_t hid_out_cb_id = tt::CBIndex::c_10;  // gate_out * up_out (single output)

    constexpr uint32_t full_in0_num_tiles = M_tiles * K_tiles;
    constexpr uint32_t in1_num_tiles = Kc_tiles * Nc_tiles;
    constexpr uint32_t block_num_tiles = M_tiles * Nc_tiles;
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t out_block_w = 1;
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    const uint32_t k_offset = k_idx * Kc_tiles;

    // ---- Phase 1: partial matmul for BOTH weights over the single gathered A ----
    cb_wait_front(full_in0_cb_id, full_in0_num_tiles);
    cb_wait_front(in1_cb_id, in1_num_tiles);
    cb_wait_front(in1b_cb_id, in1_num_tiles);

    phase1_partial<
        M_tiles,
        Kc_tiles,
        Nc_tiles,
        inA_K_tiles_per_core,
        out_block_w,
        out_block_h,
        in0_block_w,
        block_num_tiles,
        sender_slice_tiles>(full_in0_cb_id, in1_cb_id, gate_partial_cb_id, k_offset);
    phase1_partial<
        M_tiles,
        Kc_tiles,
        Nc_tiles,
        inA_K_tiles_per_core,
        out_block_w,
        out_block_h,
        in0_block_w,
        block_num_tiles,
        sender_slice_tiles>(full_in0_cb_id, in1b_cb_id, up_partial_cb_id, k_offset);

    // A is consumed once, after BOTH partial matmuls have read it.
    cb_pop_front(full_in0_cb_id, full_in0_num_tiles);

    if (is_base == 0) {
        return;
    }
    // ---- Phase 2: reduce K_blocks partials for gate (gelu) and up (no activation) ----
    phase2_reduce<M_tiles, Nc_tiles, K_blocks, block_num_tiles, /*do_gelu=*/true, (bool)fused_gelu_approx>(
        gate_reduce_cb_id, gate_out_cb_id);
    phase2_reduce<M_tiles, Nc_tiles, K_blocks, block_num_tiles, /*do_gelu=*/false, false>(
        up_reduce_cb_id, up_out_cb_id);

    // ---- Phase 3: GeGLU multiply gate_out * up_out -> hid_out (single device output) ----
    phase3_multiply<block_num_tiles>(gate_out_cb_id, up_out_cb_id, hid_out_cb_id);
}
