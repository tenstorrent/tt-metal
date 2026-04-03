// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dual gate+up matmul compute kernel.
//
// Computes gate_out = x @ gate_proj  and  up_out = x @ up_proj
// while reading x from the circular buffer ONLY ONCE per (M_block, K_block)
// combination.
//
// Loop order: M_outer → K_outer → N_inner
//
// For each (M_block, K_block):
//   1. BRISC has placed the x k-block into CB_IN0.
//   2. For each N_block:
//        a. Wait for in1_gate_cb, run gate matmul partial sum → gate_interm.
//        b. Wait for in1_up_cb,   run up   matmul partial sum → up_interm.
//        CB_IN0 is NOT popped between N blocks.
//   3. Pop CB_IN0.  (BRISC can now supply the next K block.)
//
// After all K_blocks, gate_interm and up_interm hold the complete outputs for
// this M_block.  A copy pass writes them to CB_GATE_OUT / CB_UP_OUT for the
// NCRISC writer.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

// ── matmul_blocks_with_offset ───────────────────────────────────────────────
// Identical logic to minimal_matmul's matmul_blocks but writes tiles at a
// column offset (n_tile_offset) inside the large intermediate CB that spans
// the full N dimension.
void matmul_blocks_with_offset(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_tiles,   // total N columns in out_cb
    const uint32_t n_tile_offset,  // column offset for this N block
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index,
                    in1_index,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    K_block_tiles);
                in0_index++;
                in1_index += N_block_tiles;
            }

            tile_regs_commit();
            tile_regs_wait();

            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile = n_tile_offset + N_start + w;
                    uint32_t tile_id = h_tile * full_N_tiles + w_tile;
                    pack_tile<true>(write_dst_index, out_cb, tile_id);
                    write_dst_index++;
                }
            }

            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

// ── copy_tiles_to_output ─────────────────────────────────────────────────────
// Reads num_tiles tiles from in_cb and writes them one by one to out_cb.
void copy_tiles_to_output(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);

    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0);
        pack_tile(0, out_cb);
        release_dst();
    }

    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in_cb, num_tiles);
}

// ── kernel_main ─────────────────────────────────────────────────────────────
void kernel_main() {
    constexpr uint32_t M_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t N_num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(8);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;       // x
    constexpr uint32_t in1_gate_cb = tt::CBIndex::c_1;  // gate_proj
    constexpr uint32_t in1_up_cb = tt::CBIndex::c_2;    // up_proj
    constexpr uint32_t gate_interm = tt::CBIndex::c_3;  // gate accumulator
    constexpr uint32_t up_interm = tt::CBIndex::c_4;    // up   accumulator
    constexpr uint32_t gate_out_cb = tt::CBIndex::c_5;  // gate output
    constexpr uint32_t up_out_cb = tt::CBIndex::c_6;    // up   output

    constexpr uint32_t in0_block_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t full_out_tiles = M_block_tiles * N_tiles;

    // Initialise matmul hardware for the gate-weight data format.
    // Since gate_proj and up_proj share the same dtype the FPU configuration
    // is valid for both CB_IN1_GATE and CB_IN1_UP.
    mm_init(in0_cb, in1_gate_cb, gate_interm);

    for (uint32_t m = 0; m < M_num_blocks; m++) {
        // Reserve the FULL N-wide intermediate for this M block.
        // The packer will fill it incrementally across K and N iterations.
        cb_reserve_back(gate_interm, full_out_tiles);
        cb_reserve_back(up_interm, full_out_tiles);

        // ── K-outer loop: x is read ONCE per k-block ──────────────────────
        for (uint32_t k = 0; k < K_num_blocks; k++) {
            // Wait for x k-block (filled by BRISC reader_x).
            cb_wait_front(in0_cb, in0_block_tiles);

            // ── N-inner loop: same x tiles used for every n-block ─────────
            for (uint32_t n = 0; n < N_num_blocks; n++) {
                uint32_t n_tile_offset = n * N_block_tiles;

                // ---- gate contribution ----
                mm_block_init_short(in0_cb, in1_gate_cb, false /*transpose*/, subblock_w, subblock_h, K_block_tiles);
                reconfig_data_format(in1_gate_cb, in0_cb);
                pack_reconfig_data_format(gate_interm);

                cb_wait_front(in1_gate_cb, in1_block_tiles);
                matmul_blocks_with_offset(
                    in0_cb,
                    in1_gate_cb,
                    gate_interm,
                    M_block_tiles,
                    N_block_tiles,
                    N_tiles,
                    n_tile_offset,
                    K_block_tiles,
                    subblock_h,
                    subblock_w);
                cb_pop_front(in1_gate_cb, in1_block_tiles);

                // ---- up contribution (same x, different weight CB) --------
                mm_block_init_short(in0_cb, in1_up_cb, false /*transpose*/, subblock_w, subblock_h, K_block_tiles);
                reconfig_data_format(in1_up_cb, in0_cb);
                pack_reconfig_data_format(up_interm);

                cb_wait_front(in1_up_cb, in1_block_tiles);
                matmul_blocks_with_offset(
                    in0_cb,
                    in1_up_cb,
                    up_interm,
                    M_block_tiles,
                    N_block_tiles,
                    N_tiles,
                    n_tile_offset,
                    K_block_tiles,
                    subblock_h,
                    subblock_w);
                cb_pop_front(in1_up_cb, in1_block_tiles);
            }
            // ── End N-inner loop ──────────────────────────────────────────

            // All N blocks have consumed x[m, k].  Pop so BRISC can load k+1.
            cb_pop_front(in0_cb, in0_block_tiles);

            // Enable L1 accumulation after the first k-block initialises the
            // intermediates.  Subsequent k-blocks ADD to the existing values.
            if (k == 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
        }
        // ── End K-outer loop ──────────────────────────────────────────────

        // Disable L1 accumulation before the copy pass.
        PACK((llk_pack_reconfig_l1_acc(0)));

        // Signal that the intermediates are fully accumulated.
        cb_push_back(gate_interm, full_out_tiles);
        cb_push_back(up_interm, full_out_tiles);

        // Copy accumulated results to output CBs for the NCRISC writer.
        copy_tiles_to_output(gate_interm, gate_out_cb, full_out_tiles);
        copy_tiles_to_output(up_interm, up_out_cb, full_out_tiles);
    }
}
