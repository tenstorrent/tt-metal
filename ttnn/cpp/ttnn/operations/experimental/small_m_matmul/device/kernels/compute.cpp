// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"

void copy_block_to_cb(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

// Like copy_block_to_cb but NEVER applies the fused activation. Used by split-K NON-root bands (bottom band of a
// Pk>1 chain) so they forward the RAW matmul partial up the reduction chain; the activation is applied
// exactly once at the reduction ROOT (is_top). copy_block_to_cb itself always applies activation when defined and
// is used only where the block IS the final output (no-fusion path, or activation-only root).
void copy_block_raw(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            copy_tile(in_cb, tile_id, 0 /*dst*/);
            pack_tile(0, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

// Split-K fusion root helper: intermediate_cb += reduce_cb, IN PLACE (result stays in intermediate_cb so the
// existing bias/activation/addcmul primitives can consume it exactly as on the non-split-K path). Uses the
// single-slot in-place refill idiom (wait/pop front, reserve/push back the same slot) already used by
// add_bias_and_addcmul_block. Applies NO fusion — bias/activation/addcmul are applied afterward, once.
void reduce_add_in_place(uint32_t intermediate_cb, uint32_t reduce_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    const uint32_t n_tiles = M_block_tiles * N_block_tiles;
    add_tiles_init(intermediate_cb, reduce_cb);
    reconfig_data_format(intermediate_cb, reduce_cb);
    pack_reconfig_data_format(intermediate_cb);
    cb_wait_front(intermediate_cb, n_tiles);
    for (uint32_t t = 0; t < n_tiles; t++) {
        acquire_dst();
        add_tiles(intermediate_cb, reduce_cb, t, t, 0 /*dst*/);
        pack_tile(0, intermediate_cb);
        release_dst();
    }
    cb_pop_front(intermediate_cb, n_tiles);
    cb_reserve_back(intermediate_cb, n_tiles);
    cb_push_back(intermediate_cb, n_tiles);
}

// Split-K plan B: out = a + b, full elementwise (both M_block_tiles x N_block_tiles). Used by the
// column reduction to add this band's matmul partial (a) to the running sum forwarded up from the band
// below (b). Pushes out_cb one M-row at a time, matching copy_block_to_cb/add_bias_block.
void reduce_add_block(uint32_t a_cb, uint32_t b_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    add_tiles_init(a_cb, b_cb);
    reconfig_data_format(a_cb, b_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            add_tiles(a_cb, b_cb, tile_id, tile_id, 0 /*dst*/);
            pack_tile(0, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

#ifdef RSCATTER
// Fused epilogue for ONE fully-reduced reduce-scatter slice. The writer feeds bias / residual / gate for the
// slice in SLICE ORDER, so every operand tile index equals the slice tile index -- no broadcast variants and no
// row streaming are needed here, unlike the block epilogue. Applied exactly once per slice, by its owner.
//
// out = ta + scalar * act(acc + bias) * tb, with each stage compiled in only when its define is set. Stage 1
// rewrites acc_cb in place using the reserve/push refill idiom (same as add_bias_and_addcmul_block).
// BROADCAST MATTERS even though the operands are index-aligned: bias and a [1,N] gate hold their values only in
// ROW 0 of each tile (the rest is tile padding), so they must be applied with a ROW broadcast. Plain elementwise
// ops there consume 31 rows of padding -- measured PCC 0.18 for the gate, and for bias a FALSE PASS at 0.9997,
// because a bias applied to only 1 row in 32 barely moves PCC. The residual is a true [M,N] tensor and is
// correctly applied elementwise.
[[maybe_unused]] void rs_epilogue_slice(
    uint32_t acc_cb,
    uint32_t bias_cb,
    uint32_t ta_cb,
    uint32_t tb_cb,
    uint32_t scalar_value,
    uint32_t out_cb,
    uint32_t nt,
    uint32_t slot,
    uint32_t broadcast_ternary_b) {
    // `nt` is the owner's slice length; `slot` is the CB slot size (the LARGEST slice). All acc_cb bookkeeping
    // uses `slot` while the tile loops use `nt`: with an uneven partition nt < slot for some owners, and
    // popping nt after the caller pushed slot would desynchronise the CB.
    constexpr uint32_t DST_ID = 0;
    [[maybe_unused]] uint32_t fused_act_dst_id = DST_ID;

#if defined(FUSE_BIAS) || defined(SFPU_OP_INIT_ACTIVATION)
    // ---- stage 1: acc = act(acc + bias), in place ----
    cb_wait_front(acc_cb, slot);
#ifdef FUSE_BIAS
    cb_wait_front(bias_cb, nt);
    add_bcast_rows_init_short(acc_cb, bias_cb);  // bias is [1,N]: row broadcast, not elementwise
    reconfig_data_format(acc_cb, bias_cb);
#else
    copy_tile_to_dst_init_short(acc_cb);
    reconfig_data_format_srca(acc_cb);
#endif
    pack_reconfig_data_format(acc_cb);
    for (uint32_t i = 0; i < nt; ++i) {
        tile_regs_acquire();
#ifdef FUSE_BIAS
        add_tiles_bcast<BroadcastType::ROW>(acc_cb, bias_cb, i, i, DST_ID);
#else
        copy_tile(acc_cb, i, DST_ID);
#endif
#ifdef SFPU_OP_INIT_ACTIVATION
        SFPU_OP_FUNC_ACTIVATION
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(DST_ID, acc_cb);
        tile_regs_release();
    }
#ifdef FUSE_BIAS
    cb_pop_front(bias_cb, nt);
#endif
    cb_pop_front(acc_cb, slot);
    cb_reserve_back(acc_cb, slot);
    cb_push_back(acc_cb, slot);
#endif  // bias or activation

#ifdef FUSE_TERNARY
    // ---- stage 2: acc = acc * tb * scalar, in place ----
    cb_wait_front(acc_cb, slot);
    cb_wait_front(tb_cb, nt);
#ifndef TERNARY_B_IS_FLOAT32
    if (broadcast_ternary_b) {
        mul_bcast_rows_init_short(acc_cb, tb_cb);  // gate [1,N]: values in row 0 only
    } else {
        mul_tiles_init(acc_cb, tb_cb);  // gate [M,N]: genuine elementwise
    }
#endif
    binop_with_scalar_tile_init();
    reconfig_data_format(acc_cb, tb_cb);
    pack_reconfig_data_format(acc_cb);
    for (uint32_t i = 0; i < nt; ++i) {
        tile_regs_acquire();
#ifndef TERNARY_B_IS_FLOAT32
        if (broadcast_ternary_b) {
            mul_tiles_bcast<BroadcastType::ROW>(acc_cb, tb_cb, i, i, DST_ID);
        } else {
            mul_tiles(acc_cb, tb_cb, i, i, DST_ID);
        }
#else
        constexpr uint32_t TB_DST_ID = 1;
        if (broadcast_ternary_b) {
            unary_bcast_init<BroadcastType::ROW>(tb_cb, acc_cb);
            unary_bcast<BroadcastType::ROW>(tb_cb, i, TB_DST_ID);
        } else {
            copy_tile_to_dst_init_short(tb_cb);
            copy_tile(tb_cb, i, TB_DST_ID);
        }
        copy_tile_to_dst_init_short(acc_cb);
        copy_tile(acc_cb, i, DST_ID);
        mul_binary_tile_init();
        mul_binary_tile(DST_ID, TB_DST_ID, DST_ID);
#endif
        mul_unary_tile(DST_ID, scalar_value);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(DST_ID, acc_cb);
        tile_regs_release();
    }
    cb_pop_front(tb_cb, nt);
    cb_pop_front(acc_cb, slot);
    cb_reserve_back(acc_cb, slot);
    cb_push_back(acc_cb, slot);

    // ---- stage 3: out = ta + acc ----
    cb_wait_front(acc_cb, slot);
    cb_wait_front(ta_cb, nt);
    add_tiles_init(acc_cb, ta_cb);
    reconfig_data_format(acc_cb, ta_cb);
    pack_reconfig_data_format(out_cb);
    for (uint32_t i = 0; i < nt; ++i) {
        tile_regs_acquire();
        add_tiles(acc_cb, ta_cb, i, i, DST_ID);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(DST_ID, out_cb);
        tile_regs_release();
    }
    cb_pop_front(ta_cb, nt);
    cb_pop_front(acc_cb, slot);
#else
    // no addcmul: move the (bias/activation-applied) slice to out_cb
    cb_wait_front(acc_cb, slot);
    copy_tile_to_dst_init_short(acc_cb);
    reconfig_data_format_srca(acc_cb);
    pack_reconfig_data_format(out_cb);
    for (uint32_t i = 0; i < nt; ++i) {
        tile_regs_acquire();
        copy_tile(acc_cb, i, DST_ID);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(DST_ID, out_cb);
        tile_regs_release();
    }
    cb_pop_front(acc_cb, slot);
#endif  // FUSE_TERNARY
}

// Ring reduce-scatter helpers. rs_copy_chunk: copy n_tiles from in_cb starting at tile in_tile_off -> out_cb,
// used to SEED the ring with this core's own chunk. No cb push/pop (the caller manages both CBs).
void rs_copy_chunk(uint32_t in_cb, uint32_t in_tile_off, uint32_t out_cb, uint32_t n_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, in_tile_off + i, 0);
        pack_tile(0, out_cb);
        release_dst();
    }
}
// out_cb[0..n) = acc_cb[acc_tile_off ..) + add_cb[0..n). acc_cb is this core's resident FP32 matmul partial
// (read at a chunk offset); add_cb is the received running sum. This adds MY contribution for the chunk that is
// currently travelling round the ring. Every add is FP32 in DST, exactly as in the chain's reduce_add_block.
void rs_add_chunk(uint32_t acc_cb, uint32_t acc_tile_off, uint32_t add_cb, uint32_t out_cb, uint32_t n_tiles) {
    add_tiles_init(acc_cb, add_cb);
    reconfig_data_format(acc_cb, add_cb);
    pack_reconfig_data_format(out_cb);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        acquire_dst();
        add_tiles(acc_cb, add_cb, acc_tile_off + i, i, 0 /*dst*/);
        pack_tile(0, out_cb);
        release_dst();
    }
}
#endif

// For caller: if FUSE_TERNARY defined then out_cb == in_cb
/**
 * Add bias to input block
 * Performs: output = input + bias (row broadcast)
 *
 * stream_output:
 *   - true: Pushes tiles one row at a time (for intermediate output to next stage)
 *   - false: Pushes all tiles at end (for final output)
 */
void add_bias_block(uint32_t in_cb, uint32_t bias_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    add_bcast_rows_init_short(in_cb, bias_cb);
    reconfig_data_format(in_cb, bias_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, tile_id, n, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

void add_bias_and_addcmul_block(
    uint32_t intermediate_cb,
    uint32_t bias_cb,
    uint32_t ternary_a_cb,
    uint32_t ternary_b_cb,
    uint32_t scalar_value,
    uint32_t out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t broadcast_ternary_b) {
    // Note: unary_bcast_tile does not work with fp32_acc_to_dest=True.
    // As a workaround, we perform addcmul through multiple LLKs calls (mul_tiles, mul_unary_tile, add_tiles_bcast).

    const uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t DST_ID = 0;
#ifdef FUSE_BIAS
    // ============================================
    // STEP 1: Add bias block
    // Read from intermediate_cb and write back to intermediate_cb
    // ============================================

    add_bcast_rows_init_short(intermediate_cb, bias_cb);
    reconfig_data_format(intermediate_cb, bias_cb);
    pack_reconfig_data_format(intermediate_cb);

    // Wait for ALL input data ONCE at the beginning
    cb_wait_front(bias_cb, N_block_tiles);

    // Unpacker waits for intermediate_cb to be ready
    cb_wait_front(intermediate_cb, out_block_num_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            uint32_t tile_id = m * N_block_tiles + n;

            tile_regs_acquire();
            add_tiles_bcast<BroadcastType::ROW>(intermediate_cb, bias_cb, tile_id, n, DST_ID);

            tile_regs_commit();

            tile_regs_wait();
            pack_tile(DST_ID, intermediate_cb);
            tile_regs_release();
        }
    }

    // Pop input and push output ONCE at the end
    cb_pop_front(bias_cb, N_block_tiles);

    cb_pop_front(intermediate_cb, out_block_num_tiles);

    // Restore intermediate_cb to ready (+ sync packer/unpacker)
    cb_reserve_back(intermediate_cb, out_block_num_tiles);
    cb_push_back(intermediate_cb, out_block_num_tiles);
#endif  // FUSE_BIAS

    // ============================================
    // STEP 2: Multiply by ternary_b and scalar
    // Read from intermediate_cb and write back to intermediate_cb
    // broadcast_ternary_b: 1 = single row broadcast, 0 = row-by-row streaming
    // ============================================

    cb_wait_front(intermediate_cb, out_block_num_tiles);

    uint32_t tile_id = 0;

    if (broadcast_ternary_b) {
        // === BROADCAST: single row, wait/pop once ===
        cb_wait_front(ternary_b_cb, N_block_tiles);

#ifndef TERNARY_B_IS_FLOAT32
        mul_bcast_rows_init_short(intermediate_cb, ternary_b_cb);
#else
        unary_bcast_init<BroadcastType::ROW>(ternary_b_cb, intermediate_cb);
#endif  // TERNARY_B_IS_FLOAT32

        binop_with_scalar_tile_init();
        reconfig_data_format(intermediate_cb, ternary_b_cb);
        pack_reconfig_data_format(intermediate_cb);

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                mul_tiles_bcast<BroadcastType::ROW>(intermediate_cb, ternary_b_cb, tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                unary_bcast_init<BroadcastType::ROW>(ternary_b_cb, intermediate_cb);
                unary_bcast<BroadcastType::ROW>(ternary_b_cb, n, TERNARY_B_DST_ID);

                copy_tile_to_dst_init_short(intermediate_cb);
                copy_tile(intermediate_cb, tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_cb);
                tile_regs_release();
                tile_id++;
            }
        }

        cb_pop_front(ternary_b_cb, N_block_tiles);
    } else {
        // === NO BROADCAST: row-by-row, wait/pop per M row ===
#ifndef TERNARY_B_IS_FLOAT32
        mul_tiles_init(intermediate_cb, ternary_b_cb);
#endif
        binop_with_scalar_tile_init();
        reconfig_data_format(intermediate_cb, ternary_b_cb);
        pack_reconfig_data_format(intermediate_cb);

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            cb_wait_front(ternary_b_cb, N_block_tiles);
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                mul_tiles(intermediate_cb, ternary_b_cb, tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                copy_tile_to_dst_init_short(ternary_b_cb);
                copy_tile(ternary_b_cb, n, TERNARY_B_DST_ID);

                copy_tile_to_dst_init_short(intermediate_cb);
                copy_tile(intermediate_cb, tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_cb);
                tile_regs_release();
                tile_id++;
            }
            cb_pop_front(ternary_b_cb, N_block_tiles);
        }
    }

    cb_pop_front(intermediate_cb, out_block_num_tiles);

    // 'refill' intermediate_cb (also synchronize packer/unpacker)
    cb_reserve_back(intermediate_cb, out_block_num_tiles);
    cb_push_back(intermediate_cb, out_block_num_tiles);

    cb_wait_front(intermediate_cb, out_block_num_tiles);

    add_tiles_init(intermediate_cb, ternary_a_cb);
    reconfig_data_format(intermediate_cb, ternary_a_cb);
    pack_reconfig_data_format(out_cb);

    tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of ternary_a tiles
        cb_wait_front(ternary_a_cb, N_block_tiles);

        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();

            // ternary_a_cb is pushed one row at a time, so use column index n
            add_tiles(intermediate_cb, ternary_a_cb, tile_id, n, DST_ID);

            tile_regs_commit();

            tile_regs_wait();
            pack_tile(DST_ID, out_cb);
            tile_regs_release();
            tile_id++;
        }

        cb_pop_front(ternary_a_cb, N_block_tiles);
        cb_push_back(out_cb, N_block_tiles);
    }

    cb_pop_front(intermediate_cb, out_block_num_tiles);
}

// Slightly modified from compute_common.hpp
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const uint32_t in0_base = 0,
    const uint32_t in1_base = 0) {
    uint32_t in0_index_offset = in0_base;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = in1_base;
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
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();

            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                    dst_index++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    // split-K plan B: 1 if this is the bottom K-band (no incoming running sum), else 0. Always present.
    [[maybe_unused]] const uint32_t is_reduce_bottom = get_arg_val<uint32_t>(argidx++);
#ifdef RSCATTER
    // Ring reduce-scatter: this core's cycle position, the cycle size P=Pk, and chunk_tiles (tiles per chunk =
    // M_block*N_block / Pk). Follow is_reduce_bottom; unfused only, so they cannot collide with is_reduce_top.
    const uint32_t rs_ring_pos = get_arg_val<uint32_t>(argidx++);
    const uint32_t rs_P = get_arg_val<uint32_t>(argidx++);
    const uint32_t rs_T = get_arg_val<uint32_t>(argidx++);  // tiles per sub-block (== out_block_num_tiles)
#endif

// Any fusion active => the reduction ROOT (is_top) applies bias/activation/addcmul exactly once after the
// split-K partials are summed. Non-root bands forward the RAW partial (no fusion). When no fusion is active
// this macro is undefined and the output stage is byte-identical to the historical no-fusion path.
#if defined(FUSE_BIAS) || defined(FUSE_TERNARY) || defined(SFPU_OP_INIT_ACTIVATION)
#define SMALL_M_FUSED 1
    // is_top: 1 on the reduction root (Pk==1 => every core; Pk>1 => the top K-band). Present only when fused.
    [[maybe_unused]] const uint32_t is_reduce_top = get_arg_val<uint32_t>(argidx++);
#endif

#ifdef FUSE_TERNARY
    const uint32_t fused_ternary_scalar_uint = get_arg_val<uint32_t>(argidx++);
    const uint32_t broadcast_ternary_b = get_arg_val<uint32_t>(argidx++);
#else
    // Default value when ternary is not fused (not used, helps compiler optimize)
    constexpr uint32_t fused_ternary_scalar_uint = 0;
    constexpr uint32_t broadcast_ternary_b = 1;
#endif

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    constexpr uint32_t in2_cb = tt::CBIndex::c_4;
    constexpr uint32_t ternary_a_cb = tt::CBIndex::c_5;
    constexpr uint32_t ternary_b_cb = tt::CBIndex::c_6;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_7;  // split-K B: running sum forwarded up from the band below

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);
    matmul_init(in0_cb, in1_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t M_num_subblocks = M_block_tiles / subblock_h;
    constexpr uint32_t N_num_subblocks = N_block_tiles / subblock_w;

    uint32_t current_M_block_tiles = M_block_tiles;
    uint32_t current_N_block_tiles = N_block_tiles;
    uint32_t current_subblock_h = subblock_h;
    uint32_t current_subblock_w = subblock_w;

    // Large-Mt ring: cb0 holds the full k-slice (all K_num_blocks in0 blocks, block-major). It is filled ONCE
    // (the writer pushes it incrementally, W compute blocks per ring step) and kept resident so it can be
    // reused across the N_blocks_per_core N-sub-blocks; the k-loop addresses block k_block via the in0_base
    // tile offset instead of popping. Popped once at the end.
    //
    // Progressive startup: NO startup barrier. The first N-sub-block's k-loop instead waits cumulatively per K
    // block (below), so the first matmul begins as soon as the first ring shard arrives while ring forwarding
    // + in1 reading continue concurrently. Later N-sub-blocks reuse the now-complete resident slice.

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        current_M_block_tiles = m_tile_end - m_tile;
        current_subblock_h = std::min(current_M_block_tiles, subblock_h);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            current_N_block_tiles = n_tile_end - n_tile;
            current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            matmul_block_init(
                in0_cb,
                in1_cb,
                false /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermediate_cb);
            // Accumulation buffer
            cb_reserve_back(intermediate_cb, out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                // Progressive cumulative wait: begin matmul k_block as soon as the writer's incremental per-step
                // ring pushes make CUMULATIVE (k_block+1) in0 blocks available. Only the FIRST resident traversal
                // waits (M_blocks_per_core==1, first N-sub-block); once the slice is complete it stays resident
                // and later N-sub-blocks reuse it with no further waits. Repeated cb_wait_front WITHOUT an
                // intervening pop requires cumulative counts (CB API contract) — satisfied since
                // (k_block+1)*in0_block_num_tiles is strictly increasing and CB0 is popped only once, after all
                // reuse (below). The writer pushes W blocks at a time, so a wait for a mid-batch boundary is
                // simply satisfied when that W-batch lands.
                if (m_block_iter == 0 && n_block_iter == 0) {
                    cb_wait_front(in0_cb, (k_block + 1) * in0_block_num_tiles);
                }
                cb_wait_front(in1_cb, in1_block_num_tiles);

                matmul_blocks(
                    in0_cb,
                    in1_cb,
                    intermediate_cb,
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w,
                    k_block * in0_block_num_tiles);  // block-major offset into the resident k-slice

                cb_pop_front(in1_cb, in1_block_num_tiles);
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_push_back(intermediate_cb, out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

#ifdef RSCATTER
            // ---- Ring REDUCE-SCATTER. intermediate_cb (FP32, resident) is my matmul partial for the whole
            // sub-block, row-major, partitioned into P=Pk contiguous chunks whose sizes differ by at most one
            // tile (the first rs_T%P chunks take one extra), so any rs_T >= P works. Seed cb_send with MY OWN
            // chunk `rs_ring_pos`; then over P-1 rounds receive the running sum for chunk
            // d = (rs_ring_pos - t - 1) mod P, add my resident chunk d, and either forward it (earlier rounds)
            // or keep it (last round) - at which point it is fully reduced and is exactly the chunk this core
            // owns, so it goes to out_cb for the writer to send to DRAM.
            // Every CB operation moves a full MAX-size slot while only the chunk's useful prefix is written or
            // read; that keeps the send/recv FIFOs in lockstep with the writer's constant-stride remote writes. ----
            {
                const uint32_t P = rs_P;
                const uint32_t cbase = rs_T / P;  // floor chunk size; first crem chunks carry one more
                const uint32_t crem = rs_T - cbase * P;
                const uint32_t max_chunk = cbase + (crem ? 1u : 0u);
                auto csize = [=](uint32_t c) { return cbase + (c < crem ? 1u : 0u); };
                auto coff = [=](uint32_t c) { return c * cbase + (c < crem ? c : crem); };
                constexpr uint32_t cb_send_cb = tt::CBIndex::c_8;     // NOT c_4/c_5: those are fusion operands
                constexpr uint32_t cb_recv_cb = tt::CBIndex::c_9;     // incoming slice (bf16), 2 slots
                cb_wait_front(intermediate_cb, out_block_num_tiles);  // resident; popped after the ring
                cb_reserve_back(cb_send_cb, max_chunk);
                rs_copy_chunk(intermediate_cb, coff(rs_ring_pos), cb_send_cb, csize(rs_ring_pos));
                cb_push_back(cb_send_cb, max_chunk);
                for (uint32_t t = 0; t + 1u < P; ++t) {
                    const uint32_t d = (rs_ring_pos + P - t - 1u) % P;  // chunk reduced this round
                    const uint32_t dn = csize(d);
                    cb_wait_front(cb_recv_cb, max_chunk);
                    if (t + 1u < P - 1u) {  // forward the running sum into the next round's send slot
                        cb_reserve_back(cb_send_cb, max_chunk);
                        rs_add_chunk(intermediate_cb, coff(d), cb_recv_cb, cb_send_cb, dn);
                        cb_push_back(cb_send_cb, max_chunk);
                    } else {  // last round: fully reduced AND owned by this core
#ifdef SMALL_M_FUSED
                        // Land the reduced slice in the scratch CB, then apply the epilogue ONCE to it. Every
                        // owner does this for its own slice, so the epilogue is applied exactly once per output
                        // tile -- the property the single-root chain gets for free.
                        constexpr uint32_t cb_epi_cb = tt::CBIndex::c_10;
                        cb_reserve_back(cb_epi_cb, max_chunk);
                        rs_add_chunk(intermediate_cb, coff(d), cb_recv_cb, cb_epi_cb, dn);
                        cb_push_back(cb_epi_cb, max_chunk);
                        cb_reserve_back(out_cb, max_chunk);
                        rs_epilogue_slice(
                            cb_epi_cb,
                            in2_cb,
                            ternary_a_cb,
                            ternary_b_cb,
                            fused_ternary_scalar_uint,
                            out_cb,
                            dn,
                            max_chunk,
                            broadcast_ternary_b);
                        cb_push_back(out_cb, max_chunk);
#else
                        cb_reserve_back(out_cb, max_chunk);
                        rs_add_chunk(intermediate_cb, coff(d), cb_recv_cb, out_cb, dn);
                        cb_push_back(out_cb, max_chunk);
#endif
                    }
                    cb_pop_front(cb_recv_cb, max_chunk);
                }
                cb_pop_front(intermediate_cb, out_block_num_tiles);
            }
#else
            cb_reserve_back(out_cb, out_block_num_tiles);
            // Split-K plan B column reduction: bottom band emits its own matmul partial; every other band
            // adds the running sum forwarded up from the band below. The DM then either forwards out_cb up
            // (non-top bands) or writes it to DRAM (top band).
            cb_wait_front(intermediate_cb, out_block_num_tiles);
#ifndef SMALL_M_FUSED
            // NO-FUSION path (byte-identical to the pre-fusion output stage): top and non-top reduce
            // bands are identical; the writer decides forward-up vs DRAM-write.
            if (is_reduce_bottom) {
                copy_block_to_cb(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            } else {
                cb_wait_front(cb_reduce, out_block_num_tiles);
                reduce_add_block(intermediate_cb, cb_reduce, out_cb, M_block_tiles, N_block_tiles);
                cb_pop_front(cb_reduce, out_block_num_tiles);
            }
            cb_pop_front(intermediate_cb, out_block_num_tiles);
#else
            // FUSION-AWARE split-K: bias/activation/addcmul are applied EXACTLY ONCE at the reduction ROOT
            // (is_reduce_top). Non-root bands forward the RAW partial (no fusion), so the reduction chain sums
            // un-fused partials and the epilogue sees the true A@B (+ reduced K) before bias/act/addcmul.
            if (!is_reduce_top) {
                // Partial-forwarding band. RAW (copy_block_raw / reduce_add_block never apply activation).
                if (is_reduce_bottom) {
                    copy_block_raw(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
                } else {
                    cb_wait_front(cb_reduce, out_block_num_tiles);
                    reduce_add_block(intermediate_cb, cb_reduce, out_cb, M_block_tiles, N_block_tiles);
                    cb_pop_front(cb_reduce, out_block_num_tiles);
                }
                cb_pop_front(intermediate_cb, out_block_num_tiles);
            } else {
                // Reduction ROOT (Pk==1: every core; Pk>1: top band). Sum the forwarded partial into
                // intermediate (in place) when this is a Pk>1 chain, then apply the epilogue ONCE -> out_cb.
                if (!is_reduce_bottom) {
                    cb_wait_front(cb_reduce, out_block_num_tiles);
                    reduce_add_in_place(intermediate_cb, cb_reduce, M_block_tiles, N_block_tiles);
                    cb_pop_front(cb_reduce, out_block_num_tiles);
                }
#if defined(FUSE_TERNARY)
                add_bias_and_addcmul_block(
                    intermediate_cb,
                    in2_cb,
                    ternary_a_cb,
                    ternary_b_cb,
                    fused_ternary_scalar_uint,
                    out_cb,
                    M_block_tiles,
                    N_block_tiles,
                    broadcast_ternary_b);  // pops intermediate_cb internally
#elif defined(FUSE_BIAS)
                cb_wait_front(in2_cb, N_block_tiles);
                add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);  // + activation
                cb_pop_front(in2_cb, N_block_tiles);
                cb_pop_front(intermediate_cb, out_block_num_tiles);
#else   // activation-only root
                copy_block_to_cb(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);  // applies SFPU activation
                cb_pop_front(intermediate_cb, out_block_num_tiles);
#endif  // fusion kind
            }
#endif  // no-fusion chain vs fused
#endif  // RSCATTER vs chain
        }
    }
    cb_pop_front(in0_cb, K_num_blocks * in0_block_num_tiles);
}
