// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/subchunk_bands.hpp"

#ifndef IN0_SUB_CHUNKS
#define IN0_SUB_CHUNKS 1
#endif

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_out(out_cb);
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        cb_out.push_back(N_block_tiles);
    }
}

#ifdef SPLIT_OUTPUT_WRITE
// Percent of the block's M-rows routed to NOC_1 (dm_in1 / out_cb_a); the rest go to NOC_0 (dm_in0 /
// out_cb_b). Must match dm_in0/dm_in1 so the c_2/c_8 counts stay in lockstep.
#ifndef AG_SPLIT_NOC1_PCT
#define AG_SPLIT_NOC1_PCT 50
#endif
// Two-NoC output-write split: pack the block's first split_rows M-rows into out_cb_a (drained by the NOC_1
// writer dm_in1) and the rest into out_cb_b (drained by the NOC_0 writer dm_in0). Same per-row pack as
// copy_block; only the destination CB switches at the row boundary. Both CBs share the output format.
void copy_block_split(
    uint32_t in_cb,
    uint32_t out_cb_a,
    uint32_t out_cb_b,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t split_rows) {
    CircularBuffer cb_out_a(out_cb_a);
    CircularBuffer cb_out_b(out_cb_b);
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb_a);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        const uint32_t out_cb = (m < split_rows) ? out_cb_a : out_cb_b;
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        if (m < split_rows) {
            cb_out_a.push_back(N_block_tiles);
        } else {
            cb_out_b.push_back(N_block_tiles);
        }
    }
}
#endif  // SPLIT_OUTPUT_WRITE

#ifdef FUSE_SWIGLU
// Fused SwiGLU output stage. The matmul produced an interleaved gate/up block in
// `in_cb` (the intermediate accumulator): within each M row, column tile 2p is the
// gate projection and 2p+1 is the up projection (the weight was tile-pair interleaved
// on the host). For each pair we emit one output tile = silu(gate) * up, so the block
// shrinks from N_block_tiles to N_block_tiles/2 along N. No extra CB / no extra DRAM
// round-trip: silu runs on the gate DST reg and the multiply is an SFPU dst*dst op.
//
// With FUSE_BIAS: bias is interleaved identically (tile 2p = gate bias, 2p+1 = up bias)
// and added via row-broadcast before silu/mul: out = silu(gate + bias_gate) * (up + bias_up).
//
// N_block_tiles must be even (enforced host-side).
void swiglu_block(uint32_t in_cb, uint32_t bias_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
#ifdef FUSE_BIAS
    reconfig_data_format(in_cb, bias_cb);
#else
    reconfig_data_format_srca(in_cb);
#endif
    pack_reconfig_data_format(out_cb);

    constexpr uint32_t GATE_DST = 0;
    constexpr uint32_t UP_DST = 1;
    const uint32_t out_N_block_tiles = N_block_tiles >> 1;

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        const uint32_t row_base = m * N_block_tiles;
        for (uint32_t p = 0; p < out_N_block_tiles; p++) {
            const uint32_t gate_n = p << 1;
            const uint32_t up_n = gate_n + 1;
            const uint32_t gate_tile_id = row_base + gate_n;
            const uint32_t up_tile_id = gate_tile_id + 1;

            tile_regs_acquire();
#ifdef FUSE_BIAS
            add_bcast_rows_init_short(in_cb, bias_cb);
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, gate_tile_id, gate_n, GATE_DST);
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, up_tile_id, up_n, UP_DST);
#else
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, gate_tile_id, GATE_DST);
            copy_tile(in_cb, up_tile_id, UP_DST);
#endif
            silu_tile_init();
            silu_tile(GATE_DST);
            mul_binary_tile_init();
            mul_binary_tile(GATE_DST, UP_DST, GATE_DST);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(GATE_DST, out_cb);
            tile_regs_release();
        }
        cb_push_back(out_cb, out_N_block_tiles);
    }
}
#endif  // FUSE_SWIGLU

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
    CircularBuffer cb_out(out_cb);
    add_bcast_rows_init_short(in_cb, bias_cb);
    reconfig_data_format(in_cb, bias_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, tile_id, n, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        cb_out.push_back(N_block_tiles);
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

    CircularBuffer cb_intermediate(intermediate_cb);
    CircularBuffer cb_bias(bias_cb);
    CircularBuffer cb_ternary_a(ternary_a_cb);
    CircularBuffer cb_ternary_b(ternary_b_cb);
    CircularBuffer cb_out(out_cb);

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
    cb_bias.wait_front(N_block_tiles);

    // Unpacker waits for intermediate_cb to be ready
    cb_intermediate.wait_front(out_block_num_tiles);

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
    // cb_wait_front(intermediate_cb, out_block_num_tiles); // Unpacker-Packer sync
    // cb_pop_front(intermediate_cb, out_block_num_tiles);
    cb_bias.pop_front(N_block_tiles);

    cb_intermediate.pop_front(out_block_num_tiles);

    // Restore intermediate_cb to ready (+ sync packer/unpacker)
    cb_intermediate.reserve_back(out_block_num_tiles);
    cb_intermediate.push_back(out_block_num_tiles);
#endif  // FUSE_BIAS

    // ============================================
    // STEP 2: Multiply by ternary_b and scalar
    // Read from intermediate_cb and write back to intermediate_cb
    // broadcast_ternary_b: 1 = single row broadcast, 0 = row-by-row streaming
    // ============================================

    cb_intermediate.wait_front(out_block_num_tiles);

    uint32_t tile_id = 0;

    if (broadcast_ternary_b) {
        // === BROADCAST: single row, wait/pop once ===
        cb_ternary_b.wait_front(N_block_tiles);

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

        cb_ternary_b.pop_front(N_block_tiles);
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
            cb_ternary_b.wait_front(N_block_tiles);
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
            cb_ternary_b.pop_front(N_block_tiles);
        }
    }

    cb_intermediate.pop_front(out_block_num_tiles);

    // 'refill' intermediate_cb (also synchronize packer/unpacker)
    cb_intermediate.reserve_back(out_block_num_tiles);
    cb_intermediate.push_back(out_block_num_tiles);

    cb_intermediate.wait_front(out_block_num_tiles);

    add_tiles_init(intermediate_cb, ternary_a_cb);
    reconfig_data_format(intermediate_cb, ternary_a_cb);
    pack_reconfig_data_format(out_cb);

    tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of ternary_a tiles
        cb_ternary_a.wait_front(N_block_tiles);

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

        cb_ternary_a.pop_front(N_block_tiles);
        cb_out.push_back(N_block_tiles);
    }

    cb_intermediate.pop_front(out_block_num_tiles);
}

// Slightly modified from compute_common.hpp
//
// m_out_tile_offset shifts where this call's rows land in the (full) output block. When in0 is
// delivered as M-row sub-chunk bands, each band's in0 CB slot is 0-based (M_block_tiles == band_h,
// in0 indexing starts at 0) but the band's results must accumulate into rows [band_lo, band_lo+band_h)
// of the intermediate block, so pass m_out_tile_offset = band_lo. Whole-block callers pass 0.
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
    const uint32_t m_out_tile_offset = 0) {
    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        // The in0 CB slot is padded to a uniform (M_block_tiles / nb) rows, so a ragged band whose
        // height is not a multiple of subblock_h still has real tiles for a full subblock_h read -- the
        // trailing rows are slack. Compute the full subblock (a smaller rt_dim would need an
        // mm_block_init_short re-init that stalls the engine), then pack only the real rows so a ragged
        // band never writes into the next band's output rows. subblock_h | (M_block_tiles/nb) (host
        // guard) keeps the deep read inside the slot.
        const uint32_t real_h = std::min(subblock_h, M_block_tiles - M_start);
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
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < real_h; h++) {
                uint32_t h_tile_id = M_start + h + m_out_tile_offset;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
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
    // Number of leading (self/local) k-block positions this device owns. The AG schedule always drains
    // the local slice first, so positions [0, num_local_k_blocks) are local (never sub-chunked) and the
    // rest are remote (sub-chunked into IN0_SUB_CHUNKS M-row bands on the first N-block). Kept identical
    // across the dm_in0/dm_in1 senders so per-band CB push/pop counts match.
    [[maybe_unused]] constexpr uint32_t num_local_k_blocks = get_compile_time_arg_val(8);

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);

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

    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
#ifdef SPLIT_OUTPUT_WRITE
    // Second output CB for the two-NoC write split; drained by dm_in0 on NOC_0.
    constexpr uint32_t out_cb_b = OUT_CB_B;
    CircularBuffer cb_out_b(out_cb_b);
#endif
    CircularBuffer cb_intermediate(intermediate_cb);
    CircularBuffer cb_in2(in2_cb);

    // compute_kernel_hw_startup must be the first compute API call (before SFPU/op inits).
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

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
            cb_intermediate.reserve_back(out_block_num_tiles);
            // Sub-chunked (banded) matmul. Remote k-blocks on the first N-block arrive as IN0_SUB_CHUNKS
            // M-row bands; local k-blocks and every k-block on later N-blocks arrive whole (nb == 1). in1
            // is re-presented once per band (read once, pushed nb times by the in1 sender), so in0 and in1
            // advance in lockstep and the matmul fires per band. Each k-block position is delivered fresh
            // (no N-stride in0 reuse). With IN0_SUB_CHUNKS == 1 this degenerates to one whole band per
            // position.
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                const uint32_t nb =
                    (n_block_iter == 0 && k_block >= num_local_k_blocks) ? (uint32_t)IN0_SUB_CHUNKS : 1u;
#ifdef AG_INTERLEAVE_BANDS
                // Interleave a forward remote k-block with the following backward one at band granularity:
                // A.b0, B.b0, A.b1, B.b1, ... Both members share band_lo and accumulate into the same
                // output rows; both are past k-block 0 so L1 accumulation is already on, so the two partials
                // sum correctly regardless of order. The position-based pairing matches dm_in0/dm_in1 so the
                // per-band in0/in1 CB counts stay in lockstep.
                if (n_block_iter == 0 && nb > 1 && k_block >= num_local_k_blocks && (k_block + 1) < K_num_blocks &&
                    ((k_block - num_local_k_blocks) & 1u) == 0) {
                    for (uint32_t band = 0; band < nb; band++) {
                        uint32_t band_lo, band_h;
                        balanced_band(current_M_block_tiles, nb, band, band_lo, band_h);
                        if (band_h == 0) {
                            break;
                        }
                        // Reserve a uniform slot of M_block_tiles/nb rows for every band (not the ragged
                        // band_h) so each band tiles the in0 CB exactly and no reservation wraps
                        // fifo_limit mid-block. matmul reads band_h rows and packs the real ones.
                        const uint32_t band_slot_tiles = (M_block_tiles / nb) * K_block_tiles;
                        for (uint32_t member = 0; member < 2; member++) {
                            cb_in0.wait_front(band_slot_tiles);
                            cb_in1.wait_front(in1_block_num_tiles);
                            matmul_blocks(
                                in0_cb,
                                in1_cb,
                                intermediate_cb,
                                band_h,
                                current_N_block_tiles,
                                N_block_tiles,
                                K_block_tiles,
                                current_subblock_h,
                                current_subblock_w,
                                band_lo);
                            cb_in0.pop_front(band_slot_tiles);
                            cb_in1.pop_front(in1_block_num_tiles);
                        }
                    }
                    k_block++;  // the backward member of the pair is consumed here too
                    continue;
                }
#endif
                for (uint32_t band = 0; band < nb; band++) {
                    uint32_t band_lo, band_h;
                    balanced_band(current_M_block_tiles, nb, band, band_lo, band_h);
                    if (band_h == 0) {
                        break;  // only when nb > current_M_block_tiles
                    }
                    // Reserve a uniform slot of M_block_tiles/nb rows for every band (not the ragged
                    // band_h) so each band tiles the in0 CB exactly and no reservation wraps fifo_limit
                    // mid-block. matmul reads band_h rows and packs the real ones.
                    const uint32_t band_slot_tiles = (M_block_tiles / nb) * K_block_tiles;
                    cb_in0.wait_front(band_slot_tiles);
                    cb_in1.wait_front(in1_block_num_tiles);

                    matmul_blocks(
                        in0_cb,
                        in1_cb,
                        intermediate_cb,
                        band_h,
                        current_N_block_tiles,
                        N_block_tiles,
                        K_block_tiles,
                        current_subblock_h,
                        current_subblock_w,
                        band_lo);

                    cb_in0.pop_front(band_slot_tiles);
                    cb_in1.pop_front(in1_block_num_tiles);
                }
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_intermediate.push_back(out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

#ifdef FUSE_SWIGLU
            // SwiGLU collapses the interleaved gate/up block to half its N width.
            cb_out.reserve_back(out_block_num_tiles >> 1);
            cb_intermediate.wait_front(out_block_num_tiles);
#ifdef FUSE_BIAS
            cb_in2.wait_front(N_block_tiles);
#endif
            swiglu_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
#ifdef FUSE_BIAS
            cb_in2.pop_front(N_block_tiles);
#endif
            cb_intermediate.pop_front(out_block_num_tiles);

#elif !defined(FUSE_TERNARY)
#if defined(SPLIT_OUTPUT_WRITE) && !defined(FUSE_BIAS)
            // Two-NoC split: pack low rows -> c_2 (dm_in1/NOC_1), high rows -> c_8 (dm_in0/NOC_0).
            constexpr uint32_t split_rows = (M_block_tiles * AG_SPLIT_NOC1_PCT) / 100;
            if (split_rows) {
                cb_out.reserve_back(split_rows * N_block_tiles);
            }
            if (split_rows < M_block_tiles) {
                cb_out_b.reserve_back((M_block_tiles - split_rows) * N_block_tiles);
            }
            cb_intermediate.wait_front(out_block_num_tiles);
            copy_block_split(intermediate_cb, out_cb, out_cb_b, M_block_tiles, N_block_tiles, split_rows);
            cb_intermediate.pop_front(out_block_num_tiles);
#else
            cb_out.reserve_back(out_block_num_tiles);
            cb_intermediate.wait_front(out_block_num_tiles);
#ifndef FUSE_BIAS
            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
#else
            cb_in2.wait_front(N_block_tiles);
            add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_in2.pop_front(N_block_tiles);
#endif  // FUSE_BIAS
            cb_intermediate.pop_front(out_block_num_tiles);
#endif  // SPLIT_OUTPUT_WRITE

#else   // FUSE_TERNARY is set
            cb_out.reserve_back(out_block_num_tiles);
            add_bias_and_addcmul_block(
                intermediate_cb,
                in2_cb,
                ternary_a_cb,
                ternary_b_cb,
                fused_ternary_scalar_uint,
                out_cb,
                M_block_tiles,
                N_block_tiles,
                broadcast_ternary_b);
#endif  // FUSE_TERNARY
        }
    }
}
