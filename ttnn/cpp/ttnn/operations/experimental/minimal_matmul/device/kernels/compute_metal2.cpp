// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of compute.cpp. Bound by MinimalMatmulDeviceOperation::ProgramFactory; the legacy
// original beside it still serves the fused-CCL emitter (minimal_matmul_factory_helper_common).

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
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dfb_binding_token.h"
#include "experimental/kernel_args.h"

void copy_and_pack_block(
    DFBBindingToken in_dfb, DFBBindingToken out_dfb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    DataflowBuffer dfb_out(out_dfb);
    reconfig_data_format_srca(in_dfb);
    pack_reconfig_data_format(out_dfb);
    copy_init(in_dfb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(in_dfb, tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_dfb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        dfb_out.push_back(N_block_tiles);
    }
}

#ifdef FUSE_SWIGLU
// Fused SwiGLU output stage. The matmul produced an interleaved gate/up block in
// `in_dfb` (the intermediate accumulator): within each M row, column tile 2p is the
// gate projection and 2p+1 is the up projection (the weight was tile-pair interleaved
// on the host). For each pair we emit one output tile = silu(gate) * up, so the block
// shrinks from N_block_tiles to N_block_tiles/2 along N. No extra DFB / no extra DRAM
// round-trip: silu runs on the gate DST reg and the multiply is an SFPU dst*dst op.
//
// With FUSE_BIAS: bias is interleaved identically (tile 2p = gate bias, 2p+1 = up bias)
// and added via row-broadcast before silu/mul: out = silu(gate + bias_gate) * (up + bias_up).
//
// N_block_tiles must be even (enforced host-side).
void swiglu_block(
    DFBBindingToken in_dfb,
#ifdef FUSE_BIAS
    DFBBindingToken bias_dfb,
#endif
    DFBBindingToken out_dfb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles) {
    DataflowBuffer dfb_out(out_dfb);
#ifdef FUSE_BIAS
    reconfig_data_format(in_dfb, bias_dfb);
#else
    reconfig_data_format_srca(in_dfb);
#endif
    pack_reconfig_data_format(out_dfb);

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
            add_bcast_rows_init(in_dfb, bias_dfb);
            add_tiles_bcast<BroadcastType::ROW>(in_dfb, bias_dfb, gate_tile_id, gate_n, GATE_DST);
            add_tiles_bcast<BroadcastType::ROW>(in_dfb, bias_dfb, up_tile_id, up_n, UP_DST);
#else
            copy_init(in_dfb);
            copy_tile(in_dfb, gate_tile_id, GATE_DST);
            copy_tile(in_dfb, up_tile_id, UP_DST);
#endif
            silu_tile_init();
            silu_tile(GATE_DST);
            mul_binary_tile_init();
            mul_binary_tile(GATE_DST, UP_DST, GATE_DST);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(GATE_DST, out_dfb);
            tile_regs_release();
        }
        dfb_out.push_back(out_N_block_tiles);
    }
}
#endif  // FUSE_SWIGLU

// For caller: if FUSE_TERNARY defined then out_dfb == intermediate_dfb
/**
 * Add bias to input block
 * Performs: output = input + bias (row broadcast)
 *
 * stream_output:
 *   - true: Pushes tiles one row at a time (for intermediate output to next stage)
 *   - false: Pushes all tiles at end (for final output)
 */
void add_bias_block(
    DFBBindingToken in_dfb,
    DFBBindingToken bias_dfb,
    DFBBindingToken out_dfb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles) {
    DataflowBuffer dfb_out(out_dfb);
    reconfig_data_format(in_dfb, bias_dfb);
    pack_reconfig_data_format(out_dfb);
    add_bcast_rows_init(in_dfb, bias_dfb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            add_tiles_bcast<BroadcastType::ROW>(in_dfb, bias_dfb, tile_id, n, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_dfb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        dfb_out.push_back(N_block_tiles);
    }
}

void add_bias_and_addcmul_block(
    DFBBindingToken intermediate_dfb,
#ifdef FUSE_BIAS
    DFBBindingToken bias_dfb,
#endif
    DFBBindingToken ternary_a_dfb,
    DFBBindingToken ternary_b_dfb,
    uint32_t scalar_value,
    DFBBindingToken out_dfb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t broadcast_ternary_b) {
    // Note: unary_bcast_tile does not work with fp32_acc_to_dest=True.
    // As a workaround, we perform addcmul through multiple LLKs calls (mul_tiles, mul_unary_tile, add_tiles_bcast).

    const uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    DataflowBuffer dfb_intermediate(intermediate_dfb);
#ifdef FUSE_BIAS
    DataflowBuffer dfb_bias(bias_dfb);
#endif
    DataflowBuffer dfb_ternary_a(ternary_a_dfb);
    DataflowBuffer dfb_ternary_b(ternary_b_dfb);
    DataflowBuffer dfb_out(out_dfb);

    constexpr uint32_t DST_ID = 0;
#ifdef FUSE_BIAS
    // ============================================
    // STEP 1: Add bias block
    // Read from intermediate_dfb and write back to intermediate_dfb
    // ============================================

    reconfig_data_format(intermediate_dfb, bias_dfb);
    pack_reconfig_data_format(intermediate_dfb);
    add_bcast_rows_init(intermediate_dfb, bias_dfb);

    // Wait for ALL input data ONCE at the beginning
    dfb_bias.wait_front(N_block_tiles);

    // Unpacker waits for intermediate_dfb to be ready
    dfb_intermediate.wait_front(out_block_num_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            uint32_t tile_id = m * N_block_tiles + n;

            tile_regs_acquire();
            add_tiles_bcast<BroadcastType::ROW>(intermediate_dfb, bias_dfb, tile_id, n, DST_ID);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_ID, intermediate_dfb);
            tile_regs_release();
        }
    }

    // Pop input and push output ONCE at the end
    dfb_bias.pop_front(N_block_tiles);

    dfb_intermediate.pop_front(out_block_num_tiles);

    // Restore intermediate_dfb to ready (+ sync packer/unpacker)
    dfb_intermediate.reserve_back(out_block_num_tiles);
    dfb_intermediate.push_back(out_block_num_tiles);
#endif  // FUSE_BIAS

    // ============================================
    // STEP 2: Multiply by ternary_b and scalar
    // Read from intermediate_dfb and write back to intermediate_dfb
    // broadcast_ternary_b: 1 = single row broadcast, 0 = row-by-row streaming
    // ============================================

    dfb_intermediate.wait_front(out_block_num_tiles);

    uint32_t tile_id = 0;

    if (broadcast_ternary_b) {
        // === BROADCAST: single row, wait/pop once ===
        dfb_ternary_b.wait_front(N_block_tiles);

        reconfig_data_format(intermediate_dfb, ternary_b_dfb);
        pack_reconfig_data_format(intermediate_dfb);
#ifndef TERNARY_B_IS_FLOAT32
        mul_bcast_rows_init(intermediate_dfb, ternary_b_dfb);
#else
        // Full re-arm (hw_configure + pack_dest/math_pack_sync), matching the pre-cleanup
        // 2-arg unary_bcast_init(ternary_b_dfb, intermediate_dfb); this runs after matmul_blocks
        // regardless of FUSE_BIAS, so a plain reconfig would drop the MATH<->PACK DST re-arm.
        // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the
        // pre-cleanup full-init behaviour) should become a targeted DST re-arm.
        compute_kernel_hw_startup(ternary_b_dfb, intermediate_dfb);
        unary_bcast_init<BroadcastType::ROW>(ternary_b_dfb);
#endif  // TERNARY_B_IS_FLOAT32

        binop_with_scalar_tile_init();

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                mul_tiles_bcast<BroadcastType::ROW>(intermediate_dfb, ternary_b_dfb, tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the
                // pre-cleanup full-init behaviour) should become a targeted DST re-arm.
                compute_kernel_hw_startup(ternary_b_dfb, intermediate_dfb);
                unary_bcast_init<BroadcastType::ROW>(ternary_b_dfb);
                unary_bcast<BroadcastType::ROW>(ternary_b_dfb, n, TERNARY_B_DST_ID);

                reconfig_data_format_srca(intermediate_dfb);
                copy_init(intermediate_dfb);
                copy_tile(intermediate_dfb, tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_dfb);
                tile_regs_release();
                tile_id++;
            }
        }

        dfb_ternary_b.pop_front(N_block_tiles);
    } else {
        // === NO BROADCAST: row-by-row, wait/pop per M row ===
        reconfig_data_format(intermediate_dfb, ternary_b_dfb);
        pack_reconfig_data_format(intermediate_dfb);
#ifndef TERNARY_B_IS_FLOAT32
        mul_init(intermediate_dfb, ternary_b_dfb);
#endif
        binop_with_scalar_tile_init();

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            dfb_ternary_b.wait_front(N_block_tiles);
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                mul_tiles(intermediate_dfb, ternary_b_dfb, tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                reconfig_data_format_srca(ternary_b_dfb);
                copy_init(ternary_b_dfb);
                copy_tile(ternary_b_dfb, n, TERNARY_B_DST_ID);

                reconfig_data_format_srca(intermediate_dfb);
                copy_init(intermediate_dfb);
                copy_tile(intermediate_dfb, tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_dfb);
                tile_regs_release();
                tile_id++;
            }
            dfb_ternary_b.pop_front(N_block_tiles);
        }
    }

    dfb_intermediate.pop_front(out_block_num_tiles);

    // 'refill' intermediate_dfb (also synchronize packer/unpacker)
    dfb_intermediate.reserve_back(out_block_num_tiles);
    dfb_intermediate.push_back(out_block_num_tiles);

    dfb_intermediate.wait_front(out_block_num_tiles);

    reconfig_data_format(intermediate_dfb, ternary_a_dfb);
    pack_reconfig_data_format(out_dfb);
    add_init(intermediate_dfb, ternary_a_dfb);

    tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of ternary_a tiles
        dfb_ternary_a.wait_front(N_block_tiles);

        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();

            // ternary_a_dfb is pushed one row at a time, so use column index n
            add_tiles(intermediate_dfb, ternary_a_dfb, tile_id, n, DST_ID);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_ID, out_dfb);
            tile_regs_release();
            tile_id++;
        }

        dfb_ternary_a.pop_front(N_block_tiles);
        dfb_out.push_back(N_block_tiles);
    }

    dfb_intermediate.pop_front(out_block_num_tiles);
}

// Slightly modified from compute_common.hpp
void matmul_blocks(
    const DFBBindingToken in0_dfb,
    const DFBBindingToken in1_dfb,
    const DFBBindingToken out_dfb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
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
                    in0_dfb,
                    in1_dfb,
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
                    pack_tile<true>(write_dst_index, out_dfb, out_tile_id);
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
    constexpr auto K_num_blocks = get_arg(args::K_num_blocks);
    constexpr auto M_block_tiles = get_arg(args::M_block_tiles);
    constexpr auto K_block_tiles = get_arg(args::K_block_tiles);
    constexpr auto N_block_tiles = get_arg(args::N_block_tiles);
    constexpr auto M_blocks_per_core = get_arg(args::M_blocks_per_core);
    constexpr auto N_blocks_per_core = get_arg(args::N_blocks_per_core);
    constexpr auto subblock_h = get_arg(args::subblock_h);
    constexpr auto subblock_w = get_arg(args::subblock_w);

    const auto M_start_tile = get_arg(args::M_start_tile);
    const auto M_end_tile = get_arg(args::M_end_tile);
    const auto N_start_tile = get_arg(args::N_start_tile);
    const auto N_end_tile = get_arg(args::N_end_tile);

#ifdef FUSE_TERNARY
    const auto fused_ternary_scalar_uint = get_arg(args::fused_ternary_scalar);
    const auto broadcast_ternary_b = get_arg(args::broadcast_ternary_b);
#else
    // Default value when ternary is not fused (not used, helps compiler optimize)
    constexpr uint32_t fused_ternary_scalar_uint = 0;
    constexpr uint32_t broadcast_ternary_b = 1;
#endif

    // in0 / in1 / out / intermediate are bound on every compute instance; in2 and the ternary pair
    // only when the host bound them, so their tokens are gated on the matching define.
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_intermediate(dfb::intermediate);
#ifdef FUSE_BIAS
    DataflowBuffer dfb_in2(dfb::in2);
#endif

    // compute_kernel_hw_startup must be the first compute API call (before SFPU/op inits).
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::intermediate);

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    matmul_init(dfb::in0, dfb::in1);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t M_num_subblocks = M_block_tiles / subblock_h;
    constexpr uint32_t N_num_subblocks = N_block_tiles / subblock_w;

    bool reuse_in0_block = false;

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

            // Reconfig before init: on all but the first block the unpackers are still
            // configured for the previous output stage's operands (see #55052).
            reconfig_data_format(dfb::in1, dfb::in0);
            pack_reconfig_data_format(dfb::intermediate);
            matmul_block_init(
                dfb::in0,
                dfb::in1,
                false /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            // Accumulation buffer
            dfb_intermediate.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                dfb_in0.wait_front(in0_block_num_tiles);
                dfb_in1.wait_front(in1_block_num_tiles);

                matmul_blocks(
                    dfb::in0,
                    dfb::in1,
                    dfb::intermediate,
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w);

                if (k_block == K_num_blocks - 1) {
                    /**
                     * On next iteration we might get reuse on in0
                     *
                     */
                    if (n_block_iter < N_blocks_per_core - 1) {
                        // going to stride on N, so reuse in0
                        reuse_in0_block = true;
                    }
                }
                if (!reuse_in0_block) {
                    dfb_in0.pop_front(in0_block_num_tiles);
                }
                dfb_in1.pop_front(in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    pack_reconfig_l1_acc(1);
                }
            }

            dfb_intermediate.push_back(out_block_num_tiles);
            pack_reconfig_l1_acc(0);

#ifdef FUSE_SWIGLU
            // SwiGLU collapses the interleaved gate/up block to half its N width.
            dfb_out.reserve_back(out_block_num_tiles >> 1);
            dfb_intermediate.wait_front(out_block_num_tiles);
#ifdef FUSE_BIAS
            dfb_in2.wait_front(N_block_tiles);
#endif
            swiglu_block(
                dfb::intermediate,
#ifdef FUSE_BIAS
                dfb::in2,
#endif
                dfb::out,
                M_block_tiles,
                N_block_tiles);
#ifdef FUSE_BIAS
            dfb_in2.pop_front(N_block_tiles);
#endif
            dfb_intermediate.pop_front(out_block_num_tiles);

#elif !defined(FUSE_TERNARY)
            dfb_out.reserve_back(out_block_num_tiles);
            dfb_intermediate.wait_front(out_block_num_tiles);
#ifndef FUSE_BIAS
            copy_and_pack_block(dfb::intermediate, dfb::out, M_block_tiles, N_block_tiles);
#else
            dfb_in2.wait_front(N_block_tiles);
            add_bias_block(dfb::intermediate, dfb::in2, dfb::out, M_block_tiles, N_block_tiles);
            dfb_in2.pop_front(N_block_tiles);
#endif  // FUSE_BIAS
            dfb_intermediate.pop_front(out_block_num_tiles);

#else  // FUSE_TERNARY is set
            dfb_out.reserve_back(out_block_num_tiles);
            add_bias_and_addcmul_block(
                dfb::intermediate,
#ifdef FUSE_BIAS
                dfb::in2,
#endif
                dfb::ternary_a,
                dfb::ternary_b,
                fused_ternary_scalar_uint,
                dfb::out,
                M_block_tiles,
                N_block_tiles,
                broadcast_ternary_b);
#endif  // FUSE_TERNARY
        }
    }
}
