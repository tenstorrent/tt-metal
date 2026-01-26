// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
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

/*
 * Fused bias and addcmul block
 *
 * This functions computes the following operation:
 * If FUSE_BIAS and FUSE_TERNARY are enabled then:
 *   output = ternary_a + scalar * (matmul_result + bias) * ternary_b
 * If FUSE_BIAS is enabled and FUSE_TERNARY is disabled then:
 *   output = matmul_result + bias
 * If FUSE_BIAS is disabled and FUSE_TERNARY is enabled then:
 *    output = ternary_a + scalar * matmul_result * ternary_b
 *
 * Note: One of FUSE_BIAS or FUSE_TERNARY must be enabled.
 *
 * in_cb: input buffer (matmul result)
 * bias_cb: bias buffer
 * ternary_a_cb: ternary a buffer
 * ternary_b_cb: ternary b buffer
 * scalar_value: scalar multiplier
 * out_cb: output buffer
 * M_block_tiles: number of M block tiles
 * N_block_tiles: number of N block tiles
 */
void add_bias_and_addcmul_block(
    uint32_t in_cb,
    uint32_t bias_cb,
    uint32_t ternary_a_cb,
    uint32_t ternary_b_cb,
    uint32_t scalar_value,
    uint32_t out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles) {
    // Initialize based on whether bias is fused
#ifdef FUSE_BIAS
    add_bcast_rows_init_short(in_cb, bias_cb);
    reconfig_data_format(in_cb, bias_cb);
#else
    unary_op_init_common(in_cb, out_cb);
    copy_tile_to_dst_init_short(in_cb);
#endif
    pack_reconfig_data_format(out_cb);
    constexpr uint32_t fused_act_dst_id = 0;

    constexpr uint32_t ternary_a_dst_id = 1;
    constexpr uint32_t ternary_b_dst_id = 2;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();

#ifdef FUSE_BIAS
#ifdef FUSE_TERNARY
            // If fused ternary is enabled then we have to reconfigure unpacker every iteration
            add_bcast_rows_init_short(in_cb, bias_cb);
            reconfig_data_format(in_cb, bias_cb);
#endif  // FUSE_TERNARY
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, tile_id, n, fused_act_dst_id /*dst*/);
#else
            reconfig_data_format_srca(in_cb);
            copy_tile_init(in_cb);
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
#endif

#ifdef FUSE_TERNARY
            // Read ternary tensors into destination registers
            reconfig_data_format_srca(ternary_a_cb);
            copy_tile_init(ternary_a_cb);
            copy_tile(ternary_a_cb, tile_id, ternary_a_dst_id);

            reconfig_data_format_srca(ternary_b_cb);
            copy_tile_init(ternary_b_cb);
            copy_tile(ternary_b_cb, tile_id, ternary_b_dst_id);
#endif  // FUSE_TERNARY

#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif

#ifdef FUSE_TERNARY
            // Perform addcmul: output = ternary_a + (scalar * matmul_result * ternary_b)
            addcmul_tile_init();
            addcmul_tile<DataFormat::Float16_b>(
                ternary_a_dst_id,  // idst0: base value (from ternary_a)
                fused_act_dst_id,  // idst1: matmul+bias result
                ternary_b_dst_id,  // idst2: multiplier (from ternary_b)
                fused_act_dst_id,  // odst: output destination
                scalar_value);     // value: scalar multiplier
#endif

            pack_tile(fused_act_dst_id, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
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

#ifdef FUSE_TERNARY
    const uint32_t fused_ternary_scalar_uint = get_arg_val<uint32_t>(argidx++);
#else
    // Set default value to maintain compatibility with add_bias_and_addcmul_block
    // But in this case, this value is not used (set as constexpr to help compiler optimize out)
    constexpr uint32_t fused_ternary_scalar_uint = 0;
#endif

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

#if defined(FUSE_BIAS) || defined(FUSE_TERNARY)

    // Group both for add_bias_and_addcmul_block
    constexpr uint32_t in2_cb = tt::CBIndex::c_4;
    constexpr uint32_t cb_id_ternary_a = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_ternary_b = tt::CBIndex::c_6;
#endif  // defined(FUSE_BIAS) || defined(FUSE_TERNARY)

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    mm_init(in0_cb, in1_cb, intermediate_cb);

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

            mm_block_init_short(
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
                cb_wait_front(in0_cb, in0_block_num_tiles);
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
                    cb_pop_front(in0_cb, in0_block_num_tiles);
                }
                cb_pop_front(in1_cb, in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_push_back(intermediate_cb, out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));
            cb_wait_front(intermediate_cb, out_block_num_tiles);
            cb_reserve_back(out_cb, out_block_num_tiles);

#if !defined(FUSE_BIAS) && !defined(FUSE_TERNARY)
            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
#else
            // FUSED_BIAS or FUSED_TERNARY

#ifdef FUSE_BIAS
            cb_wait_front(in2_cb, N_block_tiles);
#endif

#ifdef FUSE_TERNARY
            cb_wait_front(cb_id_ternary_a, out_block_num_tiles);
            cb_wait_front(cb_id_ternary_b, out_block_num_tiles);
#endif  // FUSE_TERNARY

            add_bias_and_addcmul_block(
                intermediate_cb,
                in2_cb,
                cb_id_ternary_a,
                cb_id_ternary_b,
                fused_ternary_scalar_uint,
                out_cb,
                M_block_tiles,
                N_block_tiles);

#ifdef FUSE_BIAS
            cb_pop_front(in2_cb, N_block_tiles);
#endif  // FUSE_BIAS

#ifdef FUSE_TERNARY
            cb_pop_front(cb_id_ternary_a, out_block_num_tiles);
            cb_pop_front(cb_id_ternary_b, out_block_num_tiles);
#endif  // FUSE_TERNARY

#endif  // !defined(FUSE_BIAS) && !defined(FUSE_TERNARY)

            cb_pop_front(intermediate_cb, out_block_num_tiles);
        }
    }
}
