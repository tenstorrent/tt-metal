// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TurboQuant SDPA decode compute kernel — interleaved version.
//
// Dequantizes K/V from BFP4 indices+norms to BF16 one chunk at a time,
// interleaved with the SDPA attention math. This avoids needing CBs large
// enough to hold the entire dequantized cache.
//
// Per K/V chunk:
//   1. Dequantize K chunk  (cb_k_idx, cb_k_norms → cb_k_in)
//   2. Q × K^T  matmul + online softmax update
//   3. Dequantize V chunk  (cb_v_idx, cb_v_norms → cb_v_in)
//   4. Attn × V  matmul + output accumulation (ping-pong)
//
// Decode simplifications:
//   - Sq_chunk_t = 1 (single query token)
//   - q_num_chunks = 1
//   - is_causal = false (attend to all prior tokens)
//   - No mask, no attention sink, no sliding window

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define EXP_APPROX_MODE 1

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"

// Include existing SDPA compute building blocks (matmul_blocks, softmax helpers, etc.)
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

// ── Inline dequantize: BFP4 index tile → BF16 centroid×norm tile ──
// Reads 1 tile from cb_idx, 1 tile from cb_norms (already waiting).
// Writes 1 dequantized tile to cb_out via cb_temp for bcast multiply.
template <uint32_t NumLevels>
inline void dequantize_one_tile(
    uint32_t cb_idx,
    uint32_t cb_norms,
    uint32_t cb_temp,
    uint32_t cb_out,
    const float* centroids,
    const uint32_t* level_bits) {
    // Phase 1: Centroid gather → cb_temp
    cb_wait_front(cb_idx, 1);
    cb_reserve_back(cb_temp, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_idx);  // configure unpacker for BFP4 cb
    copy_tile(cb_idx, 0, 0);  // DST[0] = indices (BFP4 auto-unpack to f32)
    fill_tile_init();
    fill_tile(1, centroids[0]);

    for (uint32_t lev = 1; lev < NumLevels; lev++) {
        copy_tile(cb_idx, 0, 2);
        unary_ge_tile_init();
        unary_ge_tile(2, level_bits[lev]);
        fill_tile_init();
        fill_tile(3, centroids[lev]);
        sub_binary_tile_init();
        sub_binary_tile(3, 1, 3);
        mul_binary_tile_init();
        mul_binary_tile(2, 3, 3);
        add_binary_tile_init();
        add_binary_tile(1, 3, 1);
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_temp);  // ensure packer matches temp CB format
    pack_tile(1, cb_temp);
    tile_regs_release();
    cb_pop_front(cb_idx, 1);
    cb_push_back(cb_temp, 1);

    // Phase 2: Multiply by norm (broadcast column 0 across all columns)
    cb_wait_front(cb_temp, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    mul_bcast_cols_init_short(cb_temp, cb_norms);
    mul_tiles_bcast_cols(cb_temp, cb_norms, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);  // ensure packer matches output CB format
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_pop_front(cb_temp, 1);
    cb_push_back(cb_out, 1);
}

// Dequantize a full chunk (Sk_chunk_t rows × Wt tiles per row).
// Norms: 1 tile per row (shared across Wt head-dim tiles).
template <uint32_t Sk_chunk_t, uint32_t Wt, uint32_t NumLevels>
inline void dequantize_chunk(
    uint32_t cb_idx,
    uint32_t cb_norms,
    uint32_t cb_temp,
    uint32_t cb_out,
    const float* centroids,
    const uint32_t* level_bits) {
    for (uint32_t row = 0; row < Sk_chunk_t; row++) {
        cb_wait_front(cb_norms, 1);
        for (uint32_t col = 0; col < Wt; col++) {
            dequantize_one_tile<NumLevels>(cb_idx, cb_norms, cb_temp, cb_out, centroids, level_bits);
        }
        cb_pop_front(cb_norms, 1);
    }
}

// Template helper to load centroid compile-time args (must be at namespace scope).
template <uint32_t Idx, uint32_t N, uint32_t Base>
struct LoadSDPACentroids {
    static void apply(float* c, uint32_t* lb) {
        constexpr uint32_t bits = get_compile_time_arg_val(Base + 1 + Idx);
        union {
            uint32_t u;
            float f;
        } conv;
        conv.u = bits;
        c[Idx] = conv.f;
        float fi = static_cast<float>(Idx);
        __builtin_memcpy(&lb[Idx], &fi, sizeof(uint32_t));
        LoadSDPACentroids<Idx + 1, N, Base>::apply(c, lb);
    }
};
template <uint32_t N, uint32_t Base>
struct LoadSDPACentroids<N, N, Base> {
    static void apply(float*, uint32_t*) {}
};

void kernel_main() {
    // ── Compile-time args: standard SDPA + TQ centroids ──
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(18);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(21);
    constexpr uint32_t num_cores = get_compile_time_arg_val(22);

    constexpr bool is_causal = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(27);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);

    // TQ compile-time args
    constexpr uint32_t TQ_BASE = 33;
    constexpr uint32_t num_levels = get_compile_time_arg_val(TQ_BASE);

    // Runtime args
    const uint32_t local_batch_start = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(2);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(4);
    const uint32_t local_q_start = get_arg_val<uint32_t>(5);
    const uint32_t local_q_end = get_arg_val<uint32_t>(6);
    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // Load centroid constants from compile-time args (template-unrolled,
    // defined at namespace scope above kernel_main)
    float centroids[16];
    uint32_t level_bits_arr[16];
    LoadSDPACentroids<0, num_levels, TQ_BASE>::apply(centroids, level_bits_arr);

    // ── CB indices ──
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    constexpr uint32_t cb_k_idx = tt::CBIndex::c_10;
    constexpr uint32_t cb_k_norms = tt::CBIndex::c_11;
    constexpr uint32_t cb_v_idx = tt::CBIndex::c_12;
    constexpr uint32_t cb_v_norms = tt::CBIndex::c_13;
    constexpr uint32_t cb_dq_temp = tt::CBIndex::c_14;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    mm_init(cb_q_in, cb_k_in, cb_out);

    // ── Reader pushes BF16 K (transposed) and V directly to cb_k_in/cb_v_in ──
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            mm_init(cb_q_in, cb_k_in, cb_out);
            sdpa_standard<
                cb_qk_im,
                cb_identity_scale_in,
                (uint32_t)tt::CBIndex::c_4,
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                vDHt,
                false /*use_attention_sink*/,
                false /*is_causal*/,
                false /*use_provided_mask*/,
                false /*use_padded_mask*/,
                false /*is_chunked*/,
                scale_fp32,
                (uint32_t)0 /*sliding_window*/>(
                Skt,
                qk_in0_block_w,
                qk_subblock_w,
                qk_subblock_h,
                qk_in0_num_subblocks,
                qk_in1_num_subblocks,
                qk_num_blocks,
                out_in0_block_w,
                out_subblock_w,
                out_subblock_h,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_num_blocks,
                (uint32_t)0,  // iter_q_start
                (uint32_t)1,  // iter_q_end
                (uint32_t)1,  // q_num_chunks
                (uint32_t)0,  // local_q_start
                (uint32_t)0,  // chunked_q_chunk_offset
                k_num_chunks,
                q_chunk_tiles,
                k_chunk_tiles,
                v_chunk_tiles,
                qk_chunk_tiles,
                out_chunk_tiles,
                cb_q_in,
                cb_k_in,
                cb_v_in,
                (uint32_t)tt::CBIndex::c_3 /*cb_mask_in unused*/,
                cb_col_identity,
                cb_out_im_A,
                cb_out_im_B,
                cb_max_A,
                cb_max_B,
                cb_sum_A,
                cb_sum_B,
                cb_exp_max_diff,
                cb_out);
        }
    }
}
