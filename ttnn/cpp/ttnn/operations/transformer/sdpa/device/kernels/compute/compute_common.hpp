// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/debug/assert.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/reduce_custom.h"
#include "api/dataflow/circular_buffer.h"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "cpp/ttnn/kernel_lib/dest_helpers.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif

ALWI void sdpa_reduce_copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose = 0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, true /*transpose within 16x16 face*/, cbid)));

    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false,  // is_int_fpu_en
          PackMode::Default>(cbid)));
}

/**
 * in0_cb = max(in0_cb, in1_cb)
 */
template <uint32_t num_tiles>
void max_block_inplace(uint32_t in0, uint32_t in1) {
    CircularBuffer cb_in0(in0);
    CircularBuffer cb_in1(in1);
    // inputs come in full, outputs go out full
    copy_init(in0);
    copy_init(in1);
    binary_max_tile_init();
    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_0, VectorMode::C);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_reg_0, in0);
        tile_regs_release();
    }
    cb_in0.pop_front(num_tiles);
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
}

#ifdef TRISC_MATH
inline void sdpa_max_sfpi(const std::uint32_t in0, const std::uint32_t in1, const std::uint32_t out) {
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[in0 * dst_tile_size_sfpi];
        const sfpi::vFloat b = sfpi::dst_reg[in1 * dst_tile_size_sfpi];
        v_if(b > a) { a = b; }
        v_endif;
        sfpi::dst_reg[out * dst_tile_size_sfpi] = a;
        sfpi::dst_reg++;
    }
}
#endif

// max_block without SFPLOADMACRO, whose state races the pack thread's exp when a merge follows the K loop.
void max_block_sfpi(uint32_t in0, uint32_t in1, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0);
    CircularBuffer cb_in1(in1);
    CircularBuffer cb_out(out_cb);
    copy_init(in0);
    add_binary_tile_init();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in0, i, 0);
        copy_tile(in1, i, 1);
        MATH((_llk_math_eltwise_binary_sfpu_params_(sdpa_max_sfpi, 0, 1, 0, VectorMode::RC)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb, i);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

// Pushes num_tiles zero tiles to out_cb, made as x - x from the front tile of like_cb (which stays).
void zero_block(uint32_t like_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_like(like_cb);
    CircularBuffer cb_out(out_cb);
    sub_init(like_cb, like_cb);
    cb_like.wait_front(1);
    cb_out.reserve_back(num_tiles);
    tile_regs_acquire();
    sub_tiles(like_cb, like_cb, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        pack_tile(0, out_cb);
    }
    tile_regs_release();
    cb_out.push_back(num_tiles);
}

/**
 * out_cb = eltwise_max(in0, in1)
 */
template <VectorMode vector_mode = VectorMode::RC>
void max_block(uint32_t in0, uint32_t in1, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0);
    CircularBuffer cb_in1(in1);
    CircularBuffer cb_out(out_cb);
    // inputs come in full, outputs go out full
    copy_init(in0);
    binary_max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_0, vector_mode);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_reg_0, out_cb, i);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

/**
 * out_cb = reduce[MAX,SUM](in0_cb * scale_cb)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t rows,
    uint32_t cols,
    VectorMode vector_mode = VectorMode::C>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, bool do_eltwise_max = false) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_scale(scale_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_prev(prev_cb);
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced
    // If do_eltwise_max == true, prev_cb has rows produced.

#if defined REDUCE_GRANULARITY
    constexpr uint32_t dst_tiles = (rows < REDUCE_GRANULARITY) ? rows : REDUCE_GRANULARITY;
    constexpr uint32_t granularity = (rows >= REDUCE_GRANULARITY) ? (rows / REDUCE_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = rows;
#endif

    cb_scale.wait_front(1);
    cb_out.reserve_back(rows);

    const uint32_t num_tiles_to_wait = dst_tiles * cols;
    uint32_t in0_wait_tiles = num_tiles_to_wait;

    uint32_t row_start_idx = 0;
    for (uint32_t g = 0; g < granularity; g++) {
        cb_in0.wait_front(in0_wait_tiles);
        tile_regs_acquire();

        if (do_eltwise_max) {
            cb_prev.wait_front(g * dst_tiles);
            /**
             * Copy previous max values into DST register.
             * Note that this special invocation of copy_tile is necessary to produce
             * tiles in DST with transposed faces, as `reduce_block_max_row` expects.
             */
            reconfig_data_format_srca(prev_cb);
            sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
            for (uint32_t i = 0; i < dst_tiles; i++) {
                const uint32_t cur_max_dst_idx = i;
                copy_tile(prev_cb, (row_start_idx + i), cur_max_dst_idx);
            }
            reconfig_data_format_srca(in0_cb);
        }

        /**
         * For `dst_tiles` number of rows, compute the max into the even indices of the DST register.
         */
        reduce_block_max_row_init<cols>(in0_cb, scale_cb, out_cb);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t reduce_dst_idx = i;
            reduce_block_max_row<cols>(in0_cb, scale_cb, (row_start_idx + i) * cols, reduce_dst_idx);
        }
        reduce_block_max_row_uninit(in0_cb);

        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(out_cb);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t cur_max_dst_idx = i;
            pack_tile<true>(cur_max_dst_idx, out_cb, (row_start_idx + i));
        }
        tile_regs_release();

        row_start_idx += dst_tiles;
        in0_wait_tiles += num_tiles_to_wait;
    }

    cb_out.push_back(rows);
}

/**
 * out_cb = reduce[MAX,SUM](in0_cb * scale_cb)
 *
 * In this version cols does not have to be a compile-time constant.
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t rows,
    VectorMode vector_mode = VectorMode::C>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, uint32_t cols, bool do_eltwise_max = false) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_scale(scale_cb);
    CircularBuffer cb_out(out_cb);
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    uint32_t num_tiles = rows * cols;
    cb_scale.wait_front(1);
    cb_in0.wait_front(num_tiles);
    cb_out.reserve_back(rows);

    pack_reconfig_data_format(out_cb);

    binary_max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1;

    for (uint32_t i = 0; i < rows; i++) {
        reconfig_data_format_srca(in0_cb);
        tile_regs_acquire();
        reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
        }
        reduce_uninit();
        if (do_eltwise_max) {
            reconfig_data_format_srca(prev_cb);
            copy_init(prev_cb);
            copy_tile(prev_cb, i, prev_max_dst_idx);
            binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx, vector_mode);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(reduce_dst_idx, out_cb);
        tile_regs_release();
    }

    cb_out.push_back(rows);
}

/**
 * out_cb = max(in0_cb, dim=-1), and eltwise_max(prev_cb, ...) if do_eltwise_max.
 *
 * The FPU max reduce keeps only 8 significant bits, so for an fp32 in0_cb its row max can miss the
 * row's peak by far more than exp's range once scaled. This folds and reduces on the SFPU instead, so
 * the max equals the peak as the FPU later reads it from in0_cb and (in0 - max) peaks at exactly 0.
 * out_cb gets the max in column 0 of each tile, as reduce_c does.
 */
template <uint32_t in0_cb, uint32_t rows>
void reduce_max_row_fp32(uint32_t out_cb, uint32_t prev_cb, uint32_t cols, bool do_eltwise_max = false) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_out(out_cb);
    // Precondition: in0_cb has rows*cols produced, in row-major order
    // Precondition: out_cb has rows free
    // Precondition: if do_eltwise_max, prev_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: out_cb has rows produced
    cb_in0.wait_front(rows * cols);
    cb_out.reserve_back(rows);
    pack_reconfig_data_format(out_cb);

    constexpr uint32_t max_dst_idx = 0;
    constexpr uint32_t in_dst_idx = 1;
    for (uint32_t i = 0; i < rows; i++) {
        tile_regs_acquire();
        reconfig_data_format_srca(in0_cb);
        copy_init(in0_cb);
        binary_max_tile_init();
        copy_tile(in0_cb, i * cols, max_dst_idx);
        for (uint32_t j = 1; j < cols; j++) {
            copy_tile(in0_cb, i * cols + j, in_dst_idx);
            binary_max_tile(max_dst_idx, in_dst_idx, max_dst_idx);
        }
        sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(max_dst_idx);
        if (do_eltwise_max) {
            reconfig_data_format_srca(prev_cb);
            copy_init(prev_cb);
            copy_tile(prev_cb, i, in_dst_idx);
            binary_max_tile_init();
            binary_max_tile(max_dst_idx, in_dst_idx, max_dst_idx, VectorMode::C);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(max_dst_idx, out_cb);
        tile_regs_release();
    }

    cb_out.push_back(rows);
}

#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void recip_tile_first_column(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_recip_first_column, (is_fp32_dest_acc_en), idst, VectorMode::C);
}
#endif

/**
 * in_cb = 1 / in_cb
 */
void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    reconfig_data_format_srca(in_cb);
    copy_init(in_cb);
    // The first-column helper uses SFPI, not full-tile LOADMACRO/replay state.
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::sfpu_reciprocal_init, (APPROX)));
    pack_reconfig_data_format(in_cb);

    cb_in.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0);
        MATH((recip_tile_first_column(0)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, in_cb);
        tile_regs_release();
    }
    cb_in.pop_front(num_tiles);
    cb_in.reserve_back(num_tiles);
    cb_in.push_back(num_tiles);
}

/**
 * in0_cb = exp((in0_cb - in1_cb) * scale_fp32)
 */
template <
    uint32_t in0_cb,
    uint32_t rows,
    uint32_t scale_fp32,
    bool write_result_inplace = true,
    bool do_reduce = true,
    VectorMode vector_mode = VectorMode::RC>
void sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_reduce(reduce_cb);
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced
    // llk_unpack_AB_init (inside sub_bcast_cols_init) validates the live
    // unpacker configuration.  Reconfigure first: qk_im can be FP32 while the
    // row maximum is BF16, notably after applying a windowed BF16 mask.
    reconfig_data_format(in0_cb, in1_cb);
    sub_bcast_cols_init(in0_cb, in1_cb);

    // Approximate exp skips negative-input clamping for speed. Inputs below about -88 can
    // produce negative outputs, which packer ReLU clears. Keep this path for partial faces.
    // The accurate branch below handles full RC tiles.
    if constexpr (EXP_APPROX_MODE || vector_mode != VectorMode::RC) {
        exp_tile_init<true /* approx */, scale_fp32, InputClamping::None>();
    }
    PACK((llk_pack_relu_config(ReluConfig::zero())));

    cb_in0.wait_front(rows * cols);
    cb_in1.wait_front(rows);
    if constexpr (do_reduce) {
        cb_reduce.reserve_back(rows);
    }

#ifdef SUB_EXP_GRANULARITY
    uint32_t dst_tiles = (cols < SUB_EXP_GRANULARITY) ? cols : SUB_EXP_GRANULARITY;
    uint32_t granularity = (cols >= SUB_EXP_GRANULARITY) ? (cols / SUB_EXP_GRANULARITY) : 1;
#else
    uint32_t dst_tiles = cols;
    uint32_t granularity = 1;
#endif

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
                // A 32x32 tile has four 16x16 faces, each requiring eight SFPU iterations.
                // None visits the full tile in 32 iterations; R/C traverse faces with eight each.
                constexpr int iterations = (vector_mode == VectorMode::RC) ? 32 /*ITER*/ : 8 /*ITER*/;
                constexpr VectorMode vector_mode_exp = (vector_mode == VectorMode::RC) ? VectorMode::None : vector_mode;
                if constexpr (EXP_APPROX_MODE || vector_mode != VectorMode::RC) {
                    exp_tile<true /* approx */, false /* scale_en */, InputClamping::None, iterations>(
                        j, vector_mode_exp);
                } else {
                    // Apply the full FP32 attention scale once before accurate exponentiation.
                    // The init scale 0x3F800000 is the IEEE-754 encoding of 1.0f.
                    // Negative clamping protects masked/large-negative scores in accurate BF16 exp.
                    binop_with_scalar_tile_init();
                    mul_unary_tile(j, scale_fp32);
                    exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative>();
                    exp_tile<false, false, InputClamping::ClampToNegative, iterations>(j, vector_mode_exp);
                }
            }
            tile_regs_commit();

            if constexpr (write_result_inplace) {
                cb_in0.pop_front(dst_tiles);
                cb_in0.reserve_back(dst_tiles);
            }

            tile_regs_wait();

            if constexpr (write_result_inplace) {
                pack_reconfig_data_format(in0_cb);
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, in0_cb);
                }
                // Granular write output to enable following matmul unpack to start early.
                cb_in0.push_back(dst_tiles);
            }

            if constexpr (do_reduce) {
                pack_reconfig_data_format(reduce_cb);
                // While we have results in DST, take advantage of L1 accumulation
                // to reduce row x cols tiles to rows x 1 tiles.
                if (u > 0) {
                    // If on the same row, keep accumulating
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile<true>(j, reduce_cb, i);
                    if (u == 0 && j == 0) {
                        // If this was the first tile of a row, start accumulating
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
            tile_regs_release();
            if constexpr (do_reduce) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
        }
    }
    if constexpr (do_reduce) {
        cb_reduce.push_back(rows);
    }

    PACK((llk_pack_relu_config(ReluConfig::none())));
}

/**
 * out_cb = in0_cb * in1_cb
 * @tparam rows - Number of rows of tiles
 * @tparam cols - Number of columns of tiles
 * @tparam immediate_pop - If true, uses tile-by-tile processing with immediate CB pop after each tile.
 *                         If false, uses batched processing with deferred CB pop, processing multiple tiles in
 * parallel.
 * @tparam pack_accumulate - If true, enables L1 accumulation to accumulate results onto existing tiles
 *                           in out_cb. Only supported when immediate_pop=false.
 */
template <uint32_t rows, uint32_t cols, bool immediate_pop, bool pack_accumulate>
void mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Precondition: out_cb has rows*cols produced
    // Postcondition: in0_cb empty
    // Postcondition: in1_cb empty
    // Postcondition: out_cb has rows*cols produced

    constexpr uint32_t num_tiles = rows * cols;

    reconfig_data_format(in0_cb, in1_cb);
    pack_reconfig_data_format(out_cb);
    mul_bcast_cols_init(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(rows);

    if constexpr (immediate_pop) {
        static_assert(!pack_accumulate, "Unsupported parameter configuration");
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                tile_regs_acquire();
                mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
                tile_regs_commit();
                cb_in0.pop_front(1);
                cb_out.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, out_cb);
                tile_regs_release();
                cb_out.push_back(1);
            }
        }
        cb_in1.pop_front(rows);
    } else {
#ifdef DHT_GRANULARITY
        constexpr uint32_t dst_tiles = (cols < DHT_GRANULARITY) ? cols : DHT_GRANULARITY;
        constexpr uint32_t granularity = (cols >= DHT_GRANULARITY) ? (cols / DHT_GRANULARITY) : 1;
#else
        constexpr uint32_t dst_tiles = 1;
        constexpr uint32_t granularity = cols;
#endif
        PACK((llk_pack_reconfig_l1_acc(pack_accumulate)));
        if (!pack_accumulate) {
            cb_out.reserve_back(num_tiles);
        }
        uint32_t in0_index = 0;
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t u = 0; u < granularity; ++u) {
                tile_regs_acquire();
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    mul_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                    in0_index++;
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, out_cb);
                }
                tile_regs_release();
            }
        }
        cb_in1.pop_front(rows);
        cb_in0.pop_front(num_tiles);
        if (pack_accumulate) {
            PACK((llk_pack_reconfig_l1_acc(false)));
            cb_out.pop_front(num_tiles);
            cb_out.reserve_back(num_tiles);
            cb_out.push_back(num_tiles);
        } else {
            cb_out.push_back(num_tiles);
        }
    }
}

/**
 * in0_cb *= in1_cb
 */
template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    constexpr uint32_t num_tiles = rows * cols;

#ifdef DHT_GRANULARITY
    constexpr uint32_t dst_tiles = (cols < DHT_GRANULARITY) ? cols : DHT_GRANULARITY;
    constexpr uint32_t granularity = (cols >= DHT_GRANULARITY) ? (cols / DHT_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = cols;
#endif

    reconfig_data_format(in0_cb, in1_cb);
    mul_bcast_cols_init(in0_cb, in1_cb);
    pack_reconfig_data_format(in0_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; ++u) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                mul_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
            }
            tile_regs_commit();
            cb_in0.pop_front(dst_tiles);
            cb_in0.reserve_back(dst_tiles);
            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_cb);
            }
            cb_in0.push_back(dst_tiles);
            tile_regs_release();
        }
    }
    cb_in1.pop_front(rows);
    reconfig_data_format_srcb(in0_cb);
}

template <uint32_t in1_scalar_cb, uint32_t num_tiles>
void mul_block_bcast_scalar_inplace(uint32_t in0_cb) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1_scalar(in1_scalar_cb);
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced

#ifdef STATS_GRANULARITY
    constexpr uint32_t dst_tiles = STATS_GRANULARITY;
    constexpr uint32_t granularity = num_tiles / STATS_GRANULARITY;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = num_tiles;
#endif

    reconfig_data_format(in0_cb, in1_scalar_cb);
    mul_bcast_scalar_init(in0_cb, in1_scalar_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1_scalar.wait_front(1);
    uint32_t in0_index = 0;
    for (uint32_t g = 0; g < granularity; ++g) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, in0_index, 0, i);
            in0_index++;
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_cb);
        }
        tile_regs_release();
    }
    cb_in0.pop_front(num_tiles);
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
}

/**
 * in0_cb += in1_cb
 */
template <bool pop_in1 = true>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    reconfig_data_format(in0_cb, in1_cb);
    pack_reconfig_data_format(in0_cb);
    add_init(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        add_tiles(in0_cb, in1_cb, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, in0_cb);
        tile_regs_release();
    }

    cb_in0.pop_front(num_tiles);
    if (pop_in1) {
        cb_in1.pop_front(num_tiles);
    }
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
}

void mul_tiles_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    /**
     * Given in0_cb and in1_cb, multiply each tile of in0_cb by the corresponding tile of in1_cb
     * and bcast cols of in1_cb.
     */
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    reconfig_data_format(in0_cb, in1_cb);
    mul_bcast_cols_init(in0_cb, in1_cb);
    pack_reconfig_data_format(in0_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
        tile_regs_commit();
        cb_in0.pop_front(1);
        cb_in0.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, in0_cb);
        tile_regs_release();
        cb_in0.push_back(1);
    }
}

/**
 * in0_cb *= in1_cb
 */
void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_init(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        tile_regs_acquire();
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        tile_regs_commit();
        cb_in0.pop_front(1);
        cb_in0.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, in0_cb);
        tile_regs_release();
        cb_in0.push_back(1);
    }
}

#if defined(TRISC_MATH) || defined(TRISC_PACK)

template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void exp_tile_first_column(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_exponential_first_column,
        (SDPA_EXP_APPROX_MODE, scale_bf16, is_fp32_dest_acc_en),
        idst,
        VectorMode::C);
}
#endif  // defined(TRISC_MATH) || defined(TRISC_PACK)

/**
 * out_cb = exp((in0_cb - in1_cb) * scale_fp32)
 */
template <uint32_t scale_fp32>
void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in0_cb and in1_cb has num_tiles produced

    sub_init(in0_cb, in1_cb);
    exp_tile_init<EXP_APPROX_MODE>();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);

    // Convert scale_fp32 to bf16 scale
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        tile_regs_acquire();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_out.push_back(1);
    }
}

#ifdef TRISC_MATH
template <VectorMode vector_mode = VectorMode::C, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    constexpr bool reuse_cur_max_tile = is_fp32_dest_acc_en && DST_SYNC_MODE == DstSync::SyncHalf;
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_fused_max_sub_exp_add_tile,
        (is_fp32_dest_acc_en, reuse_cur_max_tile),
        idst,
        vector_mode,
        scale_bf16);
}
#endif

template <uint32_t scale_fp32, VectorMode vector_mode = VectorMode::C>
void correction_block(
    uint32_t cb_worker_max,
    uint32_t cb_worker_sum,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    uint32_t cb_cur_sum,
    uint32_t cb_prev_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_exp_max_diff_2,
    uint32_t num_head_tiles) {
    CircularBuffer cb_worker_max_obj(cb_worker_max);
    CircularBuffer cb_worker_sum_obj(cb_worker_sum);
    CircularBuffer cb_cur_max_obj(cb_cur_max);
    CircularBuffer cb_prev_max_obj(cb_prev_max);
    CircularBuffer cb_cur_sum_obj(cb_cur_sum);
    CircularBuffer cb_prev_sum_obj(cb_prev_sum);
    CircularBuffer cb_exp_max_diff_obj(cb_exp_max_diff);
    CircularBuffer cb_exp_max_diff_2_obj(cb_exp_max_diff_2);
    cb_worker_max_obj.wait_front(num_head_tiles);
    cb_worker_sum_obj.wait_front(num_head_tiles);
    cb_prev_max_obj.wait_front(num_head_tiles);
    cb_prev_sum_obj.wait_front(num_head_tiles);

    cb_cur_max_obj.reserve_back(num_head_tiles);
    cb_cur_sum_obj.reserve_back(num_head_tiles);
    cb_exp_max_diff_obj.reserve_back(num_head_tiles);
    cb_exp_max_diff_2_obj.reserve_back(num_head_tiles);

    constexpr uint32_t dst_reg_0 = 0;  // dst_reg_0 is used for prev_max
    constexpr uint32_t dst_reg_1 = 1;  // dst_reg_1 is used for worker_max
    constexpr uint32_t dst_reg_2 = 2;  // cur_max output; also worker_sum input in FP32 half-sync
    constexpr uint32_t dst_reg_3 = 3;  // dst_reg_3 is used for prev_sum, returns cur_sum
    constexpr uint32_t dst_reg_4 = 4;  // worker_sum in the five-tile layout
    // #56171: FP32 half-sync only has slots 0..3. Reuse the cur_max output
    // slot for worker_sum, which the SFPU loads before writing cur_max.
    constexpr uint32_t worker_sum_dst = (DST_ACCUM_MODE && DST_SYNC_MODE == DstSync::SyncHalf) ? dst_reg_2 : dst_reg_4;
    static_assert(
        worker_sum_dst < compute_kernel_lib::DEST_AUTO_LIMIT,
        "correction_block DST layout exceeds DEST capacity for this sync/accum mode");

    // convert scale from fp32 to bf16
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_head_tiles; i++) {
        tile_regs_acquire();
        copy_init(cb_worker_max);
        exp_tile_init<EXP_APPROX_MODE>();
        copy_tile(cb_prev_max, i, dst_reg_0);
        copy_tile(cb_worker_max, i, dst_reg_1);
        copy_tile(cb_prev_sum, i, dst_reg_3);
        copy_tile(cb_worker_sum, i, worker_sum_dst);
        MATH((fused_max_sub_exp_add_tile<vector_mode>(0, scale_bf16)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_reg_0, cb_exp_max_diff);
        pack_tile(dst_reg_1, cb_exp_max_diff_2);
        pack_tile(dst_reg_2, cb_cur_max);
        pack_tile(dst_reg_3, cb_cur_sum);
        tile_regs_release();
        cb_cur_max_obj.push_back(1);
        cb_cur_sum_obj.push_back(1);
        cb_exp_max_diff_obj.push_back(1);
        cb_exp_max_diff_2_obj.push_back(1);
    }
    cb_prev_sum_obj.pop_front(num_head_tiles);
    cb_worker_sum_obj.pop_front(num_head_tiles);
}

/**
 * in_cb -> out_cb
 */
template <bool pop_in_cb>
void move_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced

    copy_init(in_cb);

    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);

#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0 /*dst*/);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_out.push_back(1);
    }
    if (pop_in_cb) {
        cb_in.pop_front(num_tiles);
    }
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced
    copy_init(in_cb);
    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0 /*dst*/);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_out.push_back(1);
    }
    cb_in.pop_front(num_tiles);
}

void log_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);
    copy_init(in_cb);
    log_tile_init();
    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0 /*dst*/);
        log_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_out.push_back(1);
    }
}

void sigmoid_sub(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    // out_cb = sigmoid(in0_cb - in1_cb)
    /**
     * sigmoid(x) is accurately implemented as 1 / (1 + exp(-x))
     * This function manually implements the composite, accurate sigmoid.
     *
     * Each input tile has only the first column containing valid data, so VectorMode::C is a useful optimization.
     */
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    sub_init(in0_cb, in1_cb);
    exp_tile_init<false>();
    // recip_tile_first_column() calls the scalar sfpu_reciprocal_iter path, so initialize exactly
    // that SFPU state here. Blackhole needs vConstFloatPrgm0 = 2.0 for Newton-Raphson; Wormhole
    // needs vConstFloatPrgm0/1/2 loaded with reciprocal polynomial coefficients.
    // This init programs persistent SFPU constants, not per-tile data. It intentionally comes after
    // exp_tile_init<false>() because the exp call below is the custom exp_tile_first_column<false>(),
    // whose polynomial implementation loads the constants it consumes into LREGs in the tile body; it
    // does not depend on vConstFloatPrgm* state from exp_tile_init. Conversely, that exp body also
    // does not clobber the reciprocal constants, so one reciprocal init before the tile loop is enough.
    // That holds for exp_tile_first_column only. The regular exp_tile (the commented-out call in the
    // loop) takes its constants from the programmable registers exp_tile_init sets (vConstFloatPrgm0..1
    // on a bf16 DEST, 0..2 on a Blackhole fp32 DEST; see the note on exp_tile), and exp_tile_init
    // overwrites the reciprocal constant set here. Restoring that call needs exp_tile_init and
    // sfpu_reciprocal_init re-run inside the loop, each directly before its tile call.
    MATH((ckernel::sfpu::sfpu_reciprocal_init<false>()));

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        // Restoring this exp_tile call needs exp_tile_init and sfpu_reciprocal_init re-run per iteration;
        // see the sfpu_reciprocal_init comment above the loop.
        // exp_tile<false, true /*SCALE_EN*/>(0, (int)VectorMode::C, (uint16_t)0xBF80 /*bf16(-1.0) scale*/);
        MATH((exp_tile_first_column<false /*APPROX_MODE*/, (uint16_t)0xBF80 /*bf16(-1.0) scale*/>(0)));
        // add_unary_tile(0 /*dst_index*/, 0x3F800000); // Call the macro directly to get access to VectorMode argument
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ADD_UNARY, 8 /* ITERATIONS */, DST_ACCUM_MODE),
            0 /*dst_index*/,
            VectorMode::C,
            0x3F800000 /*scalar*/));
        MATH((recip_tile_first_column(0 /*dst_index*/)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void softplus_tile_first_column(uint32_t idst, uint beta, uint beta_reciprocal, uint threshold) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_softplus_first_column,
        (is_fp32_dest_acc_en),
        idst,
        VectorMode::C,
        beta,
        beta_reciprocal,
        threshold);
}
#endif

void logsigmoid_sub(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    // out_cb = logsigmoid(in0_cb - in1_cb)
    // Implemented as softplus for numerical stability. logsigmoid(x) = -softplus(-x)
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    sub_init(in0_cb, in1_cb);
    softplus_tile_init();
    constexpr uint32_t const_1_fp32 = 0x3F800000;
    constexpr uint32_t const_20_fp32 = 0x41A00000;

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        // Negate input to softplus by swapping inputs to sub
        sub_tiles(in1_cb, in0_cb, i, i, 0);
        // softplus_tile(0, 0x3F800000, 0x3F800000, 0x41A00000);  // beta, beta_reciprocal, threshold
        // MATH((llk_math_eltwise_unary_sfpu_softplus<APPROX>(
        //     0,
        //     const_1_fp32 /*beta*/,
        //     const_1_fp32 /*beta_reciprocal*/,
        //     const_20_fp32 /*threshold*/,
        //     (int)VectorMode::C)));

        MATH((softplus_tile_first_column(0, const_1_fp32, const_1_fp32, const_20_fp32)));
        // Negate the output of softplus
        negative_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

/**
 * out_cb = in0_cb - in1_cb
 * Compile with size optimization to prevent binary size exceeding the limit.
 */
__attribute__((optimize("Os"))) void sub_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    sub_init(in0_cb, in1_cb);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

/**
 * out_cb = in0_cb @ in1_cb
 */
ALWI void matmul_blocks(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& transpose,
    const bool& add_mask = false,
    const uint32_t& mask_cb = 0,
    const uint32_t& zero_cb = 0) {
    // precondition: in0_cb has M*K produced
    // precondition: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced

    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_mask(mask_cb);
    CircularBuffer cb_zero(zero_cb);

    matmul_block_init(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    const uint32_t output_num_tiles = M * N;
    const uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    const uint32_t in0_subblock_all_cols_num_tiles = subblock_h * N;

    uint32_t in0_index_offset = 0;

    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;

    reconfig_data_format(in1_cb, in0_cb);
    cb_in1.wait_front(K * N);
    cb_out.reserve_back(output_num_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        cb_in0.wait_front(in0_wait_tiles);
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            if (add_mask) {
                cb_mask.wait_front(out_subblock_num_tiles);
                cb_zero.wait_front(1);
                reconfig_data_format(zero_cb, mask_cb);
                add_init(zero_cb, mask_cb, true);
                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                    add_tiles(zero_cb, mask_cb, 0, i, i);
                }
                reconfig_data_format(in1_cb, in0_cb);
                matmul_block_init(in0_cb, in1_cb, transpose, subblock_w, subblock_h, in0_block_w);
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = r * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;
        // Somewhat granularize the push of in0 subblocks
        cb_out.push_back(in0_subblock_all_cols_num_tiles);
    }
    cb_in1.pop_front(K * N);
}

template <uint32_t M>
void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb) {
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    // precondition: in0_cb has M*K produced
    // precondition: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced

    constexpr uint32_t N = 1;  // Result of reduce is 1 column
    constexpr uint32_t in0_block_w = N;
    constexpr uint32_t subblock_w = N;
    // Reuse the Sq_chunk_t granularity chosen for sub_exp_block
#ifdef STATS_GRANULARITY
    constexpr uint32_t subblock_h = STATS_GRANULARITY;
    constexpr uint32_t in0_num_subblocks = M / STATS_GRANULARITY;
#else
    constexpr uint32_t subblock_h = 1;
    constexpr uint32_t in0_num_subblocks = M;
#endif

    /**
     * Use matmul on Mx1 input to reduce rows within tile to produce Mx1 output.
     */

    // matmul_block_init validates the live reverse-order unpacker setup
    // (in1_cb -> SrcA, out_cb -> SrcB), so establish it before init.
    reconfig_data_format(in1_cb, out_cb);
    matmul_block_init(
        out_cb, in1_cb, 0 /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    pack_reconfig_data_format(out_cb);
    cb_in1.wait_front(N);
    cb_out.wait_front(M);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        tile_regs_acquire();

        uint32_t dst_index = 0;
        uint32_t in0_index = 0;
        uint32_t in1_index = 0;

        matmul_block(out_cb, in1_cb, in0_index, in1_index, dst_index, 0, subblock_w, subblock_h, in0_block_w);

        tile_regs_commit();
        cb_out.pop_front(subblock_h);

        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_h; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_out.push_back(subblock_h);
    }
}

/**
 * Batch-stamp a single tile onto a range of positions in out_cb using L1 accumulate.
 * Caller must have already called copy_init and llk_pack_reconfig_l1_acc(1).
 *
 * @tparam dst_batch  Max tiles per DST cycle (DST register capacity, typically 8 for fp16b half-sync).
 */
template <uint32_t dst_batch>
void stamp_tile_range_l1_acc(
    uint32_t src_cb, uint32_t src_tile_idx, uint32_t out_cb, uint32_t out_offset, uint32_t count) {
    for (uint32_t base = 0; base < count; base += dst_batch) {
        uint32_t batch = (count - base < dst_batch) ? (count - base) : dst_batch;
        tile_regs_acquire();
        for (uint32_t i = 0; i < batch; i++) {
            copy_tile(src_cb, src_tile_idx, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < batch; i++) {
            pack_tile<true>(i, out_cb, out_offset + base + i);
        }
        tile_regs_release();
    }
}

template <uint32_t dst_batch>
void apply_padded_mask_lightweight_runtime(
    uint32_t neginf_cb,
    uint32_t neginf_tile_idx,
    uint32_t out_cb,
    uint32_t num_padded,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_base = 0) {  // first out_cb tile-row of this query band; nonzero when heads span >1 DEST band
    uint32_t start = num_cols - num_padded;

    reconfig_data_format_srca(neginf_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(neginf_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        stamp_tile_range_l1_acc<dst_batch>(
            neginf_cb, neginf_tile_idx, out_cb, (row_base + row) * num_cols + start, num_padded);
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Lightweight partial mask: L1-accumulate a partial mask tile (0 for valid, -inf for padded columns)
 * onto the boundary tile position in out_cb. The partial tile is permanently fronted in the CB.
 *
 * @param mask_cb          CB holding mask tiles, permanently fronted
 * @param partial_tile_idx Index of the partial tile within the CB
 * @param out_cb           QK intermediate CB (already wait-fronted)
 * @param boundary_col     Column index within the chunk where the boundary tile is
 * @param num_cols         Total K tiles per row (Sk_chunk_t)
 * @param num_rows         Q tiles per chunk (Sq_chunk_t)
 */
void apply_partial_mask_lightweight(
    uint32_t mask_cb,
    uint32_t partial_tile_idx,
    uint32_t out_cb,
    uint32_t boundary_col,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_base = 0) {  // first out_cb tile-row of this query band; nonzero when heads span >1 DEST band
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        tile_regs_acquire();
        copy_tile(mask_cb, partial_tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb, (row_base + row) * num_cols + boundary_col);
        tile_regs_release();
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Lightweight causal mask: stamps neginf and diagonal tiles onto QKT using L1 accumulate.
 *
 * For Q tile-row i (0..num_rows-1) processing K chunk starting at k_start_tile:
 *   diag_col = q_start_tile + i - k_start_tile
 *   - diag_col < 0:           entire row above diagonal -> stamp neginf on all num_cols tiles
 *   - 0 <= diag_col < num_cols: diagonal tile at col diag_col, neginf at cols diag_col+1..num_cols-1
 *   - diag_col >= num_cols:     entire row below diagonal -> no mask needed
 */
template <uint32_t dst_batch>
void apply_causal_mask_lightweight(
    uint32_t mask_cb,
    uint32_t neginf_idx,
    uint32_t diag_idx,
    uint32_t out_cb,
    uint32_t q_start_tile,
    uint32_t k_start_tile,
    uint32_t num_rows,
    uint32_t num_cols,
    uint32_t straddle_col = 0,
    uint32_t straddle_jump = 0) {
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        uint32_t row_offset = row * num_cols;
        const int32_t q_pos = static_cast<int32_t>(q_start_tile + row);

        if (straddle_col == 0) {
            // Fast path: K coords contiguous across cols.
            int32_t diag_col = q_pos - static_cast<int32_t>(k_start_tile);
            if (diag_col < 0) {
                // Entire row above diagonal -> stamp all neginf
                stamp_tile_range_l1_acc<dst_batch>(mask_cb, neginf_idx, out_cb, row_offset, num_cols);
            } else if (static_cast<uint32_t>(diag_col) < num_cols) {
                // Stamp the diagonal tile
                tile_regs_acquire();
                copy_tile(mask_cb, diag_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, out_cb, row_offset + static_cast<uint32_t>(diag_col));
                tile_regs_release();

                // Stamp neginf tiles to the right of diagonal
                uint32_t neginf_start = static_cast<uint32_t>(diag_col) + 1;
                if (neginf_start < num_cols) {
                    stamp_tile_range_l1_acc<dst_batch>(
                        mask_cb, neginf_idx, out_cb, row_offset + neginf_start, num_cols - neginf_start);
                }
            }
            // else: diag_col >= num_cols -> entire row below diagonal, no mask needed
        } else {
            // Chunked-prefill straddle: K coord jumps by straddle_jump at col >= straddle_col
            // (the K-chunk crosses a slab boundary). Evaluate per-col.
            for (uint32_t col = 0; col < num_cols; col++) {
                int32_t k_pos = static_cast<int32_t>(k_start_tile) + static_cast<int32_t>(col);
                if (col >= straddle_col) {
                    k_pos += static_cast<int32_t>(straddle_jump);
                }
                if (k_pos > q_pos) {
                    stamp_tile_range_l1_acc<dst_batch>(mask_cb, neginf_idx, out_cb, row_offset + col, 1);
                } else if (k_pos == q_pos) {
                    tile_regs_acquire();
                    copy_tile(mask_cb, diag_idx, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, out_cb, row_offset + col);
                    tile_regs_release();
                }
            }
        }
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Context for lightweight mask application.
 * All mask tiles reside in a single CB. This struct stores the pre-resolved mask metadata used when
 * lightweight masking is enabled; enablement itself is controlled by the `lightweight_mask_enabled`
 * template parameter(s), not by default-constructing this context.
 */
struct LightweightMaskContext {
    bool is_causal = false;                       // Causal masking active for this context instance
    uint32_t neginf_tile_idx = 0;                 // Index of -inf tile in the mask CB
    uint32_t causal_diag_tile_idx = 0;            // Index of causal diagonal tile in the mask CB
    uint32_t primary_diag_tile_idx = 0;           // Causal diagonal, or sliding-window trailing-primary tile
    uint32_t sliding_leading_prev_tile_idx = 0;   // Index of previous sliding-window leading tile
    uint32_t sliding_leading_tile_idx = 0;        // Index of current sliding-window leading tile in the mask CB
    uint32_t sliding_trailing_next_tile_idx = 0;  // Index of next sliding-window trailing tile
    uint32_t global_n_padded_tiles = 0;           // Fully padded K tile columns for global_n chunk
    uint32_t local_n_padded_tiles = 0;            // Fully padded K tile columns for local_n chunk
    uint32_t joint_n_padded_tiles = 0;            // Fully padded K tile columns for joint_l chunk
    uint32_t global_n_partial_col = 0;            // Column within tile where global_n padding starts (0 = no partial)
    uint32_t joint_l_partial_col = 0;             // Column within tile where joint_l padding starts (0 = no partial)
    uint32_t global_n_partial_tile_idx = 0;       // Index of global_n partial tile in the mask CB
    uint32_t joint_l_partial_tile_idx = 0;        // Index of joint_l partial tile in the mask CB
    uint32_t straddle_num_padded_tiles = 0;       // Trailing -inf tiles on straddle chunk (0 = inactive)
    uint32_t straddle_mask_chunk_id = 0;          // K chunk index where straddle mask applies

    /**
     * Resolve which mask type applies for a given K chunk and return pre-resolved params.
     * Called once per K chunk; the resolved values are then used per subblock.
     */
    void resolve_for_chunk(
        uint32_t Sk_chunk_t,
        uint32_t k_chunk,
        uint32_t num_local_k_chunks,
        bool ring_iter_needs_global_n_mask,
        bool ring_iter_needs_joint_n_mask,
        bool local_n_needs_masking,
        uint32_t global_n_mask_chunk_id,
        uint32_t local_n_mask_chunk_id,
        uint32_t joint_n_mask_chunk_id,
        uint32_t& out_num_padded,
        uint32_t& out_boundary_col,
        uint32_t& out_partial_tile_idx,
        bool& out_has_partial) const {
        out_num_padded = 0;
        out_has_partial = false;
        out_boundary_col = 0;
        out_partial_tile_idx = 0;

        if (ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) {
            out_num_padded = global_n_padded_tiles;
            out_has_partial = (global_n_partial_col > 0);
            out_boundary_col = Sk_chunk_t - out_num_padded - (out_has_partial ? 1 : 0);
            out_partial_tile_idx = global_n_partial_tile_idx;
        } else if (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) {
            out_num_padded = local_n_padded_tiles;
        } else if (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id) {
            out_num_padded = joint_n_padded_tiles;
            out_has_partial = (joint_l_partial_col > 0);
            out_boundary_col = Sk_chunk_t - out_num_padded - (out_has_partial ? 1 : 0);
            out_partial_tile_idx = joint_l_partial_tile_idx;
        } else if (straddle_num_padded_tiles > 0 && k_chunk == straddle_mask_chunk_id) {
            out_num_padded = straddle_num_padded_tiles;
        }
    }

    /**
     * Apply the lightweight mask for a given K chunk (non-streaming path).
     * When is_causal: applies causal stamp, then also applies padding stamp if this K chunk has padding.
     * When !is_causal: applies padding stamp only.
     * L1-accumulate stamps are additive, so applying both is safe (-inf + -inf = -inf, -inf + 0 = -inf).
     *
     * @tparam dst_size  DST register capacity (8 for fp16b half-sync, 4 for fp32_dest_acc).
     */
    template <uint32_t dst_size>
    void apply(
        uint32_t cb_mask_in,
        uint32_t cb_qk_im,
        uint32_t Sk_chunk_t,
        uint32_t Sq_chunk_t,
        uint32_t k_chunk,
        uint32_t num_local_k_chunks,
        bool ring_iter_needs_global_n_mask,
        bool ring_iter_needs_joint_n_mask,
        bool local_n_needs_masking,
        uint32_t global_n_mask_chunk_id,
        uint32_t local_n_mask_chunk_id,
        uint32_t joint_n_mask_chunk_id,
        uint32_t q_start_tile,
        uint32_t k_start_tile,
        uint32_t straddle_col = 0,
        uint32_t straddle_jump = 0) const {
        if (is_causal) {
            apply_causal_mask_lightweight<dst_size>(
                cb_mask_in,
                neginf_tile_idx,
                causal_diag_tile_idx,
                cb_qk_im,
                q_start_tile,
                k_start_tile,
                Sq_chunk_t,
                Sk_chunk_t,
                straddle_col,
                straddle_jump);
        }

        // Apply padding stamp (also when is_causal — the causal stamp doesn't handle K padding).
        uint32_t num_padded, boundary_col, partial_tile_idx;
        bool has_partial;
        resolve_for_chunk(
            Sk_chunk_t,
            k_chunk,
            num_local_k_chunks,
            ring_iter_needs_global_n_mask,
            ring_iter_needs_joint_n_mask,
            local_n_needs_masking,
            global_n_mask_chunk_id,
            local_n_mask_chunk_id,
            joint_n_mask_chunk_id,
            num_padded,
            boundary_col,
            partial_tile_idx,
            has_partial);

        if (has_partial) {
            apply_partial_mask_lightweight(
                cb_mask_in, partial_tile_idx, cb_qk_im, boundary_col, Sk_chunk_t, Sq_chunk_t);
        }
        if (num_padded > 0) {
            apply_padded_mask_lightweight_runtime<dst_size>(
                cb_mask_in, neginf_tile_idx, cb_qk_im, num_padded, Sk_chunk_t, Sq_chunk_t);
        }
    }
};
