// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/softplus.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/reduce_custom.h"

ALWI void sdpa_reduce_copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose = 0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, true /*transpose within 16x16 face*/, cbid)));

    MATH((llk_math_eltwise_unary_datacopy_init<
          A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false,  // is_int_fpu_en
          false   // tilize
          >(cbid)));
}

/**
 * in0_cb = max(in0_cb, in1_cb)
 */
template <uint32_t num_tiles>
void max_block_inplace(uint32_t in0, uint32_t in1) {
    // inputs come in full, outputs go out full
    copy_tile_to_dst_init_short(in0);
    copy_tile_to_dst_init_short(in1);
    max_tile_init();
    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        max_tile(dst_reg_0, dst_reg_1, static_cast<int>(VectorMode::C));
        pack_tile(dst_reg_0, in0);
        release_dst();
    }
    cb_pop_front(in0, num_tiles);
    cb_reserve_back(in0, num_tiles);
    cb_push_back(in0, num_tiles);
}

/**
 * out_cb = eltwise_max(in0, in1)
 */
template <int vector_mode = (int)VectorMode::RC>
void max_block(uint32_t in0, uint32_t in1, uint32_t out_cb, uint32_t num_tiles) {
    // inputs come in full, outputs go out full
    copy_tile_to_dst_init_short(in0);
    max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        max_tile(dst_reg_0, dst_reg_1, static_cast<int>(vector_mode));
        pack_tile(dst_reg_0, out_cb, i);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
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
    int vector_mode = static_cast<int>(VectorMode::C)>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, bool do_eltwise_max = false) {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced
    // If do_eltwise_max == true, prev_cb has rows produced.

    constexpr uint32_t num_tiles = rows * cols;

#if defined REDUCE_GRANULARITY
    constexpr uint32_t dst_tiles = (rows < REDUCE_GRANULARITY) ? rows : REDUCE_GRANULARITY;
    constexpr uint32_t granularity = (rows >= REDUCE_GRANULARITY) ? (rows >> LOG2_REDUCE_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = rows;
    constexpr uint32_t granularity = 1;
#endif

    cb_wait_front(scale_cb, 1);
    cb_reserve_back(out_cb, rows);

    const uint32_t num_tiles_to_wait = dst_tiles * cols;
    uint32_t in0_wait_tiles = num_tiles_to_wait;

    max_tile_init();

    uint32_t row_start_idx = 0;
    for (uint32_t g = 0; g < granularity; g++) {
        cb_wait_front(in0_cb, in0_wait_tiles);
        acquire_dst();

        if (do_eltwise_max) {
            cb_wait_front(prev_cb, g * dst_tiles);
            /**
             * Copy previous max values into DST register.
             * Note that this special invocation of copy_tile is necessary to produce
             * tiles in DST with transposed faces, as `reduce_block_max_row` expects.
             */
            sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
            for (uint32_t i = 0; i < dst_tiles; i++) {
                const uint32_t cur_max_dst_idx = i;
                copy_tile(prev_cb, (row_start_idx + i), cur_max_dst_idx);
            }
        }

        /**
         * For `dst_tiles` number of rows, compute the max into the even indices of the DST register.
         */
        reduce_block_max_row_init<cols>();
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t reduce_dst_idx = i;
            reduce_block_max_row<cols>(in0_cb, scale_cb, (row_start_idx + i) * cols, reduce_dst_idx);
        }
        reduce_block_max_row_uninit();

        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t cur_max_dst_idx = i;
            pack_tile<true>(cur_max_dst_idx, out_cb, (row_start_idx + i));
        }
        release_dst();

        row_start_idx += dst_tiles;
        in0_wait_tiles += num_tiles_to_wait;
    }

    cb_push_back(out_cb, rows);
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
    int vector_mode = static_cast<int>(VectorMode::C)>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, uint32_t cols, bool do_eltwise_max = false) {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1;

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst();
        reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
        }
        reduce_uninit();
        if (do_eltwise_max) {
            copy_tile_to_dst_init_short(prev_cb);
            copy_tile(prev_cb, i, prev_max_dst_idx);
            max_tile(reduce_dst_idx, prev_max_dst_idx, vector_mode);
        }

        pack_tile(reduce_dst_idx, out_cb);
        release_dst();
    }

    cb_push_back(out_cb, rows);
}

#ifdef TRISC_MATH
/**
 * recip_tile on only the columns 0:8 of a face
 */
template <bool legacy_compat = true>
void calculate_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (legacy_compat) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out = ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
            // Note: negate check removed since in always >= 0.0
            // v_if (in < 0.0)
            // {
            //     out = -out;
            // }
            // v_endif;
            if constexpr (DST_ACCUM_MODE || APPROX) {
                sfpi::dst_reg[0] = out;
            } else {
                sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
            }
            sfpi::dst_reg += 2;
        }
    } else {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];

            if constexpr (APPROX) {
                sfpi::dst_reg[0] = ckernel::sfpu::_sfpu_reciprocal_<0>(in);
            } else {
                if constexpr (DST_ACCUM_MODE) {
                    sfpi::dst_reg[0] = ckernel::sfpu::_sfpu_reciprocal_<2>(in);
                } else {
                    sfpi::vFloat out = ckernel::sfpu::_sfpu_reciprocal_<1>(in);
                    sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
                }
            }

            sfpi::dst_reg += 2;
        }
    }
}

template <bool legacy_compat = true>
void recip_tile_first_column(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX /*APPROXIMATE*/>(
        calculate_recip_first_column<legacy_compat>, idst, (int)VectorMode::C);
}
#endif

/**
 * in_cb = 1 / in_cb
 */
void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(in_cb);

    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, i, 0);
        MATH((recip_tile_first_column(0)));
        pack_tile(0, in_cb);
        release_dst();
    }
    cb_pop_front(in_cb, num_tiles);
    cb_reserve_back(in_cb, num_tiles);
    cb_push_back(in_cb, num_tiles);
}

/**
 * in0_cb = exp((in0_cb - in1_cb) * scale_fp32)
 */
template <
    uint32_t in0_cb,
    uint32_t rows,
    uint32_t scale_fp32,
    bool write_result_inplace = true,
    int vector_mode = (int)VectorMode::RC>
void sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced
    sub_bcast_cols_init_short(in0_cb, in1_cb);

    exp_tile_init<true, true, scale_fp32>();
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);
    cb_reserve_back(reduce_cb, rows);

    if constexpr (write_result_inplace) {
        cb_pop_front(in0_cb, rows * cols);
        cb_reserve_back(in0_cb, rows * cols);
    }

#ifdef SUB_EXP_GRANULARITY
    uint32_t dst_tiles = (cols < SUB_EXP_GRANULARITY) ? cols : SUB_EXP_GRANULARITY;
    uint32_t granularity = (cols >= SUB_EXP_GRANULARITY) ? (cols >> LOG2_SUB_EXP_GRANULARITY) : 1;
#else
    uint32_t dst_tiles = cols;
    uint32_t granularity = 1;
#endif

    uint32_t in0_index = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                exp_tile<true, true>(j, vector_mode);
                in0_index++;
            }
            tile_regs_commit();
            tile_regs_wait();

            if constexpr (write_result_inplace) {
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, in0_cb);
                }
            }

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
            tile_regs_release();
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
        if constexpr (write_result_inplace) {
            // Granular write output to enable following matmul unpack to start early.
            cb_push_back(in0_cb, cols);
        }
    }
    cb_push_back(reduce_cb, rows);
}

/**
 * out_cb = in0_cb * in1_cb
 */
template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, bool pack_accumulate = false) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Precondition: out_cb has rows*cols produced
    // Postcondition: in0_cb empty
    // Postcondition: in1_cb empty
    // Postcondition: out_cb has rows*cols produced

    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t dst_tiles = DHT_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_DHT_GRANULARITY;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    PACK((llk_pack_reconfig_l1_acc(pack_accumulate)));
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    if (!pack_accumulate) {
        cb_reserve_back(out_cb, num_tiles);
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
    PACK((llk_pack_reconfig_l1_acc(false)));
    cb_pop_front(in1_cb, rows);
    cb_pop_front(in0_cb, num_tiles);
    if (!pack_accumulate) {
        cb_push_back(out_cb, num_tiles);
    } else {
        cb_pop_front(out_cb, num_tiles);
        cb_reserve_back(out_cb, num_tiles);
        cb_push_back(out_cb, num_tiles);
    }
}

/**
 * in0_cb *= in1_cb
 */
template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t dst_tiles = DHT_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_DHT_GRANULARITY;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; ++u) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                mul_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
            }
            tile_regs_commit();
            cb_pop_front(in0_cb, dst_tiles);
            cb_reserve_back(in0_cb, dst_tiles);
            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_cb);
            }
            cb_push_back(in0_cb, dst_tiles);
            tile_regs_release();
        }
    }
    cb_pop_front(in1_cb, rows);
}

template <uint32_t in1_scalar_cb, uint32_t num_tiles>
void mul_block_bcast_scalar_inplace(uint32_t in0_cb) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced

    constexpr uint32_t dst_tiles = STATS_GRANULARITY;
    constexpr uint32_t granularity = num_tiles >> LOG2_STATS_GRANULARITY;
    reconfig_data_format(in0_cb, in1_scalar_cb);
    mul_tiles_bcast_scalar_init_short(in0_cb, in1_scalar_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_scalar_cb, 1);
    uint32_t in0_index = 0;
    for (uint32_t g = 0; g < granularity; ++g) {
        acquire_dst();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, in0_index, 0, i);
            in0_index++;
        }
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_cb);
        }
        release_dst();
    }
    cb_pop_front(in0_cb, num_tiles);
    cb_reserve_back(in0_cb, num_tiles);
    cb_push_back(in0_cb, num_tiles);
}

/**
 * in0_cb += in1_cb
 */
template <bool pop_in1 = true>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, in0_cb);
        release_dst();
    }

    cb_pop_front(in0_cb, num_tiles);
    if (pop_in1) {
        cb_pop_front(in1_cb, num_tiles);
    }
    cb_reserve_back(in0_cb, num_tiles);
    cb_push_back(in0_cb, num_tiles);
}

void mul_tiles_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    /**
     * Given in0_cb and in1_cb, multiply each tile of in0_cb by the corresponding tile of in1_cb
     * and bcast cols of in1_cb.
     */
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
}

/**
 * in0_cb *= in1_cb
 */
void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        acquire_dst();
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
}

#ifdef TRISC_MATH
/**
 * exp_tile on only the columns 0:8 of a face
 */
template <bool SDPA_EXP_APPROX_MODE>
void calculate_exponential_first_column(int scale_bf16) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (SDPA_EXP_APPROX_MODE) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = ckernel::sfpu::
                _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                    val, scale_bf16);
            sfpi::dst_reg[0] = result;

            // Stride by 2 to skip columns 8:16 of the face
            sfpi::dst_reg += 2;
        }
    } else {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            val = val * sfpi::s2vFloat16b(scale_bf16);
            sfpi::vFloat result = ckernel::sfpu::_sfpu_exp_improved_<DST_ACCUM_MODE>(val);
            sfpi::dst_reg[0] = result;

            // Stride by 2 to skip columns 8:16 of the face
            sfpi::dst_reg += 2;
        }
    }
}

template <bool SDPA_EXP_APPROX_MODE>
void exp_tile_first_column(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_exponential_first_column<SDPA_EXP_APPROX_MODE>, idst, (int)VectorMode::C, scale_bf16);
}
#endif

/**
 * out_cb = exp((in0_cb - in1_cb) * scale_fp32)
 */
template <uint32_t scale_fp32>
void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in0_cb and in1_cb has num_tiles produced

    sub_tiles_init(in0_cb, in1_cb);
    exp_tile_init<EXP_APPROX_MODE, false>();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    // Convert scale_fp32 to bf16 scale
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        acquire_dst();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        MATH((exp_tile_first_column<EXP_APPROX_MODE>(0, scale_bf16)));
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
}

#ifdef TRISC_MATH
/**
 * The custom SFPI LLK function computes the following operation:
 * cur_max = max(prev_max, worker_max)
 * cur_sum = exp((worker_max - cur_max) * scale) * worker_sum + exp((prev_max - cur_max) * scale) * prev_sum
 * There are 4 results produced:
 * 1. exp_max_diff = exp((worker_max - cur_max) * scale), produced in dst_reg[prev_max_base_idx]
 * 2. exp_max_diff_2 = exp((prev_max - cur_max) * scale), produced in dst_reg[worker_max_base_idx]
 * 3. cur_sum produced in dst_reg[prev_sum_base_idx]
 * 4. cur_max produced in dst_reg[cur_max_base_idx]
 * fused_max_sub_exp_add_tile
 */
template <bool SDPA_EXP_APPROX_MODE>
void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    constexpr uint32_t prev_max_base_idx = 0;      // dst_reg_0 (Tile 0)
    constexpr uint32_t worker_max_base_idx = 32;   // dst_reg_1 (Tile 1)
    constexpr uint32_t cur_max_base_idx = 64;      // dst_reg_2 (Tile 2)
    constexpr uint32_t prev_sum_base_idx = 96;     // dst_reg_3 (Tile 3)
    constexpr uint32_t worker_sum_base_idx = 128;  // dst_reg_4 (Tile 4)

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        // Load inputs for this vector-slot into temporaries to avoid aliasing on dst_reg
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];
        v_if(prev_max_vec < worker_max_vec) { sfpi::dst_reg[cur_max_base_idx] = worker_max_vec; }
        v_else { sfpi::dst_reg[cur_max_base_idx] = prev_max_vec; }
        v_endif;
        sfpi::vFloat cur_max = sfpi::dst_reg[cur_max_base_idx];

        // Compute differences
        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        // Exponentials of differences
        sfpi::vFloat exp_prev = ckernel::sfpu::
            _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker = ckernel::sfpu::
            _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_worker, scale_bf16);

        // Store exponentials for optional debug/pack-out
        sfpi::dst_reg[prev_max_base_idx] = exp_prev;
        sfpi::dst_reg[worker_max_base_idx] = exp_worker;

        // cur_sum = exp(worker_max - cur_max) * worker_sum + exp(prev_max - cur_max) * prev_sum
        sfpi::dst_reg[worker_sum_base_idx] = exp_worker * worker_sum_vec;
        sfpi::dst_reg[prev_sum_base_idx] = exp_prev * prev_sum_vec;
        sfpi::vFloat corr_worker_sum = sfpi::dst_reg[worker_sum_base_idx];
        sfpi::vFloat corr_prev_sum = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat corr_sum = corr_worker_sum + corr_prev_sum;
        sfpi::dst_reg[prev_sum_base_idx] = corr_sum;
        sfpi::dst_reg += 2;
    }
}

template <bool SDPA_EXP_APPROX_MODE>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE>, idst, (int)VectorMode::C, scale_bf16);
}
#endif

template <uint32_t scale_fp32, int vector_mode = (int)VectorMode::C>
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
    cb_wait_front(cb_worker_max, num_head_tiles);
    cb_wait_front(cb_worker_sum, num_head_tiles);
    cb_wait_front(cb_prev_max, num_head_tiles);
    cb_wait_front(cb_prev_sum, num_head_tiles);

    cb_reserve_back(cb_cur_max, num_head_tiles);
    cb_reserve_back(cb_cur_sum, num_head_tiles);
    cb_reserve_back(cb_exp_max_diff, num_head_tiles);
    cb_reserve_back(cb_exp_max_diff_2, num_head_tiles);

    constexpr uint32_t dst_reg_0 = 0;  // dst_reg_0 is used for prev_max
    constexpr uint32_t dst_reg_1 = 1;  // dst_reg_1 is used for worker_max
    constexpr uint32_t dst_reg_2 = 2;  // dst_reg_2 is used for cur_max
    constexpr uint32_t dst_reg_3 = 3;  // dst_reg_3 is used for prev_sum, returns cur_sum
    constexpr uint32_t dst_reg_4 = 4;  // dst_reg_4 is used for worker_sum

    // convert scale from fp32 to bf16
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_head_tiles; i++) {
        acquire_dst();
        copy_tile_to_dst_init_short(cb_worker_max);
        exp_tile_init<EXP_APPROX_MODE, false>();
        max_tile_init();
        copy_tile(cb_prev_max, i, dst_reg_0);
        copy_tile(cb_worker_max, i, dst_reg_1);
        copy_tile(cb_prev_sum, i, dst_reg_3);
        copy_tile(cb_worker_sum, i, dst_reg_4);
        MATH((fused_max_sub_exp_add_tile<EXP_APPROX_MODE>(0, scale_bf16)));
        pack_tile(dst_reg_0, cb_exp_max_diff);
        pack_tile(dst_reg_1, cb_exp_max_diff_2);
        pack_tile(dst_reg_2, cb_cur_max);
        pack_tile(dst_reg_3, cb_cur_sum);
        cb_push_back(cb_cur_max, 1);
        cb_push_back(cb_cur_sum, 1);
        cb_push_back(cb_exp_max_diff, 1);
        cb_push_back(cb_exp_max_diff_2, 1);
        release_dst();
    }
    cb_pop_front(cb_prev_sum, num_head_tiles);
    cb_pop_front(cb_worker_sum, num_head_tiles);
}

/**
 * in_cb -> out_cb
 */
template <bool pop_in_cb>
void move_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced

    copy_tile_to_dst_init_short(in_cb);

    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0 /*dst*/);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
    if (pop_in_cb) {
        cb_pop_front(in_cb, num_tiles);
    }
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0 /*dst*/);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
    cb_pop_front(in_cb, num_tiles);
}

void log_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    log_tile_init();
    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0 /*dst*/);
        log_tile(0);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
}

void sigmoid_sub(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // out_cb = sigmoid(in0_cb - in1_cb)
    /**
     * sigmoid(x) is accurately implemented as 1 / (1 + exp(-x))
     * This function manually implements the composite, accurate sigmoid.
     *
     * Each input tile has only the first column containing valid data, so VectorMode::C is a useful optimization.
     */
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    sub_tiles_init(in0_cb, in1_cb);
    exp_tile_init<false, false>();
    // recip_tile_init<false>(); // Can omit this because accurate exp_tile_init performs reduce_tile_init

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        // exp_tile<false, false, true /*SCALE_EN*/>(0, (int)VectorMode::C, (uint16_t)0xBF80 /*bf16(-1.0) scale*/);
        MATH((exp_tile_first_column<false /*APPROX_MODE*/>(0, (uint16_t)0xBF80 /*bf16(-1.0) scale*/)));
        // add_unary_tile(0, 0x3F800000); // Call the LLK directly to get access to VectorMode argument
        MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar<APPROX, ADD_UNARY>(0, 0x3F800000, (int)VectorMode::C)));
        // recip_tile<false>(0, (int)VectorMode::C);
        MATH((recip_tile_first_column<false>(0)));
        pack_tile(0, out_cb);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
}

#ifdef TRISC_MATH
/**
 * softplus_tile on only the columns 0:8 of a face
 */
template <bool SDPA_EXP_APPROX_MODE>
void calculate_softplus_first_column(uint param0, uint param1, uint param2) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    vFloat beta = ckernel::sfpu::Converter::as_float(param0);
    vFloat beta_reciprocal = ckernel::sfpu::Converter::as_float(param1);
    vFloat threshold = ckernel::sfpu::Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        ckernel::sfpu::calculate_softplus_body<APPROX>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg += 2;
    }
}

void softplus_tile_first_column(uint32_t idst, uint beta, uint beta_reciprocal, uint threshold) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX /*APPROXIMATE*/>(
        calculate_softplus_first_column<APPROX>, idst, (int)VectorMode::C, beta, beta_reciprocal, threshold);
}
#endif

void logsigmoid_sub(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // out_cb = logsigmoid(in0_cb - in1_cb)
    // Implemented as softplus for numerical stability. logsigmoid(x) = -softplus(-x)
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    sub_tiles_init(in0_cb, in1_cb);
    softplus_tile_init();
    constexpr uint32_t const_1_fp32 = 0x3F800000;
    constexpr uint32_t const_20_fp32 = 0x41A00000;

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
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
        pack_tile(0, out_cb);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
}

void sub_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // out_cb = in0_cb - in1_cb

    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    sub_tiles_init(in0_cb, in1_cb);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, out_cb);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
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
    const uint32_t& num_blocks,
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

    mm_block_init_short(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    const uint32_t output_num_tiles = M * N;
    const uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    const uint32_t in0_subblock_all_cols_num_tiles = subblock_h * N;

    uint32_t in0_index_offset = 0;

    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;

    reconfig_data_format(in1_cb, in0_cb);
    cb_wait_front(in1_cb, K * N);
    cb_reserve_back(out_cb, output_num_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        cb_wait_front(in0_cb, in0_wait_tiles);
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
                cb_wait_front(mask_cb, out_subblock_num_tiles);
                cb_wait_front(zero_cb, 1);
                add_tiles_init(zero_cb, mask_cb, true);
                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                    add_tiles(zero_cb, mask_cb, 0, i, i);
                }
                cb_pop_front(mask_cb, out_subblock_num_tiles);
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
        cb_push_back(out_cb, in0_subblock_all_cols_num_tiles);
    }
    cb_pop_front(in1_cb, K * N);
}

template <uint32_t M>
void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb) {
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced

    constexpr uint32_t N = 1;  // Result of reduce is 1 column
    constexpr uint32_t in0_block_w = N;
    constexpr uint32_t subblock_w = N;
    // Reuse the Sq_chunk_t granularity chosen for sub_exp_block
    constexpr uint32_t subblock_h = STATS_GRANULARITY;
    constexpr uint32_t in0_num_subblocks = M >> LOG2_STATS_GRANULARITY;

    /**
     * Use matmul on Mx1 input to reduce rows within tile to produce Mx1 output.
     */

    mm_block_init_short(
        out_cb, in1_cb, 0 /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    constexpr uint32_t output_num_tiles = M * N;
    constexpr uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    reconfig_data_format(in1_cb, out_cb);
    cb_wait_front(in1_cb, N);
    cb_wait_front(out_cb, M);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        tile_regs_acquire();

        uint32_t dst_index = 0;
        uint32_t in0_index = 0;
        uint32_t in1_index = 0;

        matmul_block(out_cb, in1_cb, in0_index, in1_index, dst_index, 0, subblock_w, subblock_h, in0_block_w);

        tile_regs_commit();
        cb_pop_front(out_cb, subblock_h);

        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_h; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_push_back(out_cb, subblock_h);
    }
}

enum SDPAType {
    STANDARD = 0,
    JOINT = 1,
    RING = 2,
};

/**
 *
 */
template <
    SDPAType sdpa_type,
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_attention_sink,
    uint32_t cb_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    bool use_attention_sink,
    bool is_causal,
    bool use_provided_mask,
    bool use_padded_mask,
    bool use_joint_mask,
    bool is_chunked,
    uint32_t scale_fp32,
    uint32_t sliding_window_size>
void sdpa_inner_loop(
    const uint32_t Skt,
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t qk_in0_num_subblocks,
    const uint32_t qk_in1_num_subblocks,
    const uint32_t qk_num_blocks,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in0_num_subblocks,
    const uint32_t out_in1_num_subblocks,
    const uint32_t out_num_blocks,
    const uint32_t iter_q_start,
    const uint32_t iter_q_end,
    const uint32_t q_num_chunks,
    const uint32_t local_q_start,
    const uint32_t chunked_q_chunk_offset,
    const uint32_t iter_k_chunk_start,
    const uint32_t iter_k_chunk_end,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t mask_chunk_0,
    const uint32_t mask_chunk_1,
    const uint32_t ring_iter,
    const uint32_t ring_id,
    const uint32_t N_mask_ring_id,
    const uint32_t L_mask_ring_id,
    const uint32_t global_logical_NK_chunks,
    const uint32_t global_padded_NK_chunks,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_lse_in,
    const uint32_t cb_lse_out,
    const uint32_t cb_prev_out,
    const uint32_t cb_out) {
    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {
        uint32_t q_low_idx;
        uint32_t q_high_idx;
        if constexpr (sdpa_type == STANDARD) {
            uint32_t q_chunk;
#if defined BALANCED_Q_PARALLEL
            uint32_t q_chunk_div_2 = iter_q_end / 2;  // q_chunks_per_core / 2.
            if (q_iter < q_chunk_div_2) {             // bottom half
                q_chunk = local_q_start + q_iter;
            } else {
                uint32_t back_q_iter = q_iter - q_chunk_div_2;  // Back half should start at 0
                q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
            }
#else
            q_chunk = local_q_start + q_iter;
#endif
            // Get Q chunk
            if constexpr (is_chunked) {
                q_chunk = chunked_q_chunk_offset + q_chunk;
            }
            q_low_idx = q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
            if constexpr (is_causal) {
                q_high_idx = q_low_idx + Sq_chunk_t;
            } else {
                q_high_idx = Skt;
            }
        }

        // Set up ping pong buffers
        uint32_t alias_prev_sum = cb_sum_A;
        uint32_t alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A;
        uint32_t alias_cur_max = cb_max_B;
        uint32_t alias_mm2_prev_out = cb_out_im_A;
        uint32_t alias_mm2_cur_out = cb_out_im_B;

        uint32_t k_chunk_end;
        if constexpr (sdpa_type == STANDARD) {
            // loop while k_low < q_high => (k_chunk * Sk_chunk_t) < q_high_idx.
            k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;
        } else {  // RING or JOINT.
            k_chunk_end = iter_k_chunk_end;
        }

        for (uint32_t k_chunk = iter_k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            if constexpr (sdpa_type == RING) {
                if (k_chunk >= global_logical_NK_chunks && k_chunk < global_padded_NK_chunks) {
                    // This is a KV chunk on spatial input beyond the chunk-padded length of the spatial input.
                    // If k_chunk >= global_padded_NK_chunks, then this is a joint KV chunk.
                    continue;
                }
            }

            /**
             * QK = Q_CHUNK @ K_CHUNK
             *
             * matmul_blocks internally waits on both inputs
             */
            pack_reconfig_data_format(cb_qk_im);
            matmul_blocks(
                cb_q_in,
                cb_k_in,
                cb_qk_im,
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                qk_num_blocks,
                qk_in0_num_subblocks,
                qk_in1_num_subblocks,
                qk_in0_block_w,
                qk_subblock_h,
                qk_subblock_w,
                true /*transpose*/);

            /**
             * Note
             * Typically, scores is multiplied by a scalar here. We employed an optimization
             * where we fuse the scaling into exp both in exp(x - max) and exp(prev_max - cur_max).
             * This gives us scaling for free on the performance-critical exp(x - max) computation.
             */

            // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
            // Q-range = [q_low, q_high)
            // K-range = [k_low, k_high)
            // does_overlap = not (q_low >= k_high or k_low >= q_high)
            // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
            if constexpr (is_causal || sliding_window_size > 0) {
                const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
                /* QK += MASK */
                if (!(q_low_idx >= k_high_idx) || sliding_window_size > 0) {
                    // If no sliding window - simple causal case - only apply along the diagonal
                    // Otherwise, apply mask for all chunks
                    reconfig_data_format(cb_qk_im, cb_mask_in);
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                }
            } else if constexpr (use_provided_mask) {
                /* QK += MASK */
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
            } else if constexpr (use_padded_mask) {
                // only uses mask on the last K chunk if it exists at all
                if (k_chunk == iter_k_chunk_end - 1) {
                    /* QK += MASK */
                    reconfig_data_format(cb_qk_im, cb_mask_in);
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                }
            } else if constexpr (use_joint_mask) {
                if ((ring_id == N_mask_ring_id && k_chunk == mask_chunk_0) ||
                    (ring_id == L_mask_ring_id && k_chunk == mask_chunk_1)) {
                    /* QK += MASK */
                    reconfig_data_format(cb_qk_im, cb_mask_in);
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                }
            }

            /**
             * reduce_c can perform both reduce_max and eltwise max with previous result.
             * if do_eltwise_max:
             *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
             * else:
             *  cur_max = max(qk, dim=-1)
             */
            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t>(
                alias_cur_max, alias_prev_max, Sk_chunk_t, k_chunk > iter_k_chunk_start);
            /**
             * sub_exp fuses a few operations.
             * In-place it performs `QK = exp((QK - cur_max) * scale)`
             *
             * It also partially performs reduce_sum on the output using L1 accumulation.
             * `cur_sum = sum_tiles(exp((QK - cur_max) * scale), dim=-1)`
             *
             * Partial reduce_sum is used to push the final row_reduction within a tile
             * outside of the loop over K chunks.
             */
            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true>(
                alias_cur_max, alias_cur_sum, Sk_chunk_t);

            /* OUT_IM = QK @ V_CHUNK */
            matmul_blocks(
                cb_qk_im,
                cb_v_in,
                alias_mm2_cur_out,
                Sq_chunk_t,
                vDHt,
                Sk_chunk_t,
                out_num_blocks,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_in0_block_w,
                out_subblock_h,
                out_subblock_w,
                false /*transpose*/);

            cb_pop_front(cb_qk_im, qk_chunk_tiles);
            reconfig_data_format(alias_prev_max, alias_cur_max);

            /* OUT_ACC += OUT_IM */
            if (k_chunk > iter_k_chunk_start) {
                /**
                 * cb_exp_max_diff = torch.exp((cb_prev_max - cb_cur_max) * scale)
                 * Scale is fused into exp again since max is the max of unscaled scores.
                 */

                sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
                cb_pop_front(alias_prev_max, Sq_chunk_t);

                /**
                 * cb_prev_sum *= cb_exp_max_diff
                 * This is a bcast_cols since max_diff is a column vector and prev_sum is a partial
                 * reduction, containing the sum of tiles in dim=-1 of QK.
                 */
                mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                /* cb_cur_sum += cb_prev_sum */
                add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

                /**
                 * alias_mm2_cur_out += alias_mm2_prev_out * cb_exp_max_diff
                 * This uses L1 accumulation to accumulate onto mm2_cur_out.
                 */
                mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, true);
            }

            // Swap CB handles to prepare for next iteration
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
            std::swap(alias_prev_max, alias_cur_max);
        }
        /**
         * Performs final row-reduction on the partial sum.
         */
        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);

        /**
         * Process attention sink as a virtual K chunk.
         * The attention sink provides additional logits that are included in the softmax
         * denominator but don't contribute to the output (no S @ V computation).
         * This effectively allows some attention probability to be "absorbed" by the sink,
         * reducing attention weights on actual tokens.
         *
         * Shape of attention_sink: [Sq_chunk_t, 1] tiles
         * Each head has one sink logit value that is broadcast to all query positions in the chunk.
         * The reader kernel replicates the per-head value across all Sq_chunk_t positions.
         */
        if constexpr (use_attention_sink) {
            // Treat attention_sink as scores (already scaled)
            // Shape: [Sq_chunk_t, 1] tiles - same per-head sink value broadcast to all query positions

            // 1. Update running max: cur_max = max(prev_max, attention_sink)
            //    This compares the previous max with the sink logit
            reconfig_data_format(cb_attention_sink, cb_identity_scale_in);

            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_attention_sink, cb_identity_scale_in, Sq_chunk_t>(
                alias_cur_max, alias_prev_max, 1, true);

            // 2. Compute exp((prev_max - cur_max) * scale) to rescale previous statistics
            sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
            cb_pop_front(alias_prev_max, Sq_chunk_t);

            // 3. Rescale previous sum: prev_sum *= exp(prev_max - cur_max)
            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
            // 4. Compute exp((attention_sink - cur_max) * scale) and accumulate in cur_sum
            //    This adds the attention sink's contribution to the softmax denominator
            sub_exp_block_bcast_cols_inplace<cb_attention_sink, Sq_chunk_t, scale_fp32, false>(
                alias_cur_max, alias_cur_sum, 1);

            // 5. Add rescaled previous sum to current sum: cur_sum += prev_sum
            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

            // 6. Update running statistics for final normalization
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_prev_max, alias_cur_max);

            // 7. Rescale accumulated output: mm2_prev_out *= exp(prev_max - cur_max)
            //    Note: We do NOT compute attention_sink @ V, so output only has real token contributions
            //    But we need to rescale it due to the updated max
            mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, false);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
        }

        if constexpr (sdpa_type == RING) {
            log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);

            // Scale prev_max by scale_fp32
            mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);
            add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t);

            /* cb_cur_sum = 1.0 / cb_cur_sum */
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);
            /* cb_out_accumulate_im *= cb_cur_sum */
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum);
            if (ring_iter > 0) {
                // Update output according to previous and current LSE
                /**
                 * sig = torch.sigmoid(cur_lse - prev_lse)
                 * out = prev_out - sig * (prev_out - cur_out)
                 * lse = prev_lse - torch.logsigmoid(prev_lse - cur_lse)
                 */
                cb_wait_front(cb_lse_in, Sq_chunk_t);
                cb_wait_front(cb_prev_out, out_chunk_tiles);

                uint32_t alias_cur_lse = alias_prev_max;      // full
                uint32_t alias_sig = alias_cur_max;           // empty
                uint32_t alias_cur_out = alias_mm2_prev_out;  // full
                uint32_t alias_sub = alias_mm2_cur_out;       // empty

                // alias_sig = sigmoid(alias_cur_lse - cb_lse_in)
                sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);

                // alias_sub = cb_prev_out - alias_cur_out
                sub_block(cb_prev_out, alias_cur_out, alias_sub, out_chunk_tiles);
                // alias_sub *= alias_sig
                mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_sub, alias_sig);
                // cb_out = cb_prev_out - alias_sub
                sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
                cb_pop_front(cb_prev_out, out_chunk_tiles);
                cb_pop_front(alias_cur_out, out_chunk_tiles);
                cb_pop_front(alias_sub, out_chunk_tiles);

                // alias_sig = sigmoid(cb_lse_in - alias_cur_lse)
                // alias_cur_lse = log(alias_sig)
                // cb_lse_out = cb_lse_in - alias_cur_lse
                logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
                sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
                cb_pop_front(alias_sig, Sq_chunk_t);
                cb_pop_front(alias_cur_lse, Sq_chunk_t);
                cb_pop_front(cb_lse_in, Sq_chunk_t);
            } else {
                pack_reconfig_data_format(cb_out);
                copy_block(alias_mm2_prev_out, cb_out, out_chunk_tiles);

                copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
            }
        } else {
            /* cb_cur_sum = 1.0 / cb_cur_sum */
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);

            /* cb_out_accumulate_im *= cb_cur_sum */
            pack_reconfig_data_format(cb_out);
            mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum, cb_out, false);

            // free up cb_prev_max after K chunks
            cb_pop_front(alias_prev_max, Sq_chunk_t);
        }

        cb_pop_front(cb_q_in, q_chunk_tiles);
    }

    if constexpr (use_attention_sink) {
        cb_pop_front(cb_attention_sink, Sq_chunk_t);
    }
}
