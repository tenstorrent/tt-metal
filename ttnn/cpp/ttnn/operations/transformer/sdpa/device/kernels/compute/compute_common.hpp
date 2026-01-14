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

template <PoolType pool_type, ReduceDim reduce_dim, uint32_t in0_cb, uint32_t scale_cb, uint32_t rows, uint32_t cols>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, bool do_eltwise_max = false) {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    constexpr uint32_t num_tiles = rows * cols;

    constexpr uint32_t dst_tiles = (rows < REDUCE_GRANULARITY) ? rows : REDUCE_GRANULARITY;
    constexpr uint32_t granularity = (rows >= REDUCE_GRANULARITY) ? (rows >> LOG2_REDUCE_GRANULARITY) : 1;

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
        // recip_tile(0, static_cast<int>(VectorMode::C));
        MATH((recip_tile_first_column(0)));
        pack_tile(0, in_cb);
        release_dst();
    }
    cb_pop_front(in_cb, num_tiles);
    cb_reserve_back(in_cb, num_tiles);
    cb_push_back(in_cb, num_tiles);
}

template <uint32_t in0_cb, uint32_t rows, uint32_t cols, uint32_t scale_fp32, bool write_result_inplace = true>
void sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb) {
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

    constexpr uint32_t dst_tiles = (cols < SUB_EXP_GRANULARITY) ? cols : SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = (cols >= SUB_EXP_GRANULARITY) ? (cols >> LOG2_SUB_EXP_GRANULARITY) : 1;
    uint32_t in0_index = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                exp_tile<true, true>(j);
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

    cb_pop_front(in1_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
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
            sfpi::vFloat result;
            if constexpr (!DST_ACCUM_MODE) {
                // bfloat16-accurate implementation of exp ( < 1 ULP)
                result = ckernel::sfpu::_sfpu_exp_21f_<false>(val);
            } else {
                // float32 version of exp (< 150 float32 ULP)
                // this is more accurate than exp_21f, but also slower
                result = ckernel::sfpu::_sfpu_exp_61f_(val);
            }

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
        acquire_dst();

        sub_tiles(in0_cb, in1_cb, i, i, 0);

        // exp_tile<EXP_APPROX_MODE, false, true, true>(0, static_cast<int>(VectorMode::C), scale_bf16);
        MATH((exp_tile_first_column<EXP_APPROX_MODE>(0, scale_bf16)));

        pack_tile(0, out_cb);

        cb_push_back(out_cb, 1);
        release_dst();
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
    float beta = ckernel::sfpu::Converter::as_float(param0);
    float beta_reciprocal = ckernel::sfpu::Converter::as_float(param1);
    float threshold = ckernel::sfpu::Converter::as_float(param2);
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

void matmul_blocks(
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
    const bool& transpose) {
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
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
