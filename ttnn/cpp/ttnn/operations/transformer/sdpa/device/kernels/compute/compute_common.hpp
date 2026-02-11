// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/debug/assert.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/binary_max_min.h"
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
#include "api/compute/experimental/reduce_custom.h"

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
    binary_max_tile_init();
    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_0, static_cast<int>(VectorMode::C));
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
    binary_max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_0, static_cast<int>(VectorMode::C));
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
    constexpr uint32_t granularity = (rows >= REDUCE_GRANULARITY) ? (rows / REDUCE_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = rows;
#endif

    cb_wait_front(scale_cb, 1);
    cb_reserve_back(out_cb, rows);

    const uint32_t num_tiles_to_wait = dst_tiles * cols;
    uint32_t in0_wait_tiles = num_tiles_to_wait;

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
        reduce_block_max_row_uninit(in0_cb);

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

    binary_max_tile_init();
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
            binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx, static_cast<int>(vector_mode));
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
    bool do_reduce = true,
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
    if constexpr (do_reduce) {
        cb_reserve_back(reduce_cb, rows);
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
                exp_tile<true, true>(j, vector_mode);
            }
            tile_regs_commit();

            if constexpr (write_result_inplace) {
                cb_pop_front(in0_cb, dst_tiles);
                cb_reserve_back(in0_cb, dst_tiles);
            }

            tile_regs_wait();

            if constexpr (write_result_inplace) {
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, in0_cb);
                }
                // Granular write output to enable following matmul unpack to start early.
                cb_push_back(in0_cb, dst_tiles);
            }

            if constexpr (do_reduce) {
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
        cb_push_back(reduce_cb, rows);
    }
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
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Precondition: out_cb has rows*cols produced
    // Postcondition: in0_cb empty
    // Postcondition: in1_cb empty
    // Postcondition: out_cb has rows*cols produced

    constexpr uint32_t num_tiles = rows * cols;

    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);

    if constexpr (immediate_pop) {
        static_assert(!pack_accumulate, "Unsupported parameter configuration");
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                acquire_dst();
                mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
                cb_pop_front(in0_cb, 1);
                cb_reserve_back(out_cb, 1);
                pack_tile(0, out_cb);
                cb_push_back(out_cb, 1);
                release_dst();
            }
        }
        cb_pop_front(in1_cb, rows);
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
        cb_pop_front(in1_cb, rows);
        cb_pop_front(in0_cb, num_tiles);
        if (pack_accumulate) {
            PACK((llk_pack_reconfig_l1_acc(false)));
            cb_pop_front(out_cb, num_tiles);
            cb_reserve_back(out_cb, num_tiles);
            cb_push_back(out_cb, num_tiles);
        } else {
            cb_push_back(out_cb, num_tiles);
        }
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

#ifdef DHT_GRANULARITY
    constexpr uint32_t dst_tiles = (cols < DHT_GRANULARITY) ? cols : DHT_GRANULARITY;
    constexpr uint32_t granularity = (cols >= DHT_GRANULARITY) ? (cols / DHT_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = cols;
#endif

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

#ifdef STATS_GRANULARITY
    constexpr uint32_t dst_tiles = STATS_GRANULARITY;
    constexpr uint32_t granularity = num_tiles / STATS_GRANULARITY;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = num_tiles;
#endif

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

constexpr auto bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) & 0xFFFFu); };
constexpr auto hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) >> 16); };

#ifdef ARCH_WORMHOLE
#define ADDR_MOD_X ADDR_MOD_3
#else
#define ADDR_MOD_X ADDR_MOD_7
#endif

ALWI void INSERT_SFPNOP() {
#ifdef ARCH_WORMHOLE
    TTI_SFPNOP;
#endif
}

template <bool USE_SFPARECIP_INSTR, int POLY_DEGREE>
constexpr bool can_preload_ln2_constants() {
#ifdef ARCH_WORMHOLE
    return false;
#else
    return (USE_SFPARECIP_INSTR || POLY_DEGREE == 1 || POLY_DEGREE == 2);
#endif
}

/**
 * Computes exp(x) using polynomial approximation after range reduction.
 *
 * Scales by configured factor, then reduces to exp(r) * 2^k
 * where r = x - k*ln(2). Uses either SFPARECIP instruction or multi-term polynomial (degree 1-4)
 * to compute exp(r), then reconstructs full result via exponent manipulation,
 * clamping the exponent to handle large positive or negative inputs.
 *
 * @tparam USE_SFPARECIP_INSTR Use hardware SFPARECIP instruction (true) or polynomial evaluation (false). Only
 * supported on Blackhole.
 * @tparam SCALE_EN Apply scaling factor from LREG to input values
 * @tparam ITERATIONS Number of 32-element vectors to process per tile
 * @tparam POLY_DEGREE Polynomial degree (1-4) when USE_SFPARECIP_INSTR=false; higher improves accuracy
 * @tparam IS_FP32_DEST_ACC_EN Float32 accumulation to dest register enabled.
 * @tparam SCALE_BF16 Bfloat16 scale factor represented as uint16_t.
 */
template <
    bool SCALE_EN,
    int ITERATIONS,
    bool USE_SFPARECIP_INSTR,
    int POLY_DEGREE,
    bool IS_FP32_DEST_ACC_EN,
    uint16_t SCALE_BF16>
void calculate_exponential_polynomial() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    constexpr float LN2_RECIP = 1.44269504088896340736f;  // 1/ln(2)
    constexpr float M_LN2 = -0.69314718055994530942f;     // -ln(2)

    if constexpr (!USE_SFPARECIP_INSTR) {
        static_assert(POLY_DEGREE >= 1 && POLY_DEGREE <= 4);

        // Evaluate polynomial f(x) = c0 + c1 * x + c2 * x^2 + ... using Horner's method.
        constexpr float c0 = (POLY_DEGREE == 1)   ? 1.03022936050163882354355235184958220293399209290987f
                             : (POLY_DEGREE == 2) ? 0.999848792924395313327307061545061386175496934006f
                             : (POLY_DEGREE == 3) ? 0.99992449655091231753798502608929170703152709521188f
                                                  : 1.0000001510806179002040134468008959160576106495165f;
        constexpr float c1 = (POLY_DEGREE == 1)   ? 1.0201394465967894800285756834161653337107187804001f
                             : (POLY_DEGREE == 2) ? 1.01508760098521056684783640695492761469306929535975f
                             : (POLY_DEGREE == 3) ? 0.99993960415029750534472970577402987498389428593233f
                                                  : 0.99996228117047652035114096488703457970402030983204f;
        constexpr float c2 = (POLY_DEGREE == 2)   ? 0.50628367056745568861842335616023694454759126020461f
                             : (POLY_DEGREE == 3) ? 0.50502329058055065591138054839814880512001604099324f
                                                  : 0.49998365704615426417337683145647067790385638465486f;
        constexpr float c3 = (POLY_DEGREE == 3) ? 0.16817330195731531429790827442800245470170482723302f
                                                : 0.16792157982882225102649214918047336097544632172075f;
        constexpr float c4 = 4.1959439860014343843000081999668024587178974865521e-2;

        // Load polynomial coefficients.
        if constexpr (POLY_DEGREE >= 4) {
            TTI_SFPLOADI(p_sfpu::LREG3, 0xA, lo16(c4));
            TTI_SFPLOADI(p_sfpu::LREG3, 0x8, hi16(c4));
        }
        if constexpr (POLY_DEGREE >= 3) {
            TTI_SFPLOADI(p_sfpu::LREG4, 0xA, lo16(c3));
            TTI_SFPLOADI(p_sfpu::LREG4, 0x8, hi16(c3));
        }
        if constexpr (POLY_DEGREE >= 2) {
            TTI_SFPLOADI(p_sfpu::LREG5, 0xA, lo16(c2));
            TTI_SFPLOADI(p_sfpu::LREG5, 0x8, hi16(c2));
        }
        if constexpr (POLY_DEGREE >= 1) {
            TTI_SFPLOADI(p_sfpu::LREG6, 0xA, lo16(c1));
            TTI_SFPLOADI(p_sfpu::LREG6, 0x8, hi16(c1));
            TTI_SFPLOADI(p_sfpu::LREG7, 0xA, lo16(c0));
            TTI_SFPLOADI(p_sfpu::LREG7, 0x8, hi16(c0));
        }
    }

    if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
        TTI_SFPLOADI(p_sfpu::LREG3, 0xA, lo16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG3, 0x8, hi16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG4, 0xA, lo16(M_LN2));
        TTI_SFPLOADI(p_sfpu::LREG4, 0x8, hi16(M_LN2));
    }

    for (int d = 0; d < ITERATIONS; d++) {
        // Load the input.
        constexpr uint8_t input_type = IS_FP32_DEST_ACC_EN ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B;
        TTI_SFPLOAD(p_sfpu::LREG2, input_type, ADDR_MOD_X, 0);

        if constexpr (SCALE_EN) {
            TTI_SFPLOADI(p_sfpu::LREG0, 0, SCALE_BF16);
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
            INSERT_SFPNOP();
        }

        // Multiply by 1/ln(2) and round.
        if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        } else {
            TTI_SFPLOADI(p_sfpu::LREG1, 0xA, lo16(LN2_RECIP));
            TTI_SFPLOADI(p_sfpu::LREG1, 0x8, hi16(LN2_RECIP));
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        }
        INSERT_SFPNOP();
        TTI_SFP_STOCH_RND(
            0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);  // Clamp to [-127,+127].
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        if constexpr (USE_SFPARECIP_INSTR) {
#ifdef ARCH_BLACKHOLE
            // Calculate floor(x) by setting v=v-1 if v>u.
            TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);                                    // SFPGT_MOD1_SET_CC
            TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 2);  // SFPMAD_MOD1_NEGATE_VC
            TTI_SFPENCC(0, 0, 0, 0);

            // Calculate exp(x - k*ln2).
            TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, 2);
#else
            ASSERT(false);  // TTI_SFPARECIP instruction only supported on Blackhole".
#endif
        } else {
            if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            } else {
                TTI_SFPLOADI(p_sfpu::LREG0, 0xA, lo16(M_LN2));
                TTI_SFPLOADI(p_sfpu::LREG0, 0x8, hi16(M_LN2));
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            }
            INSERT_SFPNOP();

            // Calculate polynomial.
            if constexpr (POLY_DEGREE == 1) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 2) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 3) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else {  // degree 4.
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            }
            INSERT_SFPNOP();
        }

        // Multiply by 2^k.
        TT_SFPADDI(0x42fe, p_sfpu::LREG1, 0);  // Add 127.
        INSERT_SFPNOP();
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        TTI_SFPSETEXP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        INSERT_SFPNOP();

        // Handle underflow: if k == 0, exp(x) = 0 (fixes -inf case).
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 6);  // Set LaneFlags = (LREG1 == 0) and enable CC.
        TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);     // LREG2 = 0 ONLY for lanes where LREG1 == 0.
        TTI_SFPENCC(0, 0, 0, 0);               // Disable CC and clear LaneFlags - ALL lanes active again.

        // Store the result.
        if constexpr (!IS_FP32_DEST_ACC_EN) {
            // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it
            // so convert to bfloat16 using round-to-nearest-even.
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
        TTI_SFPSTORE(p_sfpu::LREG2, input_type, ADDR_MOD_X, 0);
        TTI_INCRWC(0, 4, 0, 0);  // Skip odd columns.
    }
}

/**
 * exp_tile on only the columns 0:8 of a face
 */
template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16>
void calculate_exponential_first_column() {
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
        constexpr int polynomial_degree = DST_ACCUM_MODE ? 4 : 2;
        calculate_exponential_polynomial<
            true,
            ITERATIONS_HALF_FACE,
            false,
            polynomial_degree,
            DST_ACCUM_MODE,
            scale_bf16>();
    }
}

template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16>
void exp_tile_first_column(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_exponential_first_column<SDPA_EXP_APPROX_MODE, scale_bf16>, idst, (int)VectorMode::C);
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
        MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));
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

template <bool SDPA_EXP_APPROX_MODE, int vector_mode = (int)VectorMode::C>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE>, idst, vector_mode, scale_bf16);
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
        copy_tile(cb_prev_max, i, dst_reg_0);
        copy_tile(cb_worker_max, i, dst_reg_1);
        copy_tile(cb_prev_sum, i, dst_reg_3);
        copy_tile(cb_worker_sum, i, dst_reg_4);
        MATH((fused_max_sub_exp_add_tile<EXP_APPROX_MODE, vector_mode>(0, scale_bf16)));
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
        MATH((exp_tile_first_column<false /*APPROX_MODE*/, (uint16_t)0xBF80 /*bf16(-1.0) scale*/>(0)));
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

/**
 * out_cb = in0_cb - in1_cb
 * Compile with size optimization to prevent binary size exceeding the limit.
 */
__attribute__((optimize("Os"))) void sub_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
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

/******************************************************************************
 *                             SDPA INNER LOOP                                *
 ******************************************************************************/
/**
 * Use the specialized wrapper functions below instead of calling this directly.
 *
 * Template Parameters:
 * @tparam sdpa_type - SDPA variant: STANDARD, JOINT, or RING
 * @tparam cb_qk_im - QK intermediate buffer
 * @tparam cb_identity_scale_in - Identity scale buffer
 * @tparam cb_attention_sink - Attention sink buffer
 * @tparam cb_scale_in - Scale buffer
 * @tparam Sq_chunk_t - Query chunk size in tiles
 * @tparam Sk_chunk_t - Key chunk size in tiles
 * @tparam DHt - Head dimension in tiles
 * @tparam vDHt - Value head dimension in tiles
 * @tparam use_attention_sink - Whether to use attention sink
 * @tparam is_causal - Whether to use causal masking
 * @tparam use_provided_mask - Whether to use user-provided mask
 * @tparam use_padded_mask - Whether to use padding mask
 * @tparam use_joint_mask - Whether to use joint mask
 * @tparam is_chunked - Whether query is chunked
 * @tparam scale_fp32 - FP32 scale factor
 * @tparam sliding_window_size - Sliding window attention size
 *
 * Runtime Parameters:
 * @param Skt - Sequence length in tiles
 * @param qk_in0_block_w - QK matmul block width
 * @param qk_subblock_w - QK matmul subblock width
 * @param qk_subblock_h - QK matmul subblock height
 * @param qk_in0_num_subblocks - QK input0 subblocks
 * @param qk_in1_num_subblocks - QK input1 subblocks
 * @param qk_num_blocks - QK number of blocks
 * @param out_in0_block_w - Output matmul block width
 * @param out_subblock_w - Output matmul subblock width
 * @param out_subblock_h - Output matmul subblock height
 * @param out_in0_num_subblocks - Output input0 subblocks
 * @param out_in1_num_subblocks - Output input1 subblocks
 * @param out_num_blocks - Output number of blocks
 * @param iter_q_start - Query iteration start
 * @param iter_q_end - Query iteration end
 * @param q_num_chunks - Total query chunks
 * @param local_q_start - Local query start
 * @param chunked_q_chunk_offset - Chunked query offset
 * @param iter_k_chunk_start - Key chunk iteration start
 * @param iter_k_chunk_end - Key chunk iteration end
 * @param q_chunk_tiles - Query chunk tiles
 * @param k_chunk_tiles - Key chunk tiles
 * @param qk_chunk_tiles - QK chunk tiles
 * @param out_chunk_tiles - Output chunk tiles
 * @param mask_chunk_0 - First mask chunk index
 * @param mask_chunk_1 - Second mask chunk index
 * @param ring_iter - Ring iteration index
 * @param ring_id - Ring ID
 * @param num_local_k_chunks - Number of K chunks stored locally on this device (used in Ring SDPA)
 * @param local_padded_Nt - Padded sequence length in tiles for local K/V chunks on this device
 * @param logical_nt - Logical (unpadded) sequence length in tiles for K/V
 * @param ring_iter_needs_global_n_mask - Whether current ring iteration requires global N masking
 * @param ring_iter_needs_joint_n_mask - Whether current ring iteration requires joint N masking
 * @param local_n_needs_masking - Whether local N dimension requires masking
 * @param global_n_mask_chunk_id - K chunk index where global N mask should be applied
 * @param local_n_mask_chunk_id - K chunk index where local N mask should be applied
 * @param joint_n_mask_chunk_id - K chunk index where joint N mask should be applied (relative to joint chunks)
 * @param cb_q_in - Query input buffer
 * @param cb_k_in - Key input buffer
 * @param cb_v_in - Value input buffer
 * @param cb_mask_in - Mask input buffer
 * @param cb_col_identity - Column identity buffer
 * @param cb_out_im_A - Output intermediate buffer A
 * @param cb_out_im_B - Output intermediate buffer B
 * @param cb_max_A - Max buffer A
 * @param cb_max_B - Max buffer B
 * @param cb_sum_A - Sum buffer A
 * @param cb_sum_B - Sum buffer B
 * @param cb_exp_max_diff - Exp max diff buffer
 * @param cb_lse_in - LSE input buffer
 * @param cb_lse_out - LSE output buffer
 * @param cb_prev_out - Previous output buffer
 * @param cb_out - Output buffer
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
    const uint32_t num_local_k_chunks,
    const uint32_t local_padded_Nt,
    const uint32_t logical_nt,
    const bool ring_iter_needs_global_n_mask,
    const bool ring_iter_needs_joint_n_mask,
    const bool local_n_needs_masking,
    const uint32_t global_n_mask_chunk_id,
    const uint32_t local_n_mask_chunk_id,
    const uint32_t joint_n_mask_chunk_id,
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
    uint32_t KV_chunks_processed_in_iter = 0;

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

        uint32_t processed_k_chunks = 0;

        for (uint32_t k_chunk = iter_k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            if constexpr (sdpa_type == RING) {
                const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
                // Global index into the padded KV tensor
                const uint32_t kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
                if (!kv_chunk_is_joint && (kv_global_start_tile >= logical_nt)) {
                    // This is a KV chunk on spatial input beyond the logical N, and not joint KV. Skip it.
                    continue;
                }
            }

            KV_chunks_processed_in_iter++;

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

            bool apply_mask = false;
            if constexpr (sdpa_type == RING) {
                apply_mask = (ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) ||
                             (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) ||
                             (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id);
            } else if constexpr (is_causal || sliding_window_size > 0) {
                // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                // Q-range = [q_low, q_high)
                // K-range = [k_low, k_high)
                // does_overlap = not (q_low >= k_high or k_low >= q_high)
                // Due to loop bounds, we should never have k_low >= q_high.
                const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
                // Apply mask if causal overlap or sliding window is active
                apply_mask = (q_low_idx < k_high_idx) || (sliding_window_size > 0);
            } else if constexpr (use_provided_mask) {
                apply_mask = true;
            } else if constexpr (use_padded_mask) {
                // Apply mask only on the last K chunk
                apply_mask = (k_chunk == iter_k_chunk_end - 1);
            } else if constexpr (use_joint_mask) {
                // Apply mask for specific chunk combinations
                apply_mask = (k_chunk == mask_chunk_0) || (k_chunk == mask_chunk_1);
            }

            if (apply_mask) {
                /* QK += MASK */
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
            }

            /**
             * reduce_c can perform both reduce_max and eltwise max with previous result.
             * if do_eltwise_max:
             *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
             * else:
             *  cur_max = max(qk, dim=-1)
             */
            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, Sk_chunk_t>(
                alias_cur_max, alias_prev_max, processed_k_chunks > 0);

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
            if (processed_k_chunks > 0) {
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
                mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(
                    alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
            }

            // Swap CB handles to prepare for next iteration
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
            std::swap(alias_prev_max, alias_cur_max);

            processed_k_chunks++;
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

            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_attention_sink, cb_identity_scale_in, Sq_chunk_t, 1>(
                alias_cur_max, alias_prev_max, true);

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
            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(
                alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
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
                reconfig_data_format(cb_prev_out, alias_cur_out);
                sub_block(cb_prev_out, alias_cur_out, alias_sub, out_chunk_tiles);
                // alias_sub *= alias_sig
                reconfig_data_format(alias_sub, alias_sig);
                mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_sub, alias_sig);
                // cb_out = cb_prev_out - alias_sub
                reconfig_data_format(cb_prev_out, alias_sub);
                pack_reconfig_data_format(cb_out);
                sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
                cb_pop_front(cb_prev_out, out_chunk_tiles);
                cb_pop_front(alias_cur_out, out_chunk_tiles);
                cb_pop_front(alias_sub, out_chunk_tiles);

                // alias_sig = sigmoid(cb_lse_in - alias_cur_lse)
                // alias_cur_lse = log(alias_sig)
                // cb_lse_out = cb_lse_in - alias_cur_lse
                pack_reconfig_data_format(alias_sig);
                reconfig_data_format(cb_lse_in, alias_cur_lse);
                logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
                sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
                cb_pop_front(alias_sig, Sq_chunk_t);
                cb_pop_front(alias_cur_lse, Sq_chunk_t);
                cb_pop_front(cb_lse_in, Sq_chunk_t);
            } else {
                pack_reconfig_data_format(cb_out);
                copy_block(alias_mm2_prev_out, cb_out, out_chunk_tiles);

                pack_reconfig_data_format(cb_lse_out);
                copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
            }
        } else {
            /* cb_cur_sum = 1.0 / cb_cur_sum */
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);

            /* cb_out_accumulate_im *= cb_cur_sum */
            pack_reconfig_data_format(cb_out);
            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);

            // free up cb_prev_max after K chunks
            cb_pop_front(alias_prev_max, Sq_chunk_t);
        }

        cb_pop_front(cb_q_in, q_chunk_tiles);
    }

    if constexpr (sdpa_type == RING) {
        if (KV_chunks_processed_in_iter % 2 == 0) {
            cb_wait_front(cb_k_in, k_chunk_tiles);
            cb_wait_front(cb_v_in, k_chunk_tiles);
            cb_pop_front(cb_k_in, k_chunk_tiles);
            cb_pop_front(cb_v_in, k_chunk_tiles);
        }
    }

    if constexpr (use_attention_sink) {
        cb_pop_front(cb_attention_sink, Sq_chunk_t);
    }
}

/******************************************************************************
 *                          SDPA WRAPPER FUNCTIONS                            *
 ******************************************************************************/

/**
 * Standard SDPA with optional causal masking, attention sink, and sliding window.
 */
template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_attention_sink,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    bool use_attention_sink,
    bool is_causal,
    bool use_provided_mask,
    bool use_padded_mask,
    bool is_chunked,
    uint32_t scale_fp32,
    uint32_t sliding_window_size>
void sdpa_standard(
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
    const uint32_t k_num_chunks,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
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
    const uint32_t cb_out) {
    sdpa_inner_loop<
        STANDARD,
        cb_qk_im,
        cb_identity_scale_in,
        cb_attention_sink,
        0,  // cb_scale_in (not used)
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        vDHt,
        use_attention_sink,
        is_causal,
        use_provided_mask,
        use_padded_mask,
        false,  // use_joint_mask (not used)
        is_chunked,
        scale_fp32,
        sliding_window_size>(
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
        iter_q_start,
        iter_q_end,
        q_num_chunks,
        local_q_start,
        chunked_q_chunk_offset,
        0,             // iter_k_chunk_start
        k_num_chunks,  // iter_k_chunk_end
        q_chunk_tiles,
        k_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        0,      // mask_chunk_0 (not used)
        0,      // mask_chunk_1 (not used)
        0,      // ring_iter (not used)
        0,      // ring_id (not used)
        0,      // num_local_k_chunks (not used)
        0,      // local_padded_Nt (not used)
        0,      // logical_nt (not used)
        false,  // ring_iter_needs_global_n_mask (not used)
        false,  // ring_iter_needs_joint_n_mask (not used)
        false,  // local_n_needs_masking (not used)
        0,      // global_n_mask_chunk_id (not used)
        0,      // local_n_mask_chunk_id (not used)
        0,      // joint_n_mask_chunk_id (not used)
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        0,  // cb_lse_in (not used)
        0,  // cb_lse_out (not used)
        0,  // cb_prev_out (not used)
        cb_out);
}

/**
 * Joint SDPA for multi-modal attention.
 */
template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    bool use_joint_mask,
    uint32_t scale_fp32>
void sdpa_joint(
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
    const uint32_t local_q_start,
    const uint32_t local_q_end,
    const uint32_t k_num_chunks,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t mask_chunk_0,
    const uint32_t mask_chunk_1,
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
    const uint32_t cb_out) {
    sdpa_inner_loop<
        JOINT,
        cb_qk_im,
        cb_identity_scale_in,
        0,  // cb_attention_sink (not used)
        0,  // cb_scale_in (not used)
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        DHt,    // vDHt = DHt
        false,  // use_attention_sink (not used)
        false,  // is_causal (not used)
        false,  // use_provided_mask (not used)
        false,  // use_padded_mask (not used)
        use_joint_mask,
        false,  // is_chunked (not used)
        scale_fp32,
        0>(  // sliding_window_size (not used)
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
        local_q_start,  // iter_q_start
        local_q_end,    // iter_q_end
        0,              // q_num_chunks (not used)
        local_q_start,
        0,             // chunked_q_chunk_offset (not used)
        0,             // iter_k_chunk_start
        k_num_chunks,  // iter_k_chunk_end
        q_chunk_tiles,
        k_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        mask_chunk_0,
        mask_chunk_1,
        0,      // ring_iter (not used)
        0,      // ring_id (not used)
        0,      // num_local_k_chunks (not used)
        0,      // local_padded_Nt (not used)
        0,      // logical_nt (not used)
        false,  // ring_iter_needs_global_n_mask (not used)
        false,  // ring_iter_needs_joint_n_mask (not used)
        false,  // local_n_needs_masking (not used)
        0,      // global_n_mask_chunk_id (not used)
        0,      // local_n_mask_chunk_id (not used)
        0,      // joint_n_mask_chunk_id (not used)
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        0,  // cb_lse_in (not used)
        0,  // cb_lse_out (not used)
        0,  // cb_prev_out (not used)
        cb_out);
}

/**
 * Ring SDPA for distributed multi-device attention.
 */
template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t scale_fp32>
void sdpa_ring(
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
    const uint32_t global_q_start,
    const uint32_t global_q_end,
    const uint32_t iter_k_chunk_start,
    const uint32_t iter_k_chunk_end,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t ring_iter,
    const uint32_t ring_id,
    const uint32_t num_local_k_chunks,
    const uint32_t local_padded_Nt,
    const uint32_t logical_nt,
    const bool ring_iter_needs_global_n_mask,
    const bool ring_iter_needs_joint_n_mask,
    const bool local_n_needs_masking,
    const uint32_t global_n_mask_chunk_id,
    const uint32_t local_n_mask_chunk_id,
    const uint32_t joint_n_mask_chunk_id,
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
    sdpa_inner_loop<
        RING,
        cb_qk_im,
        cb_identity_scale_in,
        0,  // cb_attention_sink (not used)
        cb_scale_in,
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        DHt,    // vDHt = DHt
        false,  // use_attention_sink (not used)
        false,  // is_causal (not used)
        false,  // use_provided_mask (not used)
        false,  // use_padded_mask (not used)
        false,  // use_joint_mask (not used)
        false,  // is_chunked (not used)
        scale_fp32,
        0>(  // sliding_window_size (not used)
        0,   // Skt (not used)
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
        global_q_start,  // iter_q_start
        global_q_end,    // iter_q_end
        0,               // q_num_chunks (not used)
        0,               // local_q_start (not used)
        0,               // chunked_q_chunk_offset (not used)
        iter_k_chunk_start,
        iter_k_chunk_end,
        q_chunk_tiles,
        k_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        0,  // mask_chunk_0 (not used)
        0,  // mask_chunk_1 (not used)
        ring_iter,
        ring_id,
        num_local_k_chunks,
        local_padded_Nt,
        logical_nt,
        ring_iter_needs_global_n_mask,
        ring_iter_needs_joint_n_mask,
        local_n_needs_masking,
        global_n_mask_chunk_id,
        local_n_mask_chunk_id,
        joint_n_mask_chunk_id,
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        cb_lse_in,
        cb_lse_out,
        cb_prev_out,
        cb_out);
}

/**
 * Windowed SDPA with user-provided mask.
 */
template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t scale_fp32>
void sdpa_windowed(
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
    const uint32_t local_q_start,
    const uint32_t q_chunks_per_core,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
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
    const uint32_t cb_out) {
    sdpa_inner_loop<
        STANDARD,
        cb_qk_im,
        cb_identity_scale_in,
        0,  // cb_attention_sink (not used)
        0,  // cb_scale_in (not used)
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        DHt,    // vDHt = DHt
        false,  // use_attention_sink (not used)
        false,  // is_causal (not used)
        true,   // use_provided_mask (used)
        false,  // use_padded_mask (not used)
        false,  // use_joint_mask (not used)
        false,  // is_chunked (not used)
        scale_fp32,
        0>(  // sliding_window_size (not used)
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
        0,                  // iter_q_start
        q_chunks_per_core,  // iter_q_end
        0,                  // q_num_chunks (not used)
        local_q_start,
        0,  // chunked_q_chunk_offset (not used)
        0,  // iter_k_chunk_start
        0,  // iter_k_chunk_end (not used -- uses Skt)
        q_chunk_tiles,
        k_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        0,      // mask_chunk_0 (not used)
        0,      // mask_chunk_1 (not used)
        0,      // ring_iter (not used)
        0,      // ring_id (not used)
        0,      // num_local_k_chunks (not used)
        0,      // local_padded_Nt (not used)
        0,      // logical_nt (not used)
        false,  // ring_iter_needs_global_n_mask (not used)
        false,  // ring_iter_needs_joint_n_mask (not used)
        false,  // local_n_needs_masking (not used)
        0,      // global_n_mask_chunk_id (not used)
        0,      // local_n_mask_chunk_id (not used)
        0,      // joint_n_mask_chunk_id (not used)
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        0,  // cb_lse_in (not used)
        0,  // cb_lse_out (not used)
        0,  // cb_prev_out (not used)
        cb_out);
}
