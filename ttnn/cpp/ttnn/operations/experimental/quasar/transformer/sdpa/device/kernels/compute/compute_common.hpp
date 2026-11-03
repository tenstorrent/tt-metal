// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Vendored copy of ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp for the
// quasar sdpa fork. Byte-identical to the main-tree header (unmodified on the port branch); vendored to
// insulate the fork from future in-place Metal 2.0 ports of the prefill sdpa op. Transitive includes
// (q_chunk_remapping.hpp, ckernel_sfpu_sdpa.h) still resolve to the
// unmodified main-tree headers.

#pragma once

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
#include "api/compute/reduce_custom.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp"
#include "cpp/ttnn/kernel_lib/dest_helpers.hpp"
// ckernel_sfpu_sdpa.h has no Quasar implementation (raw WH/BH SFPI microcode for the fused
// first-column softmax primitives). On Quasar, every helper below that used a fused primitive
// is routed to generic SFPU tile ops instead (see the ARCH_QUASAR branches throughout this file).
#if (defined(TRISC_MATH) || defined(TRISC_PACK)) && !defined(ARCH_QUASAR)
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif

#ifdef ARCH_QUASAR
// Operand of the LAST actual pack, so pack_reconfig_out() runs pack_init ONLY on a real operand change.
// Reset to the sentinel at each kernel_main entry (reset_pack_operand_tracking()). Per-TRISC global; the
// PACK thread's copy is what gates its pack_init.
inline uint32_t g_last_pack_operand = 0xFFFFFFFFu;
ALWI void reset_pack_operand_tracking() { g_last_pack_operand = 0xFFFFFFFFu; }
#else
ALWI void reset_pack_operand_tracking() {}
#endif

// Switch the packer output operand to `out_dfb`.
//
// Quasar packer quirk: `pack_reconfig_data_format(new)` only updates the pack DATA FORMAT, not the output
// ring/address (api/compute/reconfig_data_format.h NOTE ARCH_QUASAR; Quasar `_llk_pack_reconfig_data_
// format_` writes only THCON_PACKER<N>_REG0_IN_DATA_FORMAT). Re-pointing the pack dest requires
// `pack_init` (→ `_llk_pack_mop_config_(buf_desc)` re-latches L1_Dest_addr). Without it the packer keeps
// writing to the previous ring → the intended DFB is never written → all-zero output, no assert (this
// caused sdpa's first emulator run to return an all-zero tensor).
//
// BUT `pack_init` is a FULL re-init: it also resets the packer's per-tile dest write pointer /
// accumulation-section state. Calling it at a SAME-operand reconfig (e.g. mid-L1-accumulate, like
// fma_block_merged_sum pass 2 reconfiguring dfb_sum after pass 1 already packed it) clobbers the
// in-progress accumulation (bounded-wrong output). So `pack_init` must run ONLY on a genuine operand
// change vs the last ACTUAL pack — tracked at runtime here (handles cross-helper carry and if-constexpr
// branches without static per-site classification; runtime-tracking design from the sdpa_decode fork).
// Requirement: every pack-operand switch goes through this wrapper; matmul_blocks / pack_tile inherit
// the wrapper-set operand (they never call pack_init/pack_reconfig themselves).
//
// WH/BH do NOT need any of this (pack_reconfig alone re-targets correctly there; validated 13/13), so
// everything is ARCH_QUASAR-gated to leave the validated WH path byte-identical.
ALWI void pack_reconfig_out(uint32_t out_dfb) {
    pack_reconfig_data_format(out_dfb);  // format always re-applied; same-operand safe
#ifdef ARCH_QUASAR
    if (out_dfb != g_last_pack_operand) {
        pack_init(out_dfb);
        g_last_pack_operand = out_dfb;
    }
#endif
}

ALWI void sdpa_reduce_copy_tile_to_dst_init_short(uint32_t dfbid, uint32_t transpose = 0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, true /*transpose within 16x16 face*/, dfbid)));

#ifdef ARCH_QUASAR
    // Quasar's llk_math_eltwise_unary_datacopy_init takes no PackMode param; its trailing template
    // args are bools (unpack_to_dest, is_int_fpu_en, tilize).
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*unpack_to_dest*/,
          false /*is_int_fpu_en*/>(dfbid)));
#else
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false,  // is_int_fpu_en
          PackMode::Default>(dfbid)));
#endif
}

/**
 * in0_dfb = max(in0_dfb, in1_dfb)
 */
template <uint32_t num_tiles>
void max_block_inplace(uint32_t in0, uint32_t in1) {
    DataflowBuffer dfb_in0(in0);
    DataflowBuffer dfb_in1(in1);
    // inputs come in full, outputs go out full
    copy_init(in0);
    copy_init(in1);
    binary_max_tile_init();
    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(num_tiles);
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
    dfb_in0.pop_front(num_tiles);
    dfb_in0.reserve_back(num_tiles);
    // pack-side drain: reserve_back->push_back needs a PACR between them (dummy_unpack/UNPACR
    // can't order the pack side).
    dummy_pack(in0);
    dfb_in0.push_back(num_tiles);
}

/**
 * out_dfb = eltwise_max(in0, in1)
 */
template <VectorMode vector_mode = VectorMode::RC>
void max_block(uint32_t in0, uint32_t in1, uint32_t out_dfb, uint32_t num_tiles) {
    DataflowBuffer dfb_in0(in0);
    DataflowBuffer dfb_in1(in1);
    DataflowBuffer dfb_out(out_dfb);
    // inputs come in full, outputs go out full
    copy_init(in0);
    binary_max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(num_tiles);
    dfb_out.reserve_back(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_0, vector_mode);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_reg_0, out_dfb, i);
        tile_regs_release();
    }
    dfb_out.push_back(num_tiles);
}

/**
 * out_dfb = reduce[MAX,SUM](in0_dfb * scale_dfb)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_dfb,
    uint32_t scale_dfb,
    uint32_t rows,
    uint32_t cols,
    VectorMode vector_mode = VectorMode::C>
void reduce_c(uint32_t out_dfb, uint32_t prev_dfb, bool do_eltwise_max = false) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_scale(scale_dfb);
    DataflowBuffer dfb_out(out_dfb);
    DataflowBuffer dfb_prev(prev_dfb);
    // Precondition: in0_dfb has rows*cols produced. in0_dfb has tiles in row-major order
    // Precondition: scale_dfb has 1 produced
    // Precondition: out_dfb has rows free
    // Postcondition: in0_dfb has rows*cols produced
    // Precondition: scale_dfb has 1 produced
    // Postcondition: out_dfb has rows produced
    // If do_eltwise_max == true, prev_dfb has rows produced.

    constexpr uint32_t num_tiles = rows * cols;

#if defined REDUCE_GRANULARITY
    constexpr uint32_t dst_tiles = (rows < REDUCE_GRANULARITY) ? rows : REDUCE_GRANULARITY;
    constexpr uint32_t granularity = (rows >= REDUCE_GRANULARITY) ? (rows / REDUCE_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = rows;
#endif

    dfb_scale.wait_front(1);
    dfb_out.reserve_back(rows);

    const uint32_t num_tiles_to_wait = dst_tiles * cols;
    uint32_t in0_wait_tiles = num_tiles_to_wait;

    uint32_t row_start_idx = 0;
    for (uint32_t g = 0; g < granularity; g++) {
        dfb_in0.wait_front(in0_wait_tiles);
        tile_regs_acquire();

        if (do_eltwise_max) {
            dfb_prev.wait_front(g * dst_tiles);
            /**
             * Copy previous max values into DST register.
             * Note that this special invocation of copy_tile is necessary to produce
             * tiles in DST with transposed faces, as `reduce_block_max_row` expects.
             */
            reconfig_data_format_srca(prev_dfb);
            sdpa_reduce_copy_tile_to_dst_init_short(prev_dfb);
            for (uint32_t i = 0; i < dst_tiles; i++) {
                const uint32_t cur_max_dst_idx = i;
                copy_tile(prev_dfb, (row_start_idx + i), cur_max_dst_idx);
            }
            reconfig_data_format_srca(in0_dfb);
        }

        /**
         * For `dst_tiles` number of rows, compute the max into the even indices of the DST register.
         */
        reduce_block_max_row_init<cols>(out_dfb);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t reduce_dst_idx = i;
            reduce_block_max_row<cols>(in0_dfb, scale_dfb, (row_start_idx + i) * cols, reduce_dst_idx);
        }
        reduce_block_max_row_uninit(in0_dfb);

        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_out(out_dfb);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t cur_max_dst_idx = i;
            pack_tile<true>(cur_max_dst_idx, out_dfb, (row_start_idx + i));
        }
        tile_regs_release();

        row_start_idx += dst_tiles;
        in0_wait_tiles += num_tiles_to_wait;
    }

    dfb_out.push_back(rows);
}

/**
 * out_dfb = reduce[MAX,SUM](in0_dfb * scale_dfb)
 *
 * In this version cols does not have to be a compile-time constant.
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_dfb,
    uint32_t scale_dfb,
    uint32_t rows,
    VectorMode vector_mode = VectorMode::C>
void reduce_c(uint32_t out_dfb, uint32_t prev_dfb, uint32_t cols, bool do_eltwise_max = false) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_scale(scale_dfb);
    DataflowBuffer dfb_out(out_dfb);
    // Precondition: in0_dfb has rows*cols produced. in0_dfb has tiles in row-major order
    // Precondition: scale_dfb has 1 produced
    // Precondition: out_dfb has rows free
    // Postcondition: in0_dfb has rows*cols produced
    // Precondition: scale_dfb has 1 produced
    // Postcondition: out_dfb has rows produced

    uint32_t num_tiles = rows * cols;
    dfb_scale.wait_front(1);
    dfb_in0.wait_front(num_tiles);
    dfb_out.reserve_back(rows);

    pack_reconfig_out(out_dfb);

    binary_max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1;

    for (uint32_t i = 0; i < rows; i++) {
        reconfig_data_format_srca(in0_dfb);
        tile_regs_acquire();
        reduce_init<pool_type, reduce_dim>(in0_dfb, scale_dfb, out_dfb);
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_dfb, scale_dfb, i * cols + j, 0, reduce_dst_idx);
        }
        reduce_uninit();
        if (do_eltwise_max) {
            reconfig_data_format_srca(prev_dfb);
            copy_init(prev_dfb);
            copy_tile(prev_dfb, i, prev_max_dst_idx);
            binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx, vector_mode);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(reduce_dst_idx, out_dfb);
        tile_regs_release();
    }

    dfb_out.push_back(rows);
}

#if defined(TRISC_MATH) && !defined(ARCH_QUASAR)
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void recip_tile_first_column(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_recip_first_column, (is_fp32_dest_acc_en), idst, VectorMode::C);
}
#endif

/**
 * in_dfb = 1 / in_dfb
 */
void recip_block_inplace(uint32_t in_dfb, uint32_t num_tiles) {
    DataflowBuffer dfb_in(in_dfb);
    // Precondition: in_dfb has num_tiles produced
    // Postcondition: in_dfb has num_tiles produced
    reconfig_data_format_srca(in_dfb);
    copy_init(in_dfb);
#ifdef ARCH_QUASAR
    recip_tile_init();
#else
    // The first-column helper uses SFPI, not full-tile LOADMACRO/replay state.
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::sfpu_reciprocal_init, (APPROX)));
#endif
    pack_reconfig_out(in_dfb);

    dfb_in.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in_dfb, i, 0);
#ifdef ARCH_QUASAR
        recip_tile(0);  // full-tile 1/x; only col 0 is consumed downstream
#else
        MATH((recip_tile_first_column(0)));  // WH/BH fused fast path
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, in_dfb);
        tile_regs_release();
    }
    dfb_in.pop_front(num_tiles);
    dfb_in.reserve_back(num_tiles);
    // Requires Quasar pack-side drain
    dummy_pack(in_dfb);
    dfb_in.push_back(num_tiles);
}

/**
 * in0_dfb = exp((in0_dfb - in1_dfb) * scale_fp32)
 */
template <
    uint32_t in0_dfb,
    uint32_t rows,
    uint32_t scale_fp32,
    bool write_result_inplace = true,
    bool do_reduce = true,
    VectorMode vector_mode = VectorMode::RC>
void sub_exp_block_bcast_cols_inplace(uint32_t in1_dfb, uint32_t reduce_dfb, uint32_t cols) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    DataflowBuffer dfb_reduce(reduce_dfb);
    // Precondition: in0_dfb has rows*cols produced
    // Precondition: in1_dfb has rows produced
    // Postcondition: in0_dfb has rows*cols produced
    // Postcondition: in1_dfb has rows produced
    // llk_unpack_AB_init (inside sub_bcast_cols_init) validates the live
    // unpacker configuration.  Reconfigure first: qk_im can be FP32 while the
    // row maximum is BF16, notably after applying a windowed BF16 mask.
    reconfig_data_format(in0_dfb, in1_dfb);
    sub_bcast_cols_init(in0_dfb, in1_dfb);

#ifndef ARCH_QUASAR
    // The exponential function uses InputClamping::None for better performance. This version
    // produces incorrect outputs for inputs <~ -88, but those outputs are guaranteed to be negative.
    // Enable packer ReLU to zero any negative values produced by the exponential approximation.
    exp_tile_init<true /* approx */, scale_fp32, InputClamping::None>();
#endif
    // Quasar's exp only supports the default scale (1.0) and CLAMP_NEGATIVE=true (see the
    // ARCH_QUASAR branch below, which applies scale_fp32 via a separate multiply before an
    // unscaled exp). CLAMP_NEGATIVE matches the WH packer-ReLU-zeroes-negatives intent, so the
    // ReLU config below stays shared for both archs (it's a no-op once inputs are already clamped).
    PACK((llk_pack_relu_config(ReluConfig::zero())));

    dfb_in0.wait_front(rows * cols);
    dfb_in1.wait_front(rows);
    if constexpr (do_reduce) {
        dfb_reduce.reserve_back(rows);
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
#ifdef ARCH_QUASAR
            // Quasar's exp only supports scale=1.0 (see exp_init's static_assert), so scale_fp32 is
            // applied via a separate multiply pass across all tiles in this dst block, then an
            // unscaled exp pass. sub_tiles_bcast_cols is an FPU op (separate pipeline, no state to
            // reissue); mul and exp are both SFPU op families, so each family's _init is issued once,
            // immediately before its own loop over j, rather than interleaved per-tile.
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_dfb, in1_dfb, j, i, j);
            }
            binop_with_scalar_tile_init();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                mul_unary_tile(j, scale_fp32);
            }
            exp_tile_init<true /* approx */>();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                exp_tile<true /* approx */, false /* scale_en */>(j);
            }
#else
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_dfb, in1_dfb, j, i, j);
                constexpr int iterations = (vector_mode == VectorMode::RC) ? 32 /*ITER*/ : 8 /*ITER*/;
                constexpr VectorMode vector_mode_exp = (vector_mode == VectorMode::RC) ? VectorMode::None : vector_mode;
                exp_tile<true /* approx */, false /* scale_en */, InputClamping::None, iterations>(j, vector_mode_exp);
            }
#endif
            tile_regs_commit();

            if constexpr (write_result_inplace) {
                dfb_in0.pop_front(dst_tiles);
                dfb_in0.reserve_back(dst_tiles);
            }

            tile_regs_wait();

            if constexpr (write_result_inplace) {
                pack_reconfig_out(in0_dfb);
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, in0_dfb);
                }
                // Granular write output to enable following matmul unpack to start early.
                dfb_in0.push_back(dst_tiles);
            }

            if constexpr (do_reduce) {
                pack_reconfig_out(reduce_dfb);
                // While we have results in DST, take advantage of L1 accumulation
                // to reduce row x cols tiles to rows x 1 tiles.
                if (u > 0) {
                    // If on the same row, keep accumulating
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile<true>(j, reduce_dfb, i);
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
        dfb_reduce.push_back(rows);
    }

    PACK((llk_pack_relu_config(ReluConfig::none())));
}

/**
 * out_dfb = in0_dfb * in1_dfb
 * @tparam rows - Number of rows of tiles
 * @tparam cols - Number of columns of tiles
 * @tparam immediate_pop - If true, uses tile-by-tile processing with immediate DFB pop after each tile.
 *                         If false, uses batched processing with deferred DFB pop, processing multiple tiles in
 * parallel.
 * @tparam pack_accumulate - If true, enables L1 accumulation to accumulate results onto existing tiles
 *                           in out_dfb. Only supported when immediate_pop=false.
 */
template <uint32_t rows, uint32_t cols, bool immediate_pop, bool pack_accumulate>
void mul_block_bcast_cols(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t out_dfb) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    DataflowBuffer dfb_out(out_dfb);
    // Precondition: in0_dfb has rows*cols produced
    // Precondition: in1_dfb has rows produced
    // Precondition: out_dfb has rows*cols produced
    // Postcondition: in0_dfb empty
    // Postcondition: in1_dfb empty
    // Postcondition: out_dfb has rows*cols produced

    constexpr uint32_t num_tiles = rows * cols;

    reconfig_data_format(in0_dfb, in1_dfb);
    pack_reconfig_out(out_dfb);
    mul_bcast_cols_init(in0_dfb, in1_dfb);
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(rows);

    if constexpr (immediate_pop) {
        static_assert(!pack_accumulate, "Unsupported parameter configuration");
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                tile_regs_acquire();
                mul_tiles_bcast_cols(in0_dfb, in1_dfb, 0, i, 0);
                tile_regs_commit();
                dfb_in0.pop_front(1);
                dfb_out.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, out_dfb);
                tile_regs_release();
                dfb_out.push_back(1);
            }
        }
        dfb_in1.pop_front(rows);
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
            dfb_out.reserve_back(num_tiles);
        }
        uint32_t in0_index = 0;
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t u = 0; u < granularity; ++u) {
                tile_regs_acquire();
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    mul_tiles_bcast_cols(in0_dfb, in1_dfb, in0_index, i, j);
                    in0_index++;
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, out_dfb);
                }
                tile_regs_release();
            }
        }
        dfb_in1.pop_front(rows);
        dfb_in0.pop_front(num_tiles);
        if (pack_accumulate) {
            PACK((llk_pack_reconfig_l1_acc(false)));
            dfb_out.pop_front(num_tiles);
            dfb_out.reserve_back(num_tiles);
            // Requires Quasar pack-side drain
            dummy_pack(out_dfb);
            dfb_out.push_back(num_tiles);
        } else {
            dfb_out.push_back(num_tiles);
        }
    }
}

/**
 * in0_dfb *= in1_dfb
 */
template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols_inplace(uint32_t in0_dfb, uint32_t in1_dfb) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    // Precondition: in0_dfb has rows*cols produced
    // Precondition: in1_dfb has rows produced
    // Postcondition: in0_dfb has rows*cols produced
    // Postcondition: in1_dfb has rows consumed

    constexpr uint32_t num_tiles = rows * cols;

#ifdef DHT_GRANULARITY
    constexpr uint32_t dst_tiles = (cols < DHT_GRANULARITY) ? cols : DHT_GRANULARITY;
    constexpr uint32_t granularity = (cols >= DHT_GRANULARITY) ? (cols / DHT_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = cols;
#endif

    reconfig_data_format(in0_dfb, in1_dfb);
    mul_bcast_cols_init(in0_dfb, in1_dfb);
    pack_reconfig_out(in0_dfb);
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; ++u) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                mul_tiles_bcast_cols(in0_dfb, in1_dfb, j, i, j);
            }
            tile_regs_commit();
            dfb_in0.pop_front(dst_tiles);
            dfb_in0.reserve_back(dst_tiles);
            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_dfb);
            }
            dfb_in0.push_back(dst_tiles);
            tile_regs_release();
        }
    }
    dfb_in1.pop_front(rows);
    reconfig_data_format_srcb(in0_dfb);
}

template <bool pop_in1 = true>
void add_block_inplace(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t num_tiles) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    // Precondition: in0_dfb and in1_dfb have num_tiles produced
    // Postcondition: in0_dfb has num_tiles produced
    // Postcondition: in1_dfb has num_tiles consumed

    reconfig_data_format(in0_dfb, in1_dfb);
    pack_reconfig_out(in0_dfb);
    add_init(in0_dfb, in1_dfb);
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        add_tiles(in0_dfb, in1_dfb, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, in0_dfb);
        tile_regs_release();
    }

    dfb_in0.pop_front(num_tiles);
    if (pop_in1) {
        dfb_in1.pop_front(num_tiles);
    }
    dfb_in0.reserve_back(num_tiles);
    // Requires Quasar pack-side drain
    dummy_pack(in0_dfb);
    dfb_in0.push_back(num_tiles);
}

void mul_tiles_bcast_cols_inplace(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t num_tiles) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    /**
     * Given in0_dfb and in1_dfb, multiply each tile of in0_dfb by the corresponding tile of in1_dfb
     * and bcast cols of in1_dfb.
     */
    // Precondition: in0_dfb and in1_dfb have num_tiles produced
    // Postcondition: in0_dfb has num_tiles produced
    // Postcondition: in1_dfb has num_tiles produced

    reconfig_data_format(in0_dfb, in1_dfb);
    mul_bcast_cols_init(in0_dfb, in1_dfb);
    pack_reconfig_out(in0_dfb);
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(in0_dfb, in1_dfb, 0, i, 0);
        tile_regs_commit();
        dfb_in0.pop_front(1);
        dfb_in0.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, in0_dfb);
        tile_regs_release();
        dfb_in0.push_back(1);
    }
}

/**
 * in0_dfb *= in1_dfb
 */
void mul_block_inplace(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t num_tiles) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    // Precondition: in0_dfb and in1_dfb have num_tiles produced
    // Postcondition: in0_dfb has num_tiles produced
    // Postcondition: in1_dfb has num_tiles produced

    mul_init(in0_dfb, in1_dfb);
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        tile_regs_acquire();
        mul_tiles(in0_dfb, in1_dfb, 0, i, 0);
        tile_regs_commit();
        dfb_in0.pop_front(1);
        dfb_in0.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, in0_dfb);
        tile_regs_release();
        dfb_in0.push_back(1);
    }
}

#if (defined(TRISC_MATH) || defined(TRISC_PACK)) && !defined(ARCH_QUASAR)

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
#endif  // (defined(TRISC_MATH) || defined(TRISC_PACK)) && !defined(ARCH_QUASAR)

/**
 * out_dfb = exp((in0_dfb - in1_dfb) * scale_fp32)
 */
template <uint32_t scale_fp32>
void sub_exp_block(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t out_dfb, uint32_t num_tiles) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    DataflowBuffer dfb_out(out_dfb);
    // Precondition: in0_dfb and in1_dfb have num_tiles produced
    // Postcondition: out_dfb has num_tiles produced
    // Postcondition: in0_dfb and in1_dfb has num_tiles produced

    sub_init(in0_dfb, in1_dfb);
#ifndef ARCH_QUASAR
    exp_tile_init<EXP_APPROX_MODE>();
#endif
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(num_tiles);
    dfb_out.reserve_back(num_tiles);

#ifndef ARCH_QUASAR
    // Convert scale_fp32 to bf16 scale
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
#endif

    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        tile_regs_acquire();
        sub_tiles(in0_dfb, in1_dfb, i, i, 0);
#ifdef ARCH_QUASAR
        // No fused exp_tile_first_column on Quasar, and Quasar's exp only supports scale=1.0 (see
        // exp_init's static_assert), so scale_fp32 (already fp32-encoded) is applied via a separate
        // multiply before an unscaled exp. Full-tile exp is numerically safe; only column 0 is
        // consumed downstream.
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_fp32);
        exp_tile_init<EXP_APPROX_MODE>();
        exp_tile<EXP_APPROX_MODE, false /*scale_en*/>(0);
#else
        MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));  // WH/BH fused fast path
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_dfb);
        tile_regs_release();
        dfb_out.push_back(1);
    }
}

#if defined(TRISC_MATH) && !defined(ARCH_QUASAR)
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
    uint32_t dfb_worker_max,
    uint32_t dfb_worker_sum,
    uint32_t dfb_cur_max,
    uint32_t dfb_prev_max,
    uint32_t dfb_cur_sum,
    uint32_t dfb_prev_sum,
    uint32_t dfb_exp_max_diff,
    uint32_t dfb_exp_max_diff_2,
    uint32_t num_head_tiles) {
    DataflowBuffer dfb_worker_max_obj(dfb_worker_max);
    DataflowBuffer dfb_worker_sum_obj(dfb_worker_sum);
    DataflowBuffer dfb_cur_max_obj(dfb_cur_max);
    DataflowBuffer dfb_prev_max_obj(dfb_prev_max);
    DataflowBuffer dfb_cur_sum_obj(dfb_cur_sum);
    DataflowBuffer dfb_prev_sum_obj(dfb_prev_sum);
    DataflowBuffer dfb_exp_max_diff_obj(dfb_exp_max_diff);
    DataflowBuffer dfb_exp_max_diff_2_obj(dfb_exp_max_diff_2);
    dfb_worker_max_obj.wait_front(num_head_tiles);
    dfb_worker_sum_obj.wait_front(num_head_tiles);
    dfb_prev_max_obj.wait_front(num_head_tiles);
    dfb_prev_sum_obj.wait_front(num_head_tiles);

    dfb_cur_max_obj.reserve_back(num_head_tiles);
    dfb_cur_sum_obj.reserve_back(num_head_tiles);
    dfb_exp_max_diff_obj.reserve_back(num_head_tiles);
    dfb_exp_max_diff_2_obj.reserve_back(num_head_tiles);

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

#ifndef ARCH_QUASAR
    // convert scale from fp32 to bf16
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
#endif

    for (uint32_t i = 0; i < num_head_tiles; i++) {
        tile_regs_acquire();
        copy_init(dfb_worker_max);
#ifndef ARCH_QUASAR
        exp_tile_init<EXP_APPROX_MODE>();
#endif
        copy_tile(dfb_prev_max, i, dst_reg_0);
        copy_tile(dfb_worker_max, i, dst_reg_1);
        copy_tile(dfb_prev_sum, i, dst_reg_3);
        copy_tile(dfb_worker_sum, i, worker_sum_dst);
#ifdef ARCH_QUASAR
        // No fused max/sub/exp/add primitive on Quasar. Decompose into generic dst-to-dst SFPU
        // ops: cur_max = max(prev_max, worker_max); exp_prev/exp_worker = exp((prev/worker_max -
        // cur_max) * scale); cur_sum = exp_prev*prev_sum + exp_worker*worker_sum. Full-tile ops
        // are numerically safe here; only column 0 is consumed downstream. Each op family's _init
        // reprograms shared SFPU state (e.g. LREG polynomial constants), so it must be re-issued
        // immediately before that family's calculate call, not hoisted above the other families.
        binary_max_tile_init();
        binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_2);  // cur_max = max(prev_max, worker_max)
        sub_binary_tile_init();
        sub_binary_tile(dst_reg_0, dst_reg_2, dst_reg_0);  // dst_reg_0 = prev_max - cur_max
        sub_binary_tile(dst_reg_1, dst_reg_2, dst_reg_1);  // dst_reg_1 = worker_max - cur_max
        // Quasar's exp only supports scale=1.0 (see exp_init's static_assert), so scale_fp32
        // (already fp32-encoded) is applied via a separate multiply before an unscaled exp.
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_reg_0, scale_fp32);  // dst_reg_0 = (prev_max - cur_max) * scale
        mul_unary_tile(dst_reg_1, scale_fp32);  // dst_reg_1 = (worker_max - cur_max) * scale
        exp_tile_init<EXP_APPROX_MODE>();
        exp_tile<EXP_APPROX_MODE, false /*scale_en*/>(dst_reg_0);  // dst_reg_0 = exp_prev
        exp_tile<EXP_APPROX_MODE, false /*scale_en*/>(dst_reg_1);  // dst_reg_1 = exp_worker
        mul_binary_tile_init();
        mul_binary_tile(dst_reg_1, worker_sum_dst, worker_sum_dst);  // exp_worker * worker_sum
        mul_binary_tile(dst_reg_0, dst_reg_3, dst_reg_3);            // exp_prev * prev_sum
        add_binary_tile_init();
        add_binary_tile(dst_reg_3, worker_sum_dst, dst_reg_3);  // cur_sum
#else
        MATH((fused_max_sub_exp_add_tile<vector_mode>(0, scale_bf16)));  // WH/BH fused fast path
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_reg_0, dfb_exp_max_diff);
        pack_tile(dst_reg_1, dfb_exp_max_diff_2);
        pack_tile(dst_reg_2, dfb_cur_max);
        pack_tile(dst_reg_3, dfb_cur_sum);
        tile_regs_release();
        dfb_cur_max_obj.push_back(1);
        dfb_cur_sum_obj.push_back(1);
        dfb_exp_max_diff_obj.push_back(1);
        dfb_exp_max_diff_2_obj.push_back(1);
    }
    dfb_prev_sum_obj.pop_front(num_head_tiles);
    dfb_worker_sum_obj.pop_front(num_head_tiles);
}

/**
 * in_dfb -> out_dfb
 */
template <bool pop_in_dfb>
void move_block(uint32_t in_dfb, uint32_t out_dfb, uint32_t num_tiles) {
    DataflowBuffer dfb_in(in_dfb);
    DataflowBuffer dfb_out(out_dfb);
    // Precondition: in_dfb has num_tiles produced
    // Precondition: out_dfb has num_tiles free
    // Postcondition: in_dfb has num_tiles consumed
    // Postcondition: out_dfb has num_tiles produced

    copy_init(in_dfb);

    dfb_in.wait_front(num_tiles);
    dfb_out.reserve_back(num_tiles);

#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_dfb, i, 0 /*dst*/);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_dfb);
        tile_regs_release();
        dfb_out.push_back(1);
    }
    if (pop_in_dfb) {
        dfb_in.pop_front(num_tiles);
    }
}

/**
 * Merged online-softmax running-sum update, fused: sum = cur + prev * emd, on a single merged sum DFB
 * (depth 3*num_tiles) that holds prev at the ring front [0, num_tiles) and cur behind it at
 * [cur_offset, cur_offset + num_tiles), plus a separate exp_max_diff (emd) DFB.
 *
 * Why merged + re-based: the two sum blocks sit contiguously in the ring exactly as their producer
 * (sub_exp_block_bcast_cols_inplace) left them — prev at the front, cur reserve_back'd right behind it.
 * WH tile addressing is fifo_rd_ptr + page_size*idx with NO ring wrap, so the cur offset read is only
 * valid while prev and cur are physically contiguous. This pass computes the running sum into a fresh
 * reserved region, drops BOTH prev and cur, then move_block's the running sum down to the physical
 * front — re-basing the ring so the next chunk starts at rd_ptr==0. Merging sum (not max) avoids the
 * reduce_c prev==out in-place hazard.
 *
 * DECISION RULE (shared with the sdpa_decode fork's fma_block_merged_sum): a merged-sum fma uses the
 * simpler 2-deep pop-both-repush-to-front form (computes all running[i] in DST, one cur scratch reg)
 * when statistics_tiles + 1 <= dst_size (= fp32_dest_acc_en ? 4 : 8); otherwise it must use THIS
 * 3-deep DST-frugal form. decode qualifies for 2-deep (bf16 dst_size=8, statistics_tiles=1); prefill
 * does NOT (fp32 dst_size=4, Sq_chunk_t=4 => 5 > 4), so it uses this one.
 *
 * DST-frugal (unlike the sdpa_decode fork's fma, which holds num_tiles+1 DST tiles): prefill uses
 * Sq_chunk_t (up to 8) statistics tiles and runs fp32_dest_acc_en on WH, where DST holds only 4 fp32
 * tiles (sdpa_program_factory.cpp: dst_size = fp32?4:8). Holding num_tiles+1 tiles overflows DST and
 * silently corrupts. So this does two single-tile passes with L1 accumulation instead:
 *   pass 1: reserved_running[i] = cur[i]                     (copy)
 *   pass 2: reserved_running[i] += prev[i] * bcast(emd[i])   (mul_bcast_cols + L1-acc pack)
 * bcast_cols because prefill's prev/cur are FULL tiles (the final row reduction runs after the K loop),
 * while emd is a column vector (max diff per row, column 0); a plain mul_tiles would pull emd's
 * undefined columns 1..31 into the later matmul_reduce (catastrophic PCC).
 *
 * emd is read (num_tiles at the front) but NOT popped — the out-accumulator scaling that follows pops
 * it. Each pass holds exactly ONE DST tile.
 *
 * Postcondition: `dfb_sum` holds num_tiles produced (the running sum) at the physical front; prev+cur
 * consumed.
 */
void fma_block_merged_sum(uint32_t dfb_sum, uint32_t dfb_emd, uint32_t cur_offset, uint32_t num_tiles) {
    DataflowBuffer d_sum(dfb_sum);
    DataflowBuffer d_emd(dfb_emd);
    d_sum.wait_front(cur_offset + num_tiles);  // prev@front [0,num_tiles) + cur@[cur_offset,+num_tiles)
    d_emd.wait_front(num_tiles);

    // Reserve the running-sum region behind cur (buffer depth is 3*num_tiles).
    d_sum.reserve_back(num_tiles);

    // Pass 1: reserved_running[i] = cur[i].
    reconfig_data_format_srca(dfb_sum);
    copy_init(dfb_sum);
    pack_reconfig_out(dfb_sum);
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(dfb_sum, cur_offset + i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, dfb_sum, i);  // out-of-order: reserved slot i
        tile_regs_release();
    }

    // Pass 2: reserved_running[i] += prev[i] * bcast(emd[i])  (L1 accumulate onto pass-1 cur[i]).
    reconfig_data_format(dfb_sum, dfb_emd);
    pack_reconfig_out(dfb_sum);
    PACK((llk_pack_reconfig_l1_acc(1)));
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        mul_bcast_cols_init(dfb_sum, dfb_emd);
        mul_tiles_bcast_cols(dfb_sum, dfb_emd, i, i, 0);  // dst[0] = prev[i] * bcast(emd[i])
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, dfb_sum, i);  // reserved slot i, accumulate
        tile_regs_release();
    }
    PACK((llk_pack_reconfig_l1_acc(0)));

    d_sum.push_back(num_tiles);               // running sum produced behind cur
    d_sum.pop_front(cur_offset + num_tiles);  // drop the consumed prev AND cur blocks

    // Re-base: move the running sum (now the ring front) down to the physical front [0,num_tiles) so
    // the next chunk's producer appends [prev|cur] contiguously at rd_ptr==0 (no-wrap addressing).
    reconfig_data_format_srca(dfb_sum);
    pack_reconfig_out(dfb_sum);
    move_block<true>(dfb_sum, dfb_sum, num_tiles);
}

/**
 * out_dfb = in0_dfb @ in1_dfb
 */
ALWI void matmul_blocks(
    const uint32_t& in0_dfb,
    const uint32_t& in1_dfb,
    const uint32_t& out_dfb,
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
    const uint32_t& mask_dfb = 0,
    const uint32_t& zero_dfb = 0) {
    // precondition: in0_dfb has M*K produced
    // precondition: in1_dfb has K*N produced
    // postcondition: in0_dfb is full, in1_dfb is empty
    // postcondition: out_dfb has M*N produced

    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    DataflowBuffer dfb_out(out_dfb);
    DataflowBuffer dfb_mask(mask_dfb);
    DataflowBuffer dfb_zero(zero_dfb);

    matmul_block_init(
        in0_dfb,
        in1_dfb,
        transpose /*transpose*/,
        subblock_w /*ct_dim*/,
        subblock_h /*rt_dim*/,
        in0_block_w /*kt_dim*/);

    const uint32_t output_num_tiles = M * N;
    const uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    const uint32_t in0_subblock_all_cols_num_tiles = subblock_h * N;

    uint32_t in0_index_offset = 0;

    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;

    reconfig_data_format(in1_dfb, in0_dfb);
    dfb_in1.wait_front(K * N);
    dfb_out.reserve_back(output_num_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        dfb_in0.wait_front(in0_wait_tiles);
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_dfb, in1_dfb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            if (add_mask) {
                dfb_mask.wait_front(out_subblock_num_tiles);
                dfb_zero.wait_front(1);
                reconfig_data_format(zero_dfb, mask_dfb);
                add_init(zero_dfb, mask_dfb, true);
                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                    add_tiles(zero_dfb, mask_dfb, 0, i, i);
                }
                reconfig_data_format(in1_dfb, in0_dfb);
                matmul_block_init(in0_dfb, in1_dfb, transpose, subblock_w, subblock_h, in0_block_w);
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = r * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_dfb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;
        // Somewhat granularize the push of in0 subblocks
        dfb_out.push_back(in0_subblock_all_cols_num_tiles);
    }
    dfb_in1.pop_front(K * N);
}

template <uint32_t M>
void matmul_reduce(uint32_t in1_dfb, const uint32_t& out_dfb) {
    DataflowBuffer dfb_in1(in1_dfb);
    DataflowBuffer dfb_out(out_dfb);
    // precondition: in0_dfb has M*K produced
    // precondition: in1_dfb has K*N produced
    // postcondition: in0_dfb is full, in1_dfb is empty
    // postcondition: out_dfb has M*N produced

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
    // (in1_dfb -> SrcA, out_dfb -> SrcB), so establish it before init.
    reconfig_data_format(in1_dfb, out_dfb);
    matmul_block_init(
        out_dfb, in1_dfb, 0 /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    constexpr uint32_t output_num_tiles = M * N;
    constexpr uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    pack_reconfig_out(out_dfb);
    dfb_in1.wait_front(N);
    dfb_out.wait_front(M);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        tile_regs_acquire();

        uint32_t dst_index = 0;
        uint32_t in0_index = 0;
        uint32_t in1_index = 0;

        matmul_block(out_dfb, in1_dfb, in0_index, in1_index, dst_index, 0, subblock_w, subblock_h, in0_block_w);

        tile_regs_commit();
        dfb_out.pop_front(subblock_h);

        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_h; i++) {
            pack_tile(i, out_dfb);
        }
        tile_regs_release();
        dfb_out.push_back(subblock_h);
    }
}

/**
 * Batch-stamp a single tile onto a range of positions in out_dfb using L1 accumulate.
 * Caller must have already called copy_init and llk_pack_reconfig_l1_acc(1).
 *
 * @tparam dst_batch  Max tiles per DST cycle (DST register capacity, typically 8 for fp16b half-sync).
 */
template <uint32_t dst_batch>
void stamp_tile_range_l1_acc(
    uint32_t src_dfb, uint32_t src_tile_idx, uint32_t out_dfb, uint32_t out_offset, uint32_t count) {
    for (uint32_t base = 0; base < count; base += dst_batch) {
        uint32_t batch = (count - base < dst_batch) ? (count - base) : dst_batch;
        tile_regs_acquire();
        for (uint32_t i = 0; i < batch; i++) {
            copy_tile(src_dfb, src_tile_idx, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < batch; i++) {
            pack_tile<true>(i, out_dfb, out_offset + base + i);
        }
        tile_regs_release();
    }
}

template <uint32_t dst_batch>
void apply_padded_mask_lightweight_runtime(
    uint32_t neginf_dfb,
    uint32_t neginf_tile_idx,
    uint32_t out_dfb,
    uint32_t num_padded,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_base = 0) {  // first out_dfb tile-row of this query band; nonzero when heads span >1 DEST band
    uint32_t start = num_cols - num_padded;

    reconfig_data_format_srca(neginf_dfb);
    pack_reconfig_out(out_dfb);
    copy_init(neginf_dfb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        stamp_tile_range_l1_acc<dst_batch>(
            neginf_dfb, neginf_tile_idx, out_dfb, (row_base + row) * num_cols + start, num_padded);
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Lightweight partial mask: L1-accumulate a partial mask tile (0 for valid, -inf for padded columns)
 * onto the boundary tile position in out_dfb. The partial tile is permanently fronted in the DFB.
 *
 * @param mask_dfb          DFB holding mask tiles, permanently fronted
 * @param partial_tile_idx Index of the partial tile within the DFB
 * @param out_dfb           QK intermediate DFB (already wait-fronted)
 * @param boundary_col     Column index within the chunk where the boundary tile is
 * @param num_cols         Total K tiles per row (Sk_chunk_t)
 * @param num_rows         Q tiles per chunk (Sq_chunk_t)
 */
void apply_partial_mask_lightweight(
    uint32_t mask_dfb,
    uint32_t partial_tile_idx,
    uint32_t out_dfb,
    uint32_t boundary_col,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_base = 0) {  // first out_dfb tile-row of this query band; nonzero when heads span >1 DEST band
    reconfig_data_format_srca(mask_dfb);
    pack_reconfig_out(out_dfb);
    copy_init(mask_dfb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        tile_regs_acquire();
        copy_tile(mask_dfb, partial_tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_dfb, (row_base + row) * num_cols + boundary_col);
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
    uint32_t mask_dfb,
    uint32_t neginf_idx,
    uint32_t diag_idx,
    uint32_t out_dfb,
    uint32_t q_start_tile,
    uint32_t k_start_tile,
    uint32_t num_rows,
    uint32_t num_cols) {
    reconfig_data_format_srca(mask_dfb);
    pack_reconfig_out(out_dfb);
    copy_init(mask_dfb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        uint32_t row_offset = row * num_cols;
        const int32_t q_pos = static_cast<int32_t>(q_start_tile + row);

        const int32_t diag_col = q_pos - static_cast<int32_t>(k_start_tile);
        if (diag_col < 0) {
            // Entire row above diagonal -> stamp all neginf
            stamp_tile_range_l1_acc<dst_batch>(mask_dfb, neginf_idx, out_dfb, row_offset, num_cols);
        } else if (static_cast<uint32_t>(diag_col) < num_cols) {
            // Stamp the diagonal tile
            tile_regs_acquire();
            copy_tile(mask_dfb, diag_idx, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_dfb, row_offset + static_cast<uint32_t>(diag_col));
            tile_regs_release();

            // Stamp neginf tiles to the right of diagonal
            uint32_t neginf_start = static_cast<uint32_t>(diag_col) + 1;
            if (neginf_start < num_cols) {
                stamp_tile_range_l1_acc<dst_batch>(
                    mask_dfb, neginf_idx, out_dfb, row_offset + neginf_start, num_cols - neginf_start);
            }
        }
        // else: diag_col >= num_cols -> entire row below diagonal, no mask needed
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Context for lightweight mask application.
 * All mask tiles reside in a single DFB. This struct stores the pre-resolved mask metadata used when
 * lightweight masking is enabled; enablement itself is controlled by the `lightweight_mask_enabled`
 * template parameter(s), not by default-constructing this context.
 */
struct LightweightMaskContext {
    bool is_causal = false;                       // Causal masking active for this context instance
    uint32_t neginf_tile_idx = 0;                 // Index of -inf tile in the mask DFB
    uint32_t causal_diag_tile_idx = 0;            // Index of causal diagonal tile in the mask DFB
    uint32_t primary_diag_tile_idx = 0;           // Causal diagonal, or sliding-window trailing-primary tile
    uint32_t sliding_leading_prev_tile_idx = 0;   // Index of previous sliding-window leading tile
    uint32_t sliding_leading_tile_idx = 0;        // Index of current sliding-window leading tile in the mask DFB
    uint32_t sliding_trailing_next_tile_idx = 0;  // Index of next sliding-window trailing tile
    uint32_t global_n_padded_tiles = 0;           // Fully padded K tile columns for the last K chunk
    uint32_t global_n_partial_col = 0;            // Column within tile where K padding starts (0 = no partial)
    uint32_t global_n_partial_tile_idx = 0;       // Index of the partial K tile in the mask DFB

    /**
     * Apply the lightweight causal mask for a given K chunk (non-streaming path).
     * K padding on this path is handled by the dense mask, not by this context.
     *
     * @tparam dst_size  DST register capacity (8 for fp16b half-sync, 4 for fp32_dest_acc).
     */
    template <uint32_t dst_size>
    void apply(
        uint32_t dfb_mask_in,
        uint32_t dfb_qk_im,
        uint32_t Sk_chunk_t,
        uint32_t Sq_chunk_t,
        uint32_t q_start_tile,
        uint32_t k_start_tile) const {
        if (is_causal) {
            apply_causal_mask_lightweight<dst_size>(
                dfb_mask_in,
                neginf_tile_idx,
                causal_diag_tile_idx,
                dfb_qk_im,
                q_start_tile,
                k_start_tile,
                Sq_chunk_t,
                Sk_chunk_t);
        }
    }
};

enum SDPAType {
    STANDARD = 0,
    JOINT = 1,
};

/******************************************************************************
 *                             SDPA INNER LOOP                                *
 ******************************************************************************/
/**
 * Use the specialized wrapper functions below instead of calling this directly.
 *
 * Template Parameters:
 * @tparam sdpa_type - SDPA variant: STANDARD or JOINT
 * @tparam dfb_qk_im - QK intermediate buffer
 * @tparam dfb_identity_scale_in - Identity scale buffer
 * @tparam dfb_attention_sink - Attention sink buffer
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
 * @tparam lightweight_mask_enabled - Enables the lightweight mask path (compile-time gated)
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
 * @param dfb_q_in - Query input buffer
 * @param dfb_k_in - Key input buffer
 * @param dfb_v_in - Value input buffer
 * @param dfb_mask_in - Mask input buffer
 * @param dfb_col_identity - Column identity buffer
 * @param dfb_out_im_A - Output intermediate buffer A
 * @param dfb_out_im_B - Output intermediate buffer B
 * @param dfb_max_A - Max buffer A
 * @param dfb_max_B - Max buffer B
 * @param dfb_sum_A - Sum buffer A
 * @param dfb_sum_B - Sum buffer B
 * @param dfb_exp_max_diff - Exp max diff buffer
 * @param dfb_out - Output buffer
 */
template <
    SDPAType sdpa_type,
    uint32_t dfb_qk_im,
    uint32_t dfb_identity_scale_in,
    uint32_t dfb_attention_sink,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    bool use_attention_sink,
    bool use_provided_mask,
    bool use_padded_mask,
    bool use_joint_mask,
    bool is_chunked,
    uint32_t scale_fp32,
    uint32_t sliding_window_size,
    bool lightweight_mask_enabled = false,
    bool use_windowed_narrowing = false,
    uint32_t dfb_windowed_k_range = 0>
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
    const uint32_t v_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t mask_chunk_0,
    const uint32_t mask_chunk_1,
    const uint32_t dfb_q_in,
    const uint32_t dfb_k_in,
    const uint32_t dfb_v_in,
    const uint32_t dfb_mask_in,
    const uint32_t dfb_col_identity,
    const uint32_t dfb_out_im_A,
    const uint32_t dfb_out_im_B,
    const uint32_t dfb_max_A,
    const uint32_t dfb_max_B,
    const uint32_t dfb_sum_A,
    const uint32_t dfb_sum_B,
    const uint32_t dfb_exp_max_diff,
    const uint32_t dfb_out,
    const LightweightMaskContext& lw_mask = {},
    const bool is_causal = false,
    const bool use_zigzag_balancing = false) {
    // Parameter-stable DFB locals. Aliases (dfb_sum_A/B, dfb_max_A/B, dfb_out_im_A/B) are
    // std::swap-mutated below, so they use inline DataflowBuffer(alias).method() at the call
    // sites instead. dfb_out is constructed conditionally near its consumer (cur.out target).
    DataflowBuffer dfb_q_in_obj(dfb_q_in);
    DataflowBuffer dfb_k_in_obj(dfb_k_in);
    DataflowBuffer dfb_v_in_obj(dfb_v_in);
    DataflowBuffer dfb_qk_im_obj(dfb_qk_im);
    DataflowBuffer dfb_attention_sink_obj(dfb_attention_sink);
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;

    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {
        uint32_t q_start_tile = 0;  // First tile of Q chunk (tile units)
        uint32_t q_high_tile = 0;   // STANDARD: upper tile bound for K iteration
        if constexpr (sdpa_type == STANDARD) {
            const uint32_t linear_q_chunk = local_q_start + (q_iter - iter_q_start);
            // Mod is a no-op when the input is per-head ([0, q_num_chunks)) and extracts the
            // per-head q_chunk when it's a flat global index (global Q scheduling spans heads).
            uint32_t q_chunk = remap_q_index(linear_q_chunk, q_num_chunks, use_zigzag_balancing) % q_num_chunks;
            // Get Q chunk
            if constexpr (is_chunked) {
                q_chunk = chunked_q_chunk_offset + q_chunk;
            }
            q_start_tile = q_chunk * Sq_chunk_t;
            if (is_causal) {
                // Clamp to total K-tile extent. Mirrors reader_interleaved's clamp; without
                // both, the reader and compute disagree on K-chunk count when Q-chunk extends
                // past total K (Sq_chunk_t > Skt) → DFB deadlock.
                const uint32_t q_high_unclamped = q_start_tile + Sq_chunk_t;
                q_high_tile = q_high_unclamped < Skt ? q_high_unclamped : Skt;
            } else {
                q_high_tile = Skt;
            }
        }

        // Set up ping pong buffers.
        // STANDARD (no attention sink) merges the running-sum ping-pong into a single 3-deep DFB
        // (dfb_sum_A, depth 3*Sq_chunk_t; its dfb_sum_B is dropped by the factory and aliased to
        // dfb_sum_A). prev sits at the ring front [0,Sq_chunk_t); sub_exp_block_bcast_cols_inplace
        // reserve_back's cur right behind it [Sq_chunk_t,2*Sq_chunk_t) (its out-of-order pack writes to
        // the reserved region, so no producer offset is needed); fma_block_merged_sum reads
        // prev@front + cur@offset and re-bases the running sum to the front (no ring rotation → no WH
        // offset-read wrap). Arch-independent (runs on WH too) and on Quasar is what fits the op inside
        // the 8 self-loop DFB budget. Merging sum (not max) avoids the reduce_c prev==out in-place
        // hazard. JOINT, the streaming path, and the attention-sink block keep two separate sum
        // DFBs. Max stays a separate two-DFB ping-pong on every path.
        constexpr bool merged_sum = (sdpa_type == STANDARD) && !use_attention_sink;
        uint32_t alias_prev_sum = dfb_sum_A;
        uint32_t alias_cur_sum = merged_sum ? dfb_sum_A : dfb_sum_B;
        uint32_t alias_prev_max = dfb_max_A;
        uint32_t alias_cur_max = dfb_max_B;
        uint32_t alias_mm2_prev_out = dfb_out_im_A;
        uint32_t alias_mm2_cur_out = dfb_out_im_B;

        uint32_t k_chunk_end;
        if constexpr (sdpa_type == STANDARD) {
            // loop while k_low < q_high => (k_chunk * Sk_chunk_t) < q_high_tile.
            k_chunk_end = (q_high_tile + Sk_chunk_t - 1) / Sk_chunk_t;
        } else {  // JOINT.
            k_chunk_end = iter_k_chunk_end;
        }

        // Windowed K-range narrowing: this Q chunk's [k_lo, k_hi) comes from the reader's ctrl DFB —
        // read via the UNPACK mailbox so all three TRISCs agree — and overrides BOTH bounds. The
        // reader streams exactly this many K/V chunks and the writer produces exactly this many mask
        // chunks; any disagreement deadlocks the DFBs.
        uint32_t k_chunk_start = iter_k_chunk_start;
        if constexpr (use_windowed_narrowing) {
            DataflowBuffer dfb_k_range_obj(dfb_windowed_k_range);
            dfb_k_range_obj.wait_front(1);
            k_chunk_start = ckernel::read_tile_value(dfb_windowed_k_range, 0, 0);
            k_chunk_end = ckernel::read_tile_value(dfb_windowed_k_range, 0, 1);
            // read_tile_value is a plain L1 load (no UNPACR), so this wait_front->pop_front
            // is bare; dummy_unpack issues an UNPACR_NOP that orders POP after WAIT.
            dummy_unpack(dfb_windowed_k_range);
            dfb_k_range_obj.pop_front(1);
        }

        uint32_t processed_k_chunks = 0;

        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            /**
             * QK = Q_CHUNK @ K_CHUNK
             *
             * matmul_blocks internally waits on both inputs
             */
            // QK = Q @ K^T via the matmul's native SrcA transpose (transpose=true), reading dfb_k_in
            // directly. The srca-transpose LLK on the rebased base provides the SrcA transpose the Quasar
            // matmul unpacker was missing, so the transpose-free kt-staging workaround is gone (WH always
            // had SrcA transpose). matmul_blocks packs via pack_tile naming dfb_qk_im but does not itself
            // reconfig the packer, so re-point it here first.
            reconfig_data_format(dfb_k_in, dfb_q_in);
            pack_reconfig_out(dfb_qk_im);
            matmul_blocks(
                dfb_q_in,
                dfb_k_in,
                dfb_qk_im,
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
            if (is_causal || sliding_window_size > 0) {
                const uint32_t k_high_idx = k_chunk * Sk_chunk_t + Sk_chunk_t;
                // Apply mask if causal overlap or sliding window
                apply_mask = (q_start_tile < k_high_idx) || (sliding_window_size > 0);
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
                reconfig_data_format(dfb_qk_im, dfb_mask_in);
                if constexpr (lightweight_mask_enabled) {
                    // Re-enter reserved state on dfb_qk_im so the lightweight mask can be stamped in-place.
                    // matmul_blocks above already pushed the QK tiles, so tiles_received has been bumped;
                    // without the pop+push cycle below, reduce_c's wait-front would return immediately
                    // and unpack could start before the mask stamps land in L1.
                    // Safe because pop-front only moves rd_ptr (L1 data is untouched), and on a
                    // single-buffered DFB of size qk_chunk_tiles the re-reserved wr_ptr wraps back to the
                    // same physical region holding the QK scores.
                    // Warning: this won't work if dfb_qk_im is double-buffered -- the stamps would land
                    // in the other buffer, leaving the QK scores unmasked.
                    dfb_qk_im_obj.wait_front(Sk_chunk_t * Sq_chunk_t);
                    // bare wait_front->pop_front (the pop/re-reserve cycle only moves the
                    // rd/wr ptr, no op consumes the QK tiles) would trap the Quasar unpacker. dummy_unpack()
                    // orders POP after WAIT via an UNPACR_NOP; reads nothing, no-op on WH/BH.
                    dummy_unpack(dfb_qk_im);
                    dfb_qk_im_obj.pop_front(Sk_chunk_t * Sq_chunk_t);
                    dfb_qk_im_obj.reserve_back(Sk_chunk_t * Sq_chunk_t);
                    lw_mask.template apply<dst_size>(
                        dfb_mask_in, dfb_qk_im, Sk_chunk_t, Sq_chunk_t, q_start_tile, k_chunk * Sk_chunk_t);
                    dfb_qk_im_obj.push_back(Sk_chunk_t * Sq_chunk_t);
                } else {
                    add_block_inplace(dfb_qk_im, dfb_mask_in, qk_chunk_tiles);
                }
            }

            /**
             * reduce_c can perform both reduce_max and eltwise max with previous result.
             * if do_eltwise_max:
             *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
             * else:
             *  cur_max = max(qk, dim=-1)
             *
             * Use the reduce_c overload with cols as a runtime arg which uses standard
             * reduce_tile + binary_max_tile. The overload with cols as a template arg
             * is bf16-only but dfb_qk_im could be fp32.
             */
            reconfig_data_format(dfb_qk_im, dfb_identity_scale_in);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb_qk_im, dfb_identity_scale_in, Sq_chunk_t>(
                alias_cur_max, alias_prev_max, Sk_chunk_t, processed_k_chunks > 0);

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
            sub_exp_block_bcast_cols_inplace<dfb_qk_im, Sq_chunk_t, scale_fp32, true>(
                alias_cur_max, alias_cur_sum, Sk_chunk_t);

            // Reconfigure unpackers: srcA (context 0) = dfb_v_in, srcB (context 1) = dfb_qk_im (operands are swapped in
            // matmul)
            reconfig_data_format(dfb_v_in, dfb_qk_im);
            pack_reconfig_out(alias_mm2_cur_out);

            /* OUT_IM = QK @ V_CHUNK */
            matmul_blocks(
                dfb_qk_im,
                dfb_v_in,
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

            dfb_qk_im_obj.pop_front(qk_chunk_tiles);
            reconfig_data_format(alias_prev_max, alias_cur_max);

            /* OUT_ACC += OUT_IM */
            if (processed_k_chunks > 0) {
                /**
                 * dfb_exp_max_diff = torch.exp((dfb_prev_max - dfb_cur_max) * scale)
                 * Scale is fused into exp again since max is the max of unscaled scores.
                 */
                // sub_exp_block packs its EMD result via a bare pack_tile (no internal pack_reconfig_out),
                // so the caller must select the pack output operand first. On Quasar the packer is
                // sticky-dest: without this it stays latched to the previous op's output (the QK@V matmul)
                // and EMD would be written to that ring instead of dfb_exp_max_diff. (The
                // reconfig_data_format above only sets the UNPACKers.)
                pack_reconfig_out(dfb_exp_max_diff);
                sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, dfb_exp_max_diff, Sq_chunk_t);
                DataflowBuffer(alias_prev_max).pop_front(Sq_chunk_t);

                /**
                 * dfb_cur_sum = dfb_cur_sum + dfb_prev_sum * dfb_exp_max_diff
                 * bcast_cols since max_diff is a column vector and prev_sum is a partial reduction
                 * (the sum of tiles in dim=-1 of QK), still full tiles here.
                 */
                if constexpr (merged_sum) {
                    // prev at ring front, cur behind at offset Sq_chunk_t in the single dfb_sum_A;
                    // fused pass reads both and writes the running sum back as the single front block.
                    fma_block_merged_sum(alias_prev_sum, dfb_exp_max_diff, Sq_chunk_t, Sq_chunk_t);
                } else {
                    mul_tiles_bcast_cols_inplace(alias_prev_sum, dfb_exp_max_diff, Sq_chunk_t);
                    add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);
                }

                /**
                 * alias_mm2_cur_out += alias_mm2_prev_out * dfb_exp_max_diff, as two non-accumulating
                 * ops (the proven sdpa_decode flash form): scale prev in place, then add it into cur.
                 * Do NOT use the fused pack_accumulate=true form here: it L1-accumulates onto the
                 * already-pushed QK@V tiles in alias_mm2_cur_out WITHOUT a reserve_back, relying on the
                 * packer dest landing on those tiles — which the Quasar sticky-dest packer (pack_init
                 * re-latches L1_Dest_addr; WR-relative, no-wrap addressing) mis-targets → the prev-chunk
                 * contribution is lost and compounds per chunk (multi-chunk PCC ~0.03). WH is unaffected
                 * but this two-op form is arch-agnostic and validated on both.
                 */
                mul_block_bcast_cols<Sq_chunk_t, vDHt, true /*immediate_pop*/, false /*pack_accumulate*/>(
                    alias_mm2_prev_out, dfb_exp_max_diff, alias_mm2_prev_out);
                add_block_inplace<true>(alias_mm2_cur_out, alias_mm2_prev_out, out_chunk_tiles);
            }

            // Swap DFB handles to prepare for next iteration.
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
            std::swap(alias_prev_max, alias_cur_max);
            if constexpr (!merged_sum) {
                // Separate sum DFBs: swap the ping-pong halves. Merged: fma_block_merged_sum already
                // re-based the running sum to the front of the single dfb_sum_A, so there is no swap.
                std::swap(alias_prev_sum, alias_cur_sum);
            }

            processed_k_chunks++;
        }

        /**
         * Performs final row-reduction on the partial sum.
         */
        matmul_reduce<Sq_chunk_t>(dfb_col_identity, alias_prev_sum);

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
         *
         * NOTE (Quasar bring-up): the attention-sink path below is NOT brought up on Quasar and is
         * currently unverified there. Llama (the model driving this fork's bring-up) has no attention
         * sinks, so nothing exercises this path today; it is a GPT-OSS-class feature. Two known gaps
         * remain, to be resolved when a sink-using model is ported to this op:
         *   1. Code size: with USE_ATTENTION_SINK=1 the compute kernel overflows the Quasar TRISC
         *      instruction-memory region at -O3 (~32KB kernel vs ~24KB limit), so it does not even
         *      load. Needs an Os/code-size fix for the sink path before it can run on craq-sim.
         *   2. Packer retargeting: the bare-pack helpers in this block (e.g. the sub_exp_block below)
         *      lack a preceding pack_reconfig_out to point the Quasar packer at their output DFB, the
         *      same class of bug fixed in the flash loop. This is correct-by-audit but unverifiable
         *      on craq-sim until gap (1) is resolved (and adding the reconfigs worsens the overflow).
         */
        if constexpr (use_attention_sink) {
            // Treat attention_sink as scores (already scaled)
            // Shape: [Sq_chunk_t, 1] tiles - same per-head sink value broadcast to all query positions

            // 1. Update running max: cur_max = max(prev_max, attention_sink)
            //    This compares the previous max with the sink logit
            reconfig_data_format(dfb_attention_sink, dfb_identity_scale_in);

            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb_attention_sink, dfb_identity_scale_in, Sq_chunk_t, 1>(
                alias_cur_max, alias_prev_max, true);

            // 2. Compute exp((prev_max - cur_max) * scale) to rescale previous statistics
            sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, dfb_exp_max_diff, Sq_chunk_t);
            DataflowBuffer(alias_prev_max).pop_front(Sq_chunk_t);

            // 3. Rescale previous sum: prev_sum *= exp(prev_max - cur_max)
            mul_tiles_bcast_cols_inplace(alias_prev_sum, dfb_exp_max_diff, Sq_chunk_t);
            // 4. Compute exp((attention_sink - cur_max) * scale) and accumulate in cur_sum
            //    This adds the attention sink's contribution to the softmax denominator
            sub_exp_block_bcast_cols_inplace<dfb_attention_sink, Sq_chunk_t, scale_fp32, false>(
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
                alias_mm2_prev_out, dfb_exp_max_diff, alias_mm2_cur_out);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
        }

        /* dfb_cur_sum = 1.0 / dfb_cur_sum */
        recip_block_inplace(alias_prev_sum, Sq_chunk_t);

        /* dfb_out_accumulate_im *= dfb_cur_sum */
        pack_reconfig_out(dfb_out);
        mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, dfb_out);

        // free up dfb_prev_max after K chunks
        DataflowBuffer(alias_prev_max).pop_front(Sq_chunk_t);

        dfb_q_in_obj.pop_front(q_chunk_tiles);

        // Under global Q scheduling the reader pushes one dfb_attention_sink slot per Q iter,
        // so we must drain matching slots inside this loop. Pre-PR the push/pop pair was
        // 1:1 outside the loop; the new cadence is 1:1 per iter.
        if constexpr (use_attention_sink) {
            dfb_attention_sink_obj.pop_front(Sq_chunk_t);
        }
    }
}

/******************************************************************************
 *                          SDPA WRAPPER FUNCTIONS                            *
 ******************************************************************************/

/**
 * Standard SDPA with optional causal masking, attention sink, and sliding window.
 */
template <
    uint32_t dfb_qk_im,
    uint32_t dfb_identity_scale_in,
    uint32_t dfb_attention_sink,
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
    uint32_t sliding_window_size,
    bool lightweight_mask_enabled = false,
    bool use_windowed_narrowing = false,
    uint32_t dfb_windowed_k_range = 0>
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
    const uint32_t v_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t dfb_q_in,
    const uint32_t dfb_k_in,
    const uint32_t dfb_v_in,
    const uint32_t dfb_mask_in,
    const uint32_t dfb_col_identity,
    const uint32_t dfb_out_im_A,
    const uint32_t dfb_out_im_B,
    const uint32_t dfb_max_A,
    const uint32_t dfb_max_B,
    const uint32_t dfb_sum_A,
    const uint32_t dfb_sum_B,
    const uint32_t dfb_exp_max_diff,
    const uint32_t dfb_out,
    const LightweightMaskContext& lw_mask = {},
    const bool use_zigzag_balancing = false) {
    sdpa_inner_loop<
        STANDARD,
        dfb_qk_im,
        dfb_identity_scale_in,
        dfb_attention_sink,
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        vDHt,
        use_attention_sink,
        use_provided_mask,
        use_padded_mask,
        false,  // use_joint_mask (not used)
        is_chunked,
        scale_fp32,
        sliding_window_size,
        lightweight_mask_enabled,
        use_windowed_narrowing,
        dfb_windowed_k_range>(
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
        v_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        0,  // mask_chunk_0 (not used)
        0,  // mask_chunk_1 (not used)
        dfb_q_in,
        dfb_k_in,
        dfb_v_in,
        dfb_mask_in,
        dfb_col_identity,
        dfb_out_im_A,
        dfb_out_im_B,
        dfb_max_A,
        dfb_max_B,
        dfb_sum_A,
        dfb_sum_B,
        dfb_exp_max_diff,
        dfb_out,
        lw_mask,
        is_causal,
        use_zigzag_balancing);
}

/**
 * Joint SDPA for multi-modal attention.
 */
template <
    uint32_t dfb_qk_im,
    uint32_t dfb_identity_scale_in,
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
    const uint32_t dfb_q_in,
    const uint32_t dfb_k_in,
    const uint32_t dfb_v_in,
    const uint32_t dfb_mask_in,
    const uint32_t dfb_col_identity,
    const uint32_t dfb_out_im_A,
    const uint32_t dfb_out_im_B,
    const uint32_t dfb_max_A,
    const uint32_t dfb_max_B,
    const uint32_t dfb_sum_A,
    const uint32_t dfb_sum_B,
    const uint32_t dfb_exp_max_diff,
    const uint32_t dfb_out) {
    sdpa_inner_loop<
        JOINT,
        dfb_qk_im,
        dfb_identity_scale_in,
        0,  // dfb_attention_sink (not used)
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        DHt,    // vDHt = DHt
        false,  // use_attention_sink (not used)
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
        k_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        mask_chunk_0,
        mask_chunk_1,
        dfb_q_in,
        dfb_k_in,
        dfb_v_in,
        dfb_mask_in,
        dfb_col_identity,
        dfb_out_im_A,
        dfb_out_im_B,
        dfb_max_A,
        dfb_max_B,
        dfb_sum_A,
        dfb_sum_B,
        dfb_exp_max_diff,
        dfb_out);
}
