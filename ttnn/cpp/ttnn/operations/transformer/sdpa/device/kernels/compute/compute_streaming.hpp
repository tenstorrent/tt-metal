// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Streaming SDPA compute helpers.
// Included only by sdpa.cpp when use_streaming_compute is true.
// Depends on primitives from compute_common.hpp (must be included first).

#pragma once

#include <type_traits>

// Bundles the subblock decomposition of active_Sk into full subblocks + remainder,
// avoiding long parameter lists in sdpa_inner_loop_step.
struct SubblockDecomp {
    uint32_t num_full;   // number of full subblocks (active_Sk / qkt_subblock_w)
    uint32_t remainder;  // leftover tiles (active_Sk % qkt_subblock_w)
    bool has_partial;    // remainder > 0
};

#ifdef ARCH_BLACKHOLE
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
// BH has ample code size headroom; let GCC clone (IPA-CP) without inlining.
#define SDPA_NOINLINE
#else
// WH/T3000: Mochi leaves only ~12 KB headroom — prevent inlining and cloning.
#define SDPA_NOINLINE __attribute__((noinline, noclone))
#endif
#include "tools/profiler/kernel_profiler.hpp"

// Template-driven profiling: MaybeDeviceZoneScopedN(ENABLED, name)
// When ENABLED=true: RAII profileScope writes timestamps (same as DeviceZoneScopedN)
// When ENABLED=false: empty struct, zero overhead (compiler eliminates entirely)
#if defined(PROFILE_KERNEL)
template <bool Enabled, uint32_t timer_id>
struct MaybeProfileScope {
    inline __attribute__((always_inline)) MaybeProfileScope() {}
    inline __attribute__((always_inline)) ~MaybeProfileScope() {}
};
template <uint32_t timer_id>
struct MaybeProfileScope<true, timer_id> : kernel_profiler::profileScope<timer_id> {};

#define MaybeDeviceZoneScopedN(ENABLED, name)                                  \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    MaybeProfileScope<ENABLED, hash> zone;
#else
#define MaybeDeviceZoneScopedN(ENABLED, name)
#endif

/**
 * Push tiles to make them visible for UNPACK reads, but rewind wr_ptr so
 * subsequent pack_tile<true> offsets remain relative to a stable base.
 * This eliminates the need for separate row buffers.
 */
ALWI void cb_push_back_hold_wr_ptr(uint32_t cb_id, uint32_t num_tiles) {
    cb_push_back(cb_id, num_tiles);
    PACK(({
        auto& intf = get_local_cb_interface(cb_id);
        intf.fifo_wr_ptr -= num_tiles * intf.fifo_page_size;
        uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;
        }
    }));
}

// Accumulator half: one side of the ping-pong buffer (sum, max, output CB indices).
struct AccumulatorHalf {
    uint32_t sum, max, out;
};

// Persistent accumulator state for ring SDPA deferred normalization.
// When each core processes exactly 1 Q chunk, this state carries across ring iterations.
struct RingAccumulatorState {
    AccumulatorHalf prev, cur;
};

/**
 * Blocked subblock matmul with absolute offset packing.
 * Always uses pack_tile<true> at row-major positions in out_cb.
 */
template <bool TRANSPOSE, uint32_t IN1_STRIDE, uint32_t OUT_NUM_COLS, bool blocked_pack = false>
SDPA_NOINLINE void blocked_matmul_and_pack(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t q_subblock,
    uint32_t out_col_offset,
    uint32_t SUBBLOCK_W,
    uint32_t SUBBLOCK_H,
    uint32_t INNER_DIM,
    uint32_t KT_DIM_MATMUL = 0,
    bool trigger_reduce = false) {
    if (KT_DIM_MATMUL == 0) {
        KT_DIM_MATMUL = INNER_DIM;
    }
    tile_regs_acquire();
    uint32_t dst_index = 0;
    uint32_t in0_index = in0_index_start;
    uint32_t in1_index = in1_index_start;
    for (uint32_t inner = 0; inner < INNER_DIM; ++inner) {
#ifdef ARCH_BLACKHOLE
        matmul_block_no_mop(
            in0_cb, in1_cb, in0_index, in1_index, dst_index, TRANSPOSE, SUBBLOCK_W, SUBBLOCK_H, KT_DIM_MATMUL);
#else
        matmul_block(in0_cb, in1_cb, in0_index, in1_index, dst_index, TRANSPOSE, SUBBLOCK_W, SUBBLOCK_H, KT_DIM_MATMUL);
#endif
        in0_index++;
        in1_index += IN1_STRIDE;
    }
    tile_regs_commit();

    tile_regs_wait();
    uint32_t dst_idx = 0;
#ifdef ARCH_BLACKHOLE
    if constexpr (blocked_pack) {
        for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
            uint32_t out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS;
            pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset);
            dst_idx += SUBBLOCK_W;
        }
    } else
#endif
    {
        for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
            uint32_t out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS;
            for (uint32_t c = 0; c < SUBBLOCK_W; c++) {
                pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                dst_idx++;
            }
        }
    }
    if (trigger_reduce) {
        PACK((t6_semaphore_post<p_stall::NONE>(semaphore::FPU_SFPU)));
    }
    tile_regs_release();
}

/**
 * Per-row-group max reduction with optional eltwise_max against prev values.
 * Reads from in0_cb at row group offset, writes to out_cb sequentially.
 */
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t scale_cb, uint32_t cols, uint32_t ROW_STRIDE = cols>
void reduce_c_row_group(
    uint32_t in0_cb,
    uint32_t out_cb,
    uint32_t prev_cb,
    uint32_t row_group_index,
    bool do_eltwise_max,
    uint32_t SBH,
    uint32_t reduce_cols = cols,
    bool respect_trigger = false) {
    const uint32_t GROUP_SIZE = SBH;
    const uint32_t row_start = row_group_index * GROUP_SIZE;

    // ROW_STRIDE: physical row width in the CB (may exceed cols on the reduced path).
    const uint32_t cumulative_input_tiles = (row_group_index + 1) * GROUP_SIZE * ROW_STRIDE;
    const uint32_t cumulative_prev_tiles = (row_group_index + 1) * GROUP_SIZE;

    // scale_cb assumed ready (waited once at kernel init)
    // cb_wait_front(scale_cb, 1);

    tile_regs_acquire();

    if (do_eltwise_max) {
        cb_wait_front(prev_cb, cumulative_prev_tiles);
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < GROUP_SIZE; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
    }

    // Deferred: wait for in0_cb just before its first use (reduce_block_max_row).
    // When do_eltwise_max=true, the prev_cb wait + copy_tile work above can overlap
    // with in0_cb data arrival.
    // When respect_trigger=true, the unpack MOP is split into two halves with a
    // HW semaphore wait in between, so we don't need cb_wait_front here.
    if (!respect_trigger) {
        cb_wait_front(in0_cb, cumulative_input_tiles);
    }

    reduce_block_max_row_init_runtime(reduce_cols, respect_trigger);
    for (uint32_t i = 0; i < GROUP_SIZE; i++) {
        const uint32_t input_tile_start = (row_start + i) * ROW_STRIDE;
        reduce_block_max_row_runtime(in0_cb, scale_cb, input_tile_start, i, respect_trigger);
    }
    reduce_block_max_row_uninit_runtime(in0_cb, respect_trigger);

    tile_regs_commit();
    tile_regs_wait();

    for (uint32_t i = 0; i < GROUP_SIZE; i++) {
        pack_tile<false>(i, out_cb);
    }

    tile_regs_release();
}

/**
 * In-place sub_exp on cb_qkt_im: subtracts max, applies exp with ReLU clamping,
 * writes back to same positions. Accumulates row sums into reduce_cb.
 */
template <
    bool PROFILING_ENABLED,
    uint32_t scale_fp32,
    bool do_reduce,
    int vector_mode = (int)VectorMode::RC,
    bool blocked_pack = false>
SDPA_NOINLINE void sub_exp_block_bcast_cols(
    uint32_t inout_cb,
    uint32_t max_cb,
    uint32_t reduce_cb,
    uint32_t cols_in_row,
    uint32_t q_subblock,
    uint32_t global_col_base,
    uint32_t SBH,
    uint32_t SBW) {
    const uint32_t tiles_per_row = SBH;
    const uint32_t tiles_per_column = SBW;
    const uint32_t max_row_base = q_subblock * tiles_per_row;

    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "SUB_EXP_BLOCK_INIT");
#ifdef ARCH_BLACKHOLE
        sub_bcast_cols_init_short_custom(inout_cb, max_cb, tiles_per_column);
#else
        sub_bcast_cols_init_short(inout_cb, max_cb);
#endif
        PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
    }

    // inout_cb assumed ready (max_cb was already computed from it)
    // cb_wait_front(inout_cb, (q_subblock + 1) * tiles_per_row * cols_in_row);
    cb_wait_front(max_cb, (q_subblock + 1) * tiles_per_row);

    tile_regs_acquire();
    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "SUB");
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
#ifdef ARCH_BLACKHOLE
            uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base;
            sub_tiles_bcast_cols_custom(
                inout_cb, max_cb, in0_tile_index, max_row_base + i, dst_index, tiles_per_column);
            dst_index += tiles_per_column;
#else
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base + j;
                sub_tiles_bcast_cols(inout_cb, max_cb, in0_tile_index, max_row_base + i, dst_index++);
            }
#endif
        }
    }
    tile_regs_commit();

    tile_regs_wait();
    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "EXP");
        uint32_t dst_index = 0;
        constexpr int iterations = (vector_mode == (int)VectorMode::RC) ? 32 : 8;
        constexpr int vector_mode_exp = (vector_mode == (int)VectorMode::RC) ? (int)VectorMode::None : vector_mode;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                exp_packthread_tile<true, true, false, false, InputClamping::None, iterations>(
                    dst_index++, vector_mode_exp);
            }
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }

    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "PACK SUB_EXP");
        // Pack back to inout_cb at the same absolute positions
        uint32_t dst_index = 0;
#ifdef ARCH_BLACKHOLE
        if constexpr (blocked_pack) {
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base;
                pack_tile<true>(dst_index, inout_cb, in0_tile_index);
                dst_index += tiles_per_column;
            }
        } else
#endif
        {
#pragma GCC unroll 1
            for (uint32_t i = 0; i < tiles_per_row; i++) {
#pragma GCC unroll 1
                for (uint32_t j = 0; j < tiles_per_column; ++j) {
                    uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base + j;
                    pack_tile<true>(dst_index++, inout_cb, in0_tile_index);
                }
            }
        }

        // Reduce to reduce_cb: first tile of first kt_subblock overwrites, rest accumulate
        if constexpr (do_reduce) {
#ifdef ARCH_BLACKHOLE
            if constexpr (blocked_pack) {
                PACK((llk_pack_mop_config<false, false, false>(reduce_cb, 1)));
            }
#endif
            dst_index = 0;
#pragma GCC unroll 1
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                if (global_col_base > 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                } else {
                    PACK((llk_pack_reconfig_l1_acc(0)));
                }
#pragma GCC unroll 1
                for (uint32_t j = 0; j < tiles_per_column; ++j) {
                    pack_tile<true>(dst_index++, reduce_cb, max_row_base + i);
                    if (global_col_base == 0 && j == 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
        }
    }

    tile_regs_release();

    // Restore packer ReLU config after all exp operations complete
    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
    if constexpr (do_reduce) {
#ifdef ARCH_BLACKHOLE
        if constexpr (blocked_pack) {
            PACK((llk_pack_mop_config<false, false, false>(reduce_cb, tiles_per_column)));
        }
#endif
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
}

/**
 * Column-only exp(prev_max - cur_max) for SALAD corrections.
 * Operates on first-column subset of tiles.
 */
template <bool PROFILING_ENABLED, uint32_t scale_fp32>
SDPA_NOINLINE void sub_exp_first_col_blocks(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t SBH) {
    const uint32_t tiles_per_row = SBH;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    sub_tiles_init(in0_cb, in1_cb);

    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    {
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t tile_index = global_row_base + i;
            sub_tiles(in0_cb, in1_cb, tile_index, tile_index, i);
        }
        tile_regs_commit();
    }

    {
        tile_regs_wait();
        for (uint32_t dst_index = 0; dst_index < tiles_per_row; dst_index++) {
            PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(dst_index)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

        for (uint32_t i = 0; i < tiles_per_row; i++) {
            pack_tile<false>(i, out_cb);
        }

        tile_regs_release();
    }
}

/**
 * SALAD sum correction: out_cb[row] += in0_cb[row] * bcast_cols(in1_cb[row])
 * with L1 accumulate at explicit offset.
 */
inline void mul_bcast_cols_l1_acc(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t write_q_subblock, uint32_t SBH) {
    const uint32_t tiles_per_row = SBH;
    const uint32_t read_row_base = q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;

    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        uint32_t src_tile_index = read_row_base + i;
        mul_tiles_bcast_cols(in0_cb, in1_cb, src_tile_index, src_tile_index, i);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        pack_tile<true>(i, out_cb, write_row_base + i);
    }
    tile_regs_release();
}

/**
 * SALAD output correction: block multiply with bcast_cols and L1 accumulate.
 */
template <uint32_t SBW>
void mul_block_bcast_cols_acc(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t write_q_subblock, uint32_t SBH) {
    const uint32_t tiles_per_row = SBH;
    constexpr uint32_t tiles_per_column = SBW;
    // Process columns in batches that fit SBH rows in DST (8 tiles max).
    const uint32_t COL_BATCH = (8 / SBH < SBW) ? 8 / SBH : SBW;
    const uint32_t read_row_base = q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;
    mul_bcast_cols_init_short(in0_cb, in1_cb);

    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row * tiles_per_column);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    for (uint32_t col_base = 0; col_base < tiles_per_column; col_base += COL_BATCH) {
        const uint32_t last_batch = tiles_per_column % COL_BATCH;
        const uint32_t cur_cols =
            (col_base + COL_BATCH <= tiles_per_column) ? COL_BATCH : (last_batch > 0 ? last_batch : COL_BATCH);
        tile_regs_acquire();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < cur_cols; j++) {
                uint32_t in0_tile_index = (read_row_base + i) * tiles_per_column + col_base + j;
                mul_tiles_bcast_cols(in0_cb, in1_cb, in0_tile_index, read_row_base + i, dst_index++);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        dst_index = 0;
#ifdef ARCH_BLACKHOLE
        PACK((llk_pack_mop_config<false, false, false>(out_cb, cur_cols)));
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t out_tile_index = (write_row_base + i) * tiles_per_column + col_base;
            pack_tile<true>(dst_index, out_cb, out_tile_index);
            dst_index += cur_cols;
        }
        PACK((llk_pack_mop_config<false, false, false>(out_cb, 1)));
#else
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < cur_cols; j++) {
                uint32_t out_tile_index = (write_row_base + i) * tiles_per_column + col_base + j;
                pack_tile<true>(dst_index++, out_cb, out_tile_index);
            }
        }
#endif
        tile_regs_release();
    }
}

/**
 * Per-row streaming normalization: matmul_reduce + recip-in-DST + mul_bcast_cols.
 * Consumes (pops) sum and output tiles, writes normalized output.
 * scratch_cb is a 1-tile CB reused for the reciprocal intermediate.
 */
template <bool PROFILING_ENABLED, uint32_t head_dim_t_>
void normalize_row_streaming(
    uint32_t cur_sum_cb,
    uint32_t cur_out_cb,
    uint32_t col_identity_cb,
    uint32_t scratch_cb,
    uint32_t normalized_out_cb,
    uint32_t SBH) {
    for (uint32_t s = 0; s < SBH; s++) {
        // 1+2. Fused matmul_reduce + recip: sum × col_identity → recip → 1/sum in scratch
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "NORM_MATMUL_RECIP");
            constexpr uint32_t N = 1;
            mm_block_init_short(cur_sum_cb, col_identity_cb, 0, N, 1, N);
            reconfig_data_format(col_identity_cb, cur_sum_cb);

            cb_wait_front(col_identity_cb, N);
            cb_wait_front(cur_sum_cb, 1);

            cb_reserve_back(scratch_cb, 1);
            tile_regs_acquire();
            matmul_block(cur_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
#ifdef ARCH_BLACKHOLE
            recip_tile_init<false>();
            MATH((recip_tile<false>(0, (int)VectorMode::C)));
#else
            recip_tile_init();
            MATH((recip_tile_first_column(0)));
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, scratch_cb);
            tile_regs_release();
            cb_push_back(scratch_cb, 1);

            cb_pop_front(cur_sum_cb, 1);
        }

        // 3. Normalize: multiply output tiles by bcast_cols(1/sum)
        // Process in batches of up to 8 tiles (DST capacity for fp16b half-sync).
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "NORM_MUL_BCAST");
            constexpr uint32_t BATCH = (head_dim_t_ < 8) ? head_dim_t_ : 8;
            mul_bcast_cols_init_short(cur_out_cb, scratch_cb);
            cb_wait_front(cur_out_cb, head_dim_t_);
            cb_wait_front(scratch_cb, 1);

            cb_reserve_back(normalized_out_cb, head_dim_t_);
            for (uint32_t base = 0; base < head_dim_t_; base += BATCH) {
                constexpr uint32_t last_batch = head_dim_t_ % BATCH;
                const uint32_t cur_batch = (base + BATCH <= head_dim_t_) ? BATCH : last_batch;
                tile_regs_acquire();
                for (uint32_t j = 0; j < cur_batch; ++j) {
                    mul_tiles_bcast_cols(cur_out_cb, scratch_cb, base + j, 0, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < cur_batch; ++j) {
                    pack_tile(j, normalized_out_cb);
                }
                tile_regs_release();
            }
            cb_push_back(normalized_out_cb, head_dim_t_);

            cb_pop_front(scratch_cb, 1);
            cb_pop_front(cur_out_cb, head_dim_t_);
        }
    }
}

// ===================== Streaming SDPA Core Functions =====================

/**
 * L1-accumulate ring mask tiles onto QKT for one row group (q_subblock).
 * Copies mask tiles from cb_mask_in and adds them in-place to cb_qkt_im using L1 accumulation.
 */
template <uint32_t Sk_chunk_t, uint32_t cb_mask_in, uint32_t cb_qkt_im>
static void apply_ring_mask_to_qkt(uint32_t q_subblock, uint32_t sbh) {
    constexpr uint32_t DST_BATCH = 8;
    for (uint32_t row = 0; row < sbh; row++) {
        uint32_t out_row_offset = (q_subblock * sbh + row) * Sk_chunk_t;
        uint32_t mask_row_offset = (q_subblock * sbh + row) * Sk_chunk_t;
        copy_tile_to_dst_init_short(cb_mask_in);
        PACK((llk_pack_reconfig_l1_acc(1)));
        for (uint32_t base = 0; base < Sk_chunk_t; base += DST_BATCH) {
            uint32_t batch = (Sk_chunk_t - base < DST_BATCH) ? (Sk_chunk_t - base) : DST_BATCH;
            tile_regs_acquire();
            for (uint32_t i = 0; i < batch; i++) {
                copy_tile(cb_mask_in, mask_row_offset + base + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < batch; i++) {
                pack_tile<true>(i, cb_qkt_im, out_row_offset + base + i);
            }
            tile_regs_release();
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
}

/**
 * L1-accumulate a single mask tile onto one position in out_cb.
 * Minimal primitive used by the lightweight ring mask path.
 */
static inline void l1_acc_single_tile(uint32_t mask_cb, uint32_t tile_idx, uint32_t out_cb, uint32_t out_pos) {
    tile_regs_acquire();
    copy_tile(mask_cb, tile_idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, out_cb, out_pos);
    tile_regs_release();
}

/**
 * Combined lightweight mask for streaming ring SDPA: applies partial + padded mask.
 * Processes SBH rows per call (one q_subblock).
 * Caller must set up copy_tile_to_dst_init_short and llk_pack_reconfig_l1_acc(1) before calling,
 * and llk_pack_reconfig_l1_acc(0) after calling.
 */
template <uint32_t num_cols>
static SDPA_NOINLINE void apply_lightweight_mask_streaming(
    uint32_t mask_cb,
    uint32_t out_cb,
    uint32_t q_subblock,
    uint32_t num_padded,
    bool has_partial,
    uint32_t partial_tile_idx,
    uint32_t SBH) {
    uint32_t boundary_col = num_cols - num_padded - (has_partial ? 1 : 0);
    for (uint32_t row = 0; row < SBH; row++) {
        uint32_t row_offset = (q_subblock * SBH + row) * num_cols;

        if (has_partial) {
            l1_acc_single_tile(mask_cb, partial_tile_idx, out_cb, row_offset + boundary_col);
        }

        uint32_t start = num_cols - num_padded;
        for (uint32_t col = start; col < num_cols; col++) {
            l1_acc_single_tile(mask_cb, 0, out_cb, row_offset + col);  // neginf is always tile 0
        }
    }
}

/**
 * One K-chunk iteration of the streaming SDPA algorithm (v2 — no row buffers).
 * Phase 1: Q@KT directly into cb_qkt_im with cb_push_back_hold_wr_ptr, in-place sub_exp.
 * Phase 2: Drain + QKT@V with SALAD corrections, streaming normalization on last K iter.
 */
template <
    bool PROFILING_ENABLED,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t subblock_h,
    uint32_t qkt_subblock_w,
    bool use_padded_mask,
    bool ring_mode = false,
    bool use_ring_mask = false,

    bool uniform_dataformat = false,
    uint32_t cb_q_in = 0,
    uint32_t cb_kt_in = 0,
    uint32_t cb_v_in = 0,
    uint32_t cb_qkt_im = 0,
    uint32_t cb_identity_scale_in = 0,
    uint32_t cb_exp_max_diff = 0,
    uint32_t cb_col_identity = 0,
    uint32_t cb_recip_scratch = 0,
    uint32_t cb_normalized_out = 0,
    uint32_t cb_mask_in = 0,
    uint32_t KT_stride = Sk_chunk_t>
static void sdpa_inner_loop_step(
    AccumulatorHalf& prev,
    AccumulatorHalf& cur,
    const bool is_last_iter,
    const bool is_first_iter,
    [[maybe_unused]] const bool apply_mask = false,
    const uint32_t lw_partial_tile_idx = 0,
    const uint32_t active_Sk = Sk_chunk_t,
    const bool reduce_trigger = false,
    const SubblockDecomp kt_decomp = {
        Sk_chunk_t / qkt_subblock_w, Sk_chunk_t % qkt_subblock_w, (Sk_chunk_t % qkt_subblock_w) > 0}) {
    const uint32_t kt_num_full_subblocks = kt_decomp.num_full;
    const uint32_t kt_remainder = kt_decomp.remainder;
    const bool has_partial_subblock = kt_decomp.has_partial;
    constexpr uint32_t in0_block_w = DHt;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / subblock_h;
    constexpr uint32_t q_subblock_num_tiles = subblock_h * in0_block_w;
    constexpr uint32_t row_tiles = subblock_h * KT_stride;  // Use KT_stride for cb_qkt_im row width

    static_assert(!(use_padded_mask && use_ring_mask), "use_padded_mask and use_ring_mask are mutually exclusive");

    // When all CBs share the same data format (e.g. all Float16_b for bf16 models),
    // reconfig calls are no-ops — skip entirely to save code size.
    // uniform_dataformat is passed as a template parameter from the program factory.
    constexpr bool uniform_unpack_format = uniform_dataformat;
    constexpr bool uniform_pack_format = uniform_dataformat;

    uint32_t pushed_rows = 0;
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    exp_packthread_tile_init<true, true, scale_fp32, InputClamping::None>();

    if constexpr (!uniform_pack_format) {
        pack_reconfig_data_format(cb_qkt_im);
    }
    if constexpr (!uniform_unpack_format) {
        reconfig_data_format(cb_kt_in, cb_q_in);
    }
    // Use KT_stride for cb_qkt_im layout to keep CB pointers aligned across iterations
    cb_reserve_back(cb_qkt_im, Sq_chunk_t * KT_stride);

    cb_reserve_back(cur.sum, Sq_chunk_t);

    // ========== PHASE 1: Q@KT directly into cb_qkt_im ==========
    // All matmul output goes to cb_qkt_im at absolute offsets via pack_tile<true>.
    // cb_push_back_hold_wr_ptr makes each row visible to UNPACK without advancing wr_ptr.
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_mop_config<false, false, false>(cb_qkt_im, qkt_subblock_w)));
#endif
    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)");
        cb_wait_front(cb_q_in, q_wait_tiles);
        // Deferred KT wait: reader pushes Q before KT, so waiting for Q first
        // allows reserves + MOP config + Q wait to overlap with KT DMA.
        if (q_subblock == 0) {
            cb_wait_front(cb_kt_in, DHt * KT_stride);
        }
        kt_index_offset = 0;
#ifdef ARCH_BLACKHOLE
        mm_no_mop_init_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, subblock_h, in0_block_w);
#else
        mm_block_init_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, subblock_h, in0_block_w);
#endif
        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_full_subblocks; ++kt_subblock) {
            if (q_subblock > 0) {
                uint32_t prev_q_subblock = q_subblock - 1;
                if constexpr (!uniform_unpack_format) {
                    reconfig_data_format(cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im);
                }
                sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, true, VectorMode::RC, true /*blocked_pack*/>(
                    cb_qkt_im,
                    cur.max,
                    cur.sum,
                    KT_stride,
                    prev_q_subblock,
                    kt_subblock * qkt_subblock_w,
                    subblock_h,
                    qkt_subblock_w);
#ifdef ARCH_BLACKHOLE
                mm_no_mop_reinit_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, subblock_h, in0_block_w);
#else
                mm_block_init_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, subblock_h, in0_block_w);
#endif
            }
            {
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Q@KT MM+Pack");
                if constexpr (!uniform_unpack_format) {
                    reconfig_data_format(cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in);
                }
                // Only the LAST full subblock posts the semaphore — one post per reduce,
                // matching the single semaphore get in the UNPACK MOP split.
                bool kt_trigger_reduce = reduce_trigger && (kt_subblock == kt_num_full_subblocks - 1);
                blocked_matmul_and_pack<true, KT_stride, KT_stride, true /*blocked_pack*/>(
                    cb_q_in,
                    cb_kt_in,
                    cb_qkt_im,
                    q_index_offset,
                    kt_index_offset,
                    q_subblock,
                    kt_subblock * qkt_subblock_w,
                    qkt_subblock_w,
                    subblock_h,
                    in0_block_w,
                    0,
                    kt_trigger_reduce);
                kt_index_offset += qkt_subblock_w;
            }
        }
        // Partial last subblock: handles tile-level tail when active_Sk % qkt_subblock_w != 0.
        if (has_partial_subblock) {
            {
#ifdef ARCH_BLACKHOLE
                PACK((llk_pack_mop_config<false, false, false>(cb_qkt_im, kt_remainder)));
#endif
                if (q_subblock > 0) {
                    uint32_t prev_q_subblock = q_subblock - 1;
                    if constexpr (!uniform_unpack_format) {
                        reconfig_data_format(cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im);
                    }
                    sub_exp_block_bcast_cols<
                        PROFILING_ENABLED,
                        scale_fp32,
                        true,
                        VectorMode::RC,
                        true /*blocked_pack*/>(
                        cb_qkt_im,
                        cur.max,
                        cur.sum,
                        KT_stride,
                        prev_q_subblock,
                        kt_num_full_subblocks * qkt_subblock_w,
                        subblock_h,
                        kt_remainder);
                }
                {
                    MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Q@KT MM+Pack (partial)");
#ifdef ARCH_BLACKHOLE
                    mm_no_mop_init_short(cb_q_in, cb_kt_in, true, kt_remainder, subblock_h, in0_block_w);
#else
                    mm_block_init_short(cb_q_in, cb_kt_in, true, kt_remainder, subblock_h, in0_block_w);
#endif
                if constexpr (!uniform_unpack_format) {
                    reconfig_data_format(cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in);
                }
                // Partial subblock is always the last subblock — never trigger custom reduce sync.
                // The reduce trigger mechanism is only for full subblocks.
                constexpr bool partial_trigger_reduce = false;
                blocked_matmul_and_pack<true, KT_stride, KT_stride, true /*blocked_pack*/>(
                    cb_q_in,
                    cb_kt_in,
                    cb_qkt_im,
                    q_index_offset,
                    kt_index_offset,
                    q_subblock,
                    kt_num_full_subblocks * qkt_subblock_w,
                    kt_remainder,
                    subblock_h,
                    in0_block_w,
                    0,
                    partial_trigger_reduce);
                }
            }
        }
        // Restore float16b for mask/reduce after Q@KT.
        if constexpr (!uniform_unpack_format) {
            reconfig_data_format(cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im);
        }

        // Ring mask: L1-accumulate partial-tile mask onto cb_qkt_im for this row group.
        // Full-tile padding handled by active_Sk narrowing (reduce/sub_exp/V skip padded tiles),
        // but num_padded must still reflect the actual count so boundary_col places the partial
        // mask at active_Sk-1 instead of Sk_chunk_t-1.
        if constexpr (use_ring_mask) {
            if (apply_mask && lw_partial_tile_idx > 0) {
                copy_tile_to_dst_init_short(cb_mask_in);
                PACK((llk_pack_reconfig_l1_acc(1)));
                apply_lightweight_mask_streaming<KT_stride>(
                    cb_mask_in,
                    cb_qkt_im,
                    q_subblock,
                    Sk_chunk_t - active_Sk,  // padded tile count for correct boundary_col
                    true,                    // has_partial — guaranteed by outer guard (lw_partial_tile_idx > 0)
                    lw_partial_tile_idx,
                    subblock_h);
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
        }

        // Push row (visible for UNPACK reads) but keep wr_ptr stable
        cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles);

        // Max reduce: reads from cb_qkt_im at q_subblock position
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Reduce max");
            cb_reserve_back(cur.max, subblock_h);
#ifdef ARCH_BLACKHOLE
            PACK((llk_pack_mop_config<false, false, false>(cur.max, 1)));
#endif
            // Use reduce_trigger to enable early reduce start (before all matmul output is ready).
            // When reduce_trigger=true, the packer signals the unpacker via semaphore after partial output.
            reduce_c_row_group<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_identity_scale_in, Sk_chunk_t, KT_stride>(
                cb_qkt_im,
                cur.max,
                prev.max,
                q_subblock,
                !is_first_iter /*do_eltwise_max*/,
                subblock_h,
                active_Sk,
                reduce_trigger);
            cb_push_back(cur.max, subblock_h);
#ifdef ARCH_BLACKHOLE
            PACK((llk_pack_mop_config<false, false, false>(cur.max, qkt_subblock_w)));
#endif
        }

        q_index_offset += subblock_h * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }

    cb_pop_front(cb_kt_in, DHt * KT_stride);

    // Q is no longer needed after Phase 1. On the last K chunk, pop early so the
    // reader can start fetching the next Q chunk during Phase 2.
    // In ring_mode, is_last_iter is always false — skip entirely.
    if constexpr (!ring_mode) {
        if (is_last_iter) {
            cb_pop_front(cb_q_in, Sq_chunk_t * DHt);
        }
    }

    // Lightweight ring mask tiles are permanently fronted — no pop needed.

    // ========== PHASE 2: Drain last row + QKT@V + SALAD ==========
    // After Phase 1: all rows are pushed (via hold_wr_ptr) in cb_qkt_im.
    // Rows 0..N-2 are softmax'd in-place; row N-1 has raw matmul output.
    {
        const uint32_t qktv_subblock_h = subblock_h;
        constexpr uint32_t qktv_subblock_w = (vDHt < 4) ? vDHt : 4;
        constexpr uint32_t qktv_in0_block_w = Sk_chunk_t;
        // V matmul narrowed to sub_exp'd width. Tile-level when partial subblock is used,
        // subblock-aligned otherwise (when partial was rounded up to avoid code size).
        const uint32_t v_matmul_dim =
            kt_num_full_subblocks * qkt_subblock_w + (has_partial_subblock ? kt_remainder : 0);
        const uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_subblock_h;
        constexpr uint32_t qktv_v_num_subblocks = vDHt / qktv_subblock_w;
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * vDHt;
        // cb_qkt_im row width is KT_stride (for pointer alignment), not Sk_chunk_t
        const uint32_t qktv_in0_row_tiles = qktv_subblock_h * KT_stride;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_row_tiles;

        // V wait deferred: don't block here. The sub_exp drain loop below
        // doesn't touch V, so the reader's V DMA can overlap with the drain.
        cb_reserve_back(cur.out, qktv_output_num_tiles);

        // q_subblock 0: drain last row's sub_exp in-place + first QKT@V matmul
#ifdef ARCH_BLACKHOLE
        PACK((llk_pack_mop_config<false, false, false>(cb_qkt_im, 1)));
#endif
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)@V");
            // subblock_h=1 with no partial subblock: split-drain (sub_exp interleaved with V matmul).
            // subblock_h>1 or partial subblock: non-split drain (all sub_exp first, then full V matmul).
            // Note: constexpr on subblock_h eliminates the dead branch; has_partial_subblock is runtime.
            if (subblock_h == 1 && !has_partial_subblock) {
                const uint32_t matmul_inner = qktv_in0_block_w / kt_num_full_subblocks;
                for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {
                    sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, true>(
                        cb_qkt_im,
                        cur.max,
                        cur.sum,
                        KT_stride,
                        q_num_subblocks - 1,
                        kt_sub * qkt_subblock_w,
                        subblock_h,
                        qkt_subblock_w);

                    if (kt_sub == 0) {
                        cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);
                        cb_wait_front(cb_v_in, Sk_chunk_t * vDHt);
                    }
                    if (kt_sub > 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }

                    {
                        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack");
                        uint32_t v_index_offset = 0;
                        if constexpr (!uniform_unpack_format) {
                            reconfig_data_format(cur.out, cb_v_in, cur.out, cb_qkt_im);
                        }
#ifdef ARCH_BLACKHOLE
                        mm_no_mop_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, matmul_inner);
#else
                        mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, matmul_inner);
#endif
                        for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                            blocked_matmul_and_pack<false, vDHt, vDHt>(
                                cb_qkt_im,
                                cb_v_in,
                                cur.out,
                                qktv_in0_index_offset + kt_sub * matmul_inner,
                                kt_sub * matmul_inner * vDHt + v_index_offset,
                                0,
                                v_subblock * qktv_subblock_w,
                                qktv_subblock_w,
                                qktv_subblock_h,
                                matmul_inner,
                                KT_stride);
                            v_index_offset += qktv_subblock_w;
                        }
                        if constexpr (!uniform_unpack_format) {
                            reconfig_data_format(cb_v_in, cur.out, cb_qkt_im, cur.out);
                        }
                    }

                    if (kt_sub > 0) {
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    }
                }
            } else {
                // Non-split drain: drain all sub_exp in-place, then full matmul.
                for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {
                    sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, true>(
                        cb_qkt_im,
                        cur.max,
                        cur.sum,
                        KT_stride,
                        q_num_subblocks - 1,
                        kt_sub * qkt_subblock_w,
                        subblock_h,
                        qkt_subblock_w);
                }
                if (has_partial_subblock) {
                    sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, true>(
                        cb_qkt_im,
                        cur.max,
                        cur.sum,
                        KT_stride,
                        q_num_subblocks - 1,
                        kt_num_full_subblocks * qkt_subblock_w,
                        subblock_h,
                        kt_remainder);
                }

                cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);
                cb_wait_front(cb_v_in, Sk_chunk_t * vDHt);
                {
                    MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack");
                    uint32_t v_index_offset = 0;
                    if constexpr (!uniform_unpack_format) {
                        reconfig_data_format(cur.out, cb_v_in, cur.out, cb_qkt_im);
                    }
#ifdef ARCH_BLACKHOLE
                    mm_no_mop_reinit_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, v_matmul_dim);
#else
                    mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, v_matmul_dim);
#endif
                    for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                        blocked_matmul_and_pack<false, vDHt, vDHt>(
                            cb_qkt_im,
                            cb_v_in,
                            cur.out,
                            qktv_in0_index_offset,
                            v_index_offset,
                            0,
                            v_subblock * qktv_subblock_w,
                            qktv_subblock_w,
                            qktv_subblock_h,
                            v_matmul_dim,
                            KT_stride);
                        v_index_offset += qktv_subblock_w;
                    }
                    if constexpr (!uniform_unpack_format) {
                        reconfig_data_format(cb_v_in, cur.out, cb_qkt_im, cur.out);
                    }
                }
            }
            qktv_in0_index_offset += qktv_subblock_h * KT_stride;
            qktv_in0_wait_tiles += qktv_in0_row_tiles;
        }

        // Per-row normalization lambda — fires on last K chunk (standard or deferred norm)
        [[maybe_unused]] auto normalize_row = [&](uint32_t w_salad, uint32_t& pushed) {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "ROW_NORM");
            cb_push_back(cur.sum, subblock_h);
            cb_push_back(cur.out, subblock_h * vDHt);
            normalize_row_streaming<PROFILING_ENABLED, vDHt>(
                cur.sum, cur.out, cb_col_identity, cb_recip_scratch, cb_normalized_out, subblock_h);
            pushed++;
        };

        // SALAD correction + optional normalization lambda
        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, bool last_iter, uint32_t& pushed) {
            PACK((llk_pack_reconfig_l1_acc(1)));
            {
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "S_SUM_CORR");
                mul_bcast_cols_l1_acc(prev.sum, cb_exp_max_diff, cur.sum, salad_row, w_salad, subblock_h);
            }
            {
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "S_OUT_CORR");
                mul_block_bcast_cols_acc<vDHt>(prev.out, cb_exp_max_diff, cur.out, salad_row, w_salad, subblock_h);
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            if (last_iter) {
                normalize_row(w_salad, pushed);
            }
        };

        // q_subblock 1..N-1: SALAD(prev) overlapped with matmul(cur)
        exp_packthread_tile_init<EXP_APPROX_MODE, false>();
        for (uint32_t q_subblock = 1; q_subblock < qktv_q_num_subblocks; ++q_subblock) {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)@V");
            uint32_t salad_row = q_subblock - 1;
            uint32_t w_salad = salad_row - pushed_rows;
            uint32_t w_q = q_subblock - pushed_rows;

            if (!is_first_iter) {
                cb_reserve_back(cb_exp_max_diff, subblock_h);
                sub_exp_first_col_blocks<PROFILING_ENABLED, scale_fp32>(
                    prev.max, cur.max, cb_exp_max_diff, salad_row, subblock_h);
                cb_push_back(cb_exp_max_diff, subblock_h);
            }

            cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);
            // Full matmul for current row
            {
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack");
                uint32_t v_index_offset = 0;
                if constexpr (!uniform_unpack_format) {
                    reconfig_data_format(cur.out, cb_v_in, cur.out, cb_qkt_im);
                }
#ifdef ARCH_BLACKHOLE
                mm_no_mop_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, v_matmul_dim);
#else
                mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, v_matmul_dim);
#endif
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    blocked_matmul_and_pack<false, vDHt, vDHt>(
                        cb_qkt_im,
                        cb_v_in,
                        cur.out,
                        qktv_in0_index_offset,
                        v_index_offset,
                        w_q,
                        v_subblock * qktv_subblock_w,
                        qktv_subblock_w,
                        qktv_subblock_h,
                        v_matmul_dim,
                        KT_stride);
                    v_index_offset += qktv_subblock_w;
                }
                if constexpr (!uniform_unpack_format) {
                    reconfig_data_format(cb_v_in, cur.out, cb_qkt_im, cur.out);
                }
            }

            // SALAD corrections + optional per-row normalization
            if (!is_first_iter) {
                salad_correct_row(salad_row, w_salad, is_last_iter, pushed_rows);
            } else if (is_last_iter) {
                normalize_row(w_salad, pushed_rows);
            }

            qktv_in0_index_offset += qktv_subblock_h * KT_stride;
            qktv_in0_wait_tiles += qktv_in0_row_tiles;
        }

        // Pipeline drain: SALAD for last row
        {
            const uint32_t salad_row = qktv_q_num_subblocks - 1;
            uint32_t w_salad = salad_row - pushed_rows;

            if (!is_first_iter) {
                cb_reserve_back(cb_exp_max_diff, subblock_h);
                sub_exp_first_col_blocks<PROFILING_ENABLED, scale_fp32>(
                    prev.max, cur.max, cb_exp_max_diff, salad_row, subblock_h);
                cb_push_back(cb_exp_max_diff, subblock_h);

                salad_correct_row(salad_row, w_salad, is_last_iter, pushed_rows);
            } else {
                if (is_last_iter) {
                    MaybeDeviceZoneScopedN(PROFILING_ENABLED, "ROW_NORM");
                    cb_push_back(cur.sum, subblock_h);
                    cb_push_back(cur.out, subblock_h * vDHt);
                    normalize_row_streaming<PROFILING_ENABLED, vDHt>(
                        cur.sum, cur.out, cb_col_identity, cb_recip_scratch, cb_normalized_out, subblock_h);
                    pushed_rows++;
                }
            }
        }

        // Bulk push — skip on last iteration when rows are consumed by normalization.
        if (!is_last_iter) {
            cb_push_back(cur.sum, Sq_chunk_t);
            cb_push_back(cur.out, qktv_output_num_tiles);
        }

        cb_pop_front(cb_v_in, KT_stride * vDHt);
        cb_pop_front(cb_qkt_im, Sq_chunk_t * KT_stride);
    }
}

/**
 * Streaming SDPA (v2): single-device, non-ring variant.
 * Q-chunk / K-chunk outer loop with ping-pong buffer management.
 * No row buffers — uses cb_push_back_hold_wr_ptr for direct cb_qkt_im writes.
 *
 * @tparam Sq_chunk_t   Q chunk size in tiles (rows per attention block)
 * @tparam Sk_chunk_t   K chunk size in tiles (columns per attention block)
 * @tparam Skt          Total K sequence length in tiles (used for last-chunk padding detection)
 * @tparam DHt          Head dimension in tiles
 * @tparam vDHt         V head dimension in tiles (== DHt unless V has different width)
 * @tparam scale_fp32   Attention scale factor as raw uint32_t bits (reinterpreted as float)
 * @tparam subblock_h   Matmul subblock height (rows processed per DST acquire/release cycle)
 *
 * @param q_chunks_per_core  Number of Q chunks this core processes
 * @param k_num_chunks       Total number of K chunks in the sequence
 * @param cb_out_im_A/B      Ping-pong output accumulator CBs (hold un-normalized QK@V)
 * @param cb_max_A/B          Ping-pong row-max CBs (for numerical stability)
 * @param cb_sum_A/B          Ping-pong row-sum CBs (softmax denominator)
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t subblock_h,
    uint32_t qkt_subblock_w,
    bool use_padded_mask,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    uint32_t cb_mask_in,
    bool uniform_dataformat = false>
void sdpa_standard_v2(
    const uint32_t q_chunks_per_core,
    const uint32_t k_num_chunks,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B) {
    // Neginf tile is permanently fronted by the writer — wait once before any K-chunk loop.
    constexpr uint32_t padded_k_tiles_v2 = (Sk_chunk_t - (Skt % Sk_chunk_t)) % Sk_chunk_t;
    if constexpr (use_padded_mask && padded_k_tiles_v2 > 0) {
        cb_wait_front(cb_mask_in, 1);
    }

    constexpr uint32_t padded_k_tiles_inner = (Sk_chunk_t - (Skt % Sk_chunk_t)) % Sk_chunk_t;
    constexpr uint32_t last_chunk_Sk = Sk_chunk_t - padded_k_tiles_inner;

    for (uint32_t q = 0; q < q_chunks_per_core; q++) {
        AccumulatorHalf prev = {cb_sum_A, cb_max_A, cb_out_im_A};
        AccumulatorHalf cur = {cb_sum_B, cb_max_B, cb_out_im_B};

        // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
        // All conditions are compile-time except the active_Sk == Sk_chunk_t guard (padded last chunk).
        constexpr bool can_reduce_trigger =
            (Sk_chunk_t % qkt_subblock_w == 0) && (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

        // Subblock decomposition for the non-padded (common) case — compile-time constants.
        constexpr SubblockDecomp full_decomp = {
            Sk_chunk_t / qkt_subblock_w, Sk_chunk_t % qkt_subblock_w, (Sk_chunk_t % qkt_subblock_w) > 0};

        // Subblock decomposition for the padded last chunk (computed once, used at most once).
        constexpr SubblockDecomp padded_decomp = {
            last_chunk_Sk / qkt_subblock_w, last_chunk_Sk % qkt_subblock_w, (last_chunk_Sk % qkt_subblock_w) > 0};

        auto call_step = [&](auto profiling_tag,
                             bool is_last,
                             bool is_first,
                             uint32_t active_Sk,
                             bool reduce_trigger,
                             const SubblockDecomp& decomp) {
            sdpa_inner_loop_step<
                decltype(profiling_tag)::value,
                Sq_chunk_t,
                Sk_chunk_t,
                Skt,
                DHt,
                vDHt,
                scale_fp32,
                subblock_h,
                qkt_subblock_w,
                use_padded_mask,
                false,  // ring_mode
                false,  // use_ring_mask

                uniform_dataformat,
                cb_q_in,
                cb_kt_in,
                cb_v_in,
                cb_qkt_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                cb_mask_in>(
                prev,
                cur,
                is_last,
                is_first,
                false,  // apply_mask
                0,      // lw_partial_tile_idx
                active_Sk,
                reduce_trigger,
                decomp);
        };

        for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; k_chunk++) {
            bool is_first = (k_chunk == 0);
            bool is_last = (k_chunk == k_num_chunks - 1);

            bool is_padded = is_last && padded_k_tiles_inner > 0;
            uint32_t chunk_active_Sk = is_padded ? last_chunk_Sk : Sk_chunk_t;
            bool chunk_reduce_trigger = can_reduce_trigger && !is_padded;
            call_step(
                std::false_type{},
                is_last,
                is_first,
                chunk_active_Sk,
                chunk_reduce_trigger,
                is_padded ? padded_decomp : full_decomp);

            // Post-iteration cleanup
            if (!is_first) {
                cb_pop_front(cb_exp_max_diff, Sq_chunk_t);
                cb_pop_front(prev.max, Sq_chunk_t);
                cb_pop_front(prev.sum, Sq_chunk_t);
                cb_pop_front(prev.out, Sq_chunk_t * vDHt);
            }

            if (is_last) {
                cb_pop_front(cur.max, Sq_chunk_t);
            } else {
                std::swap(prev, cur);
            }
        }
        // Q already popped inside sdpa_inner_loop_step after Phase 1 of the last K chunk.
    }
}

/**
 * Streaming Ring SDPA (v2): Ring-aware variant of sdpa_standard_v2 with deferred normalization.
 * Accumulates raw (un-normalized) softmax state across ring iterations; normalizes once on the
 * last K chunk of the last ring iteration. Single Q-chunk: accumulators persist in L1 across
 * ring iterations (no DRAM traffic). Multi Q-chunk: accumulators round-trip through DRAM.
 *
 * @tparam Sq_chunk_t       Q chunk size in tiles
 * @tparam Sk_chunk_t       K chunk size in tiles
 * @tparam Skt              Not used for ring (pass 0)
 * @tparam DHt              Head dimension in tiles
 * @tparam vDHt             V head dimension in tiles (== DHt for ring)
 * @tparam scale_fp32       Attention scale factor as raw uint32_t bits
 * @tparam subblock_h       Matmul subblock height
 * @tparam cb_max_in      CB for restoring row-max from DRAM (multi Q-chunk, c_6)
 * @tparam cb_max_out     CB for saving row-max to DRAM (multi Q-chunk, c_17)
 * @tparam cb_normalized_out CB for normalized output rows (written by normalize_row_streaming)
 * @tparam cb_sum_out       CB for saving row-sum to DRAM (multi Q-chunk, c_10)
 * @tparam cb_sum_in        CB for restoring row-sum from DRAM (multi Q-chunk, c_11)
 *
 * @param global_q_start     First global Q chunk index for this core
 * @param global_q_end       One-past-last global Q chunk index for this core
 * @param num_kv_chunks      Total K chunks this ring iter (local + joint if applicable)
 * @param ring_iter          Current ring iteration (0..ring_size-1)
 * @param ring_id            Device ID within the ring that owns this iter's KV shard
 * @param num_local_k_chunks Number of K chunks from the local (non-joint) sequence
 * @param local_padded_Nt    Per-device padded sequence length in tiles (N_local / TILE_H)
 * @param logical_nt         Actual (unpadded) global sequence length in tiles
 * @param acc_state          Persistent accumulator state (prev/cur CB halves for ping-pong)
 * @param is_last_ring_iter  True on the final ring iteration — triggers normalization
 * @param q_per_core         Number of Q chunks per core (1 = L1-only, >1 = DRAM round-trip)
 * @param lw_mask            Lightweight mask context for partial-tile padding
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t subblock_h,
    uint32_t qkt_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_mask_in,
    uint32_t cb_scale_in,
    uint32_t cb_max_in,
    uint32_t cb_max_out,
    uint32_t cb_prev_out,
    uint32_t cb_out,
    bool uniform_dataformat = false,
    uint32_t cb_normalized_out = 0,
    uint32_t cb_sum_out = 0,
    uint32_t cb_sum_in = 0>
void sdpa_ring_v2(
    const uint32_t global_q_start,
    const uint32_t global_q_end,
    const uint32_t num_kv_chunks,
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
    RingAccumulatorState& acc_state,
    const bool is_last_ring_iter = false,
    const uint32_t q_per_core = 1,
    const LightweightMaskContext& lw_mask = {}) {
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr bool uniform_format = uniform_dataformat;

    // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
    // All conditions are compile-time except the active_Sk == Sk_chunk_t guard (padded chunks).
    constexpr bool can_reduce_trigger =
        (Sk_chunk_t % qkt_subblock_w == 0) && (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

    // Subblock decomposition for the non-padded (common) case — compile-time constants.
    constexpr SubblockDecomp full_decomp = {
        Sk_chunk_t / qkt_subblock_w, Sk_chunk_t % qkt_subblock_w, (Sk_chunk_t % qkt_subblock_w) > 0};

    uint32_t KV_chunks_processed_in_iter = 0;

    // Pre-scan to count valid K chunks (needed to find last K chunk for normalization)
    uint32_t total_valid_kv = 0;
    for (uint32_t k = 0; k < num_kv_chunks; ++k) {
        const bool is_joint = k >= num_local_k_chunks;
        const uint32_t kv_start = local_padded_Nt * ring_id + k * Sk_chunk_t;
        if (!is_joint && (kv_start >= logical_nt)) {
            continue;
        }
        total_valid_kv++;
    }

    for (uint32_t q = global_q_start; q < global_q_end; q++) {
        // Use persistent accumulator state from caller (single Q-chunk)
        // or restore from DRAM (multi Q-chunk).
        AccumulatorHalf q_prev = acc_state.prev, q_cur = acc_state.cur;

        // First ring iteration starts fresh; subsequent ones have prior accumulated state.
        const bool is_first_kv_for_this_q = (ring_iter == 0);

        // Multi Q-chunk: restore accumulators from DRAM on non-first ring iterations.
        if (q_per_core > 1 && ring_iter > 0) {
            copy_block(cb_prev_out, q_prev.out, out_chunk_tiles);
            copy_block(cb_max_in, q_prev.max, Sq_chunk_t);
            copy_block(cb_sum_in, q_prev.sum, Sq_chunk_t);
        }

        uint32_t KV_chunks_processed = 0;

        for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
            // Skip KV chunks beyond logical sequence length (non-joint only)
            const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
            const uint32_t kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
            if (!kv_chunk_is_joint && (kv_global_start_tile >= logical_nt)) {
                continue;
            }

            KV_chunks_processed++;
            KV_chunks_processed_in_iter++;

            const bool is_first = is_first_kv_for_this_q && (KV_chunks_processed == 1);

            // Last K chunk of last ring_iter triggers per-row normalization
            const bool is_last = is_last_ring_iter && (KV_chunks_processed == total_valid_kv);

            // Determine if this K chunk needs masking (partial tile within a tile boundary)
            const bool is_global_n_mask_chunk = ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id;
            const bool is_local_n_mask_chunk = local_n_needs_masking && k_chunk == local_n_mask_chunk_id;
            const bool is_joint_n_mask_chunk = ring_iter_needs_joint_n_mask && kv_chunk_is_joint &&
                                               (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id;

            bool apply_mask = is_global_n_mask_chunk || is_joint_n_mask_chunk;

            // Resolve lightweight mask params for partial tile masking
            uint32_t lw_partial_tile_idx = 0;
            if (apply_mask && lw_mask.enabled) {
                if (is_global_n_mask_chunk) {
                    lw_partial_tile_idx = lw_mask.global_n_partial_tile_idx;
                } else if (is_joint_n_mask_chunk) {
                    lw_partial_tile_idx = lw_mask.joint_l_partial_tile_idx;
                }
            }

            // Tile-level matmul skip for global_n, local_n, or joint_n padding.
            // Runtime reduce narrows to active_Sk; matmul/sub_exp/V also need narrowing.
            uint32_t active_Sk_param = Sk_chunk_t;
            if (is_global_n_mask_chunk) {
                active_Sk_param = Sk_chunk_t - lw_mask.global_n_padded_tiles;
            } else if (is_local_n_mask_chunk) {
                active_Sk_param = Sk_chunk_t - lw_mask.local_n_padded_tiles;
            } else if (is_joint_n_mask_chunk) {
                active_Sk_param = Sk_chunk_t - lw_mask.joint_n_padded_tiles;
            }

            // Subblock decomposition: use compile-time constants for non-padded chunks,
            // runtime computation only for padded chunks.
            bool is_padded_chunk = (active_Sk_param != Sk_chunk_t);
            SubblockDecomp chunk_decomp =
                is_padded_chunk
                    ? SubblockDecomp{active_Sk_param / qkt_subblock_w, active_Sk_param % qkt_subblock_w, (active_Sk_param % qkt_subblock_w) > 0}
                    : full_decomp;

            sdpa_inner_loop_step<
                false,  // PROFILING_ENABLED
                Sq_chunk_t,
                Sk_chunk_t,
                Skt,
                DHt,
                vDHt,
                scale_fp32,
                subblock_h,
                qkt_subblock_w,
                false,  // use_padded_mask — ring uses ring mask instead
                true,   // ring_mode
                true,   // use_ring_mask

                uniform_dataformat,
                cb_q_in,
                cb_kt_in,
                cb_v_in,
                cb_qkt_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                cb_mask_in>(
                q_prev,
                q_cur,
                is_last,
                is_first,
                apply_mask,
                lw_partial_tile_idx,
                active_Sk_param,
                can_reduce_trigger && (active_Sk_param == Sk_chunk_t),
                chunk_decomp);

            // Post-iteration cleanup: pop previous values and swap aliases
            if (!is_first) {
                cb_pop_front(cb_exp_max_diff, Sq_chunk_t);
                cb_pop_front(q_prev.max, Sq_chunk_t);
                cb_pop_front(q_prev.sum, Sq_chunk_t);
                cb_pop_front(q_prev.out, Sq_chunk_t * vDHt);
            }

            if (is_last) {
                // Normalization consumed cur.sum and cur.out; pop cur.max.
                cb_pop_front(q_cur.max, Sq_chunk_t);
            } else {
                std::swap(q_prev, q_cur);
            }
        }

        // Pop Q — not popped inside step since ring_mode gates the early Q pop
        cb_pop_front(cb_q_in, Sq_chunk_t * DHt);

        // Persist or save accumulators for next ring iteration
        if (q_per_core == 1) {
            // Single Q-chunk: persist in L1 (no DRAM round-trip)
            acc_state.prev = q_prev;
            acc_state.cur = q_cur;
        } else if (!is_last_ring_iter) {
            // Multi Q-chunk: save raw accumulators to DRAM via writer CBs.
            // q_prev holds final accumulators after the last swap.
            if constexpr (!uniform_format) {
                pack_reconfig_data_format(cb_out);
            }
            copy_block(q_prev.out, cb_out, out_chunk_tiles);
            copy_block(q_prev.max, cb_max_out, Sq_chunk_t);
            copy_block(q_prev.sum, cb_sum_out, Sq_chunk_t);
        }
        // On last ring_iter: normalized output already in cb_out from normalize_row_streaming
    }

    // Dummy KV pop for double-buffer alignment (same as sdpa_inner_loop for RING)
    if (KV_chunks_processed_in_iter % 2 == 0) {
        cb_wait_front(cb_kt_in, DHt * Sk_chunk_t);
        cb_pop_front(cb_kt_in, DHt * Sk_chunk_t);
        cb_wait_front(cb_v_in, Sk_chunk_t * vDHt);
        cb_pop_front(cb_v_in, Sk_chunk_t * vDHt);
    }
}
