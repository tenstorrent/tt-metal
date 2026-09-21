// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streaming SDPA compute helpers.
// Included by sdpa_recipe.cpp for the explicit Blackhole numerical recipes.
// Depends on primitives from compute_common.hpp (must be included first).

#pragma once
#ifdef SDPA_RECIPE_FP32
#ifndef SDPA_RECIPE_ACCURATE
#include "balanced_exp.hpp"
#endif

#endif

#include <type_traits>
#ifndef SDPA_RECIPE_FP32
#include "compensated_macros.hpp"
#include "compensated_reuse.hpp"
#include "compensated_identity.hpp"
#include "compensated_group_sfpu.hpp"
#include "compensated_group_replay.hpp"
#endif

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"

#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifdef SDPA_RECIPE_FP32
constexpr uint32_t sdpa_sum_stride = 1;
#else
// Interleaved high/low denominator tiles per query-tile row.
constexpr uint32_t sdpa_sum_stride = 2;
#endif

#ifdef SDPA_RECIPE_FP32
constexpr uint32_t sdpa_out_stride = 1;
#else
// Each query-tile row contains a contiguous high plane, then its low plane.
constexpr uint32_t sdpa_out_stride = 2;
#endif

// reduce_trigger uses a packer->unpacker semaphore handshake to start the reduce early and skip the
// input CB wait. Quasar has no such handshake, so it stays disabled there and the normal CB
// synchronization is kept (see can_reduce_trigger below).
constexpr bool reduce_trigger_supported = true;

// Template-driven profiling: MaybeDeviceZoneScopedN(ENABLED, name)
// When ENABLED=true: RAII profileScope writes timestamps (same as DeviceZoneScopedN)
// When ENABLED=false: empty struct, zero overhead (compiler eliminates entirely)
#if defined(PROFILE_STREAMING)
#define MaybeDeviceZoneScopedN(ENABLED, name)
#elif defined(PROFILE_KERNEL)
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

// --- Outlined out-of-order pack (code-size) ---
#ifdef SDPA_RECIPE_FP32
// A full tile's four contiguous FP32 faces extend naturally to eight faces
// for two adjacent score tiles. One unpack-to-DST handshake owns both tiles.
// Only valid for the fixed two-score, full-face, unpadded recipe path.
ALWI void sdpa_score_unpack_mop(uint32_t faces) {
    UNPACK((
        ckernel_template(faces, 1, TT_OP_UNPACR(0, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1))
            .program()));
}
#endif

#define SDPA_UTIL_MM_SCOPE()

#include "circular_buffer.hpp"

struct AccumulatorHalf {
    uint32_t sum, max, out;
};

// Sentinel for "no CB" — beyond the valid 0-31 range.
constexpr uint32_t INVALID_CB = 32;
#ifdef SDPA_RECIPE_FP32
static bool sdpa_skip_prev_sum_pop = false;
#endif
// BH benefits from blocked pack at width 4; WH keeps the threshold at 8 because
// width-4 blocked-pack reconfiguration costs more than it saves there.
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 4;
ALWI bool should_use_blocked_pack_width(uint32_t pack_width) { return pack_width >= MIN_BLOCKED_PACK_TILES; }

template <typename... Args>
ALWI void sdpa_stream_reconfig(Args... args) {
#ifdef SDPA_RECIPE_FP32
    reconfig_data_format_skip_int8(args...);
#else
    reconfig_data_format(args...);
#endif
}

template <typename... Args>
ALWI void sdpa_stream_reconfig_srca(Args... args) {
#ifdef SDPA_RECIPE_FP32
    reconfig_data_format_srca_skip_int8(args...);
#else
    reconfig_data_format_srca(args...);
#endif
}

template <typename... Args>
ALWI void sdpa_stream_reconfig_srcb(Args... args) {
#ifdef SDPA_RECIPE_FP32
    reconfig_data_format_srcb_skip_int8(args...);
#else
    reconfig_data_format_srcb(args...);
#endif
}

#ifdef SDPA_RECIPE_FP32
#ifdef TRISC_PACK
static uint32_t sdpa_pack_format_cb = INVALID_CB;
static uint32_t sdpa_pack_width = 0;
ALWI void sdpa_cached_pack_format(uint32_t cb) {
    if (sdpa_pack_format_cb == INVALID_CB || pack_src_format[sdpa_pack_format_cb] != pack_src_format[cb] ||
        pack_dst_format[sdpa_pack_format_cb] != pack_dst_format[cb]) {
        pack_reconfig_data_format(cb);
    }
    sdpa_pack_format_cb = cb;
}
#endif
#endif

template <uint32_t old_cb, uint32_t new_cb>
ALWI void sdpa_maybe_pack_reconfig_data_format() {
#ifdef SDPA_RECIPE_FP32
#ifdef TRISC_PACK
    sdpa_cached_pack_format(new_cb);
#endif
#else
#ifdef TRISC_PACK
    if constexpr (pack_dst_format[old_cb] != pack_dst_format[new_cb]) {
        pack_reconfig_data_format(old_cb, new_cb);
    }
#endif
#endif
}

template <uint32_t old_cb, uint32_t new_cb>
constexpr bool sdpa_unpack_format_changed() {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    return unpack_src_format[old_cb] != unpack_src_format[new_cb] ||
           unpack_dst_format[old_cb] != unpack_dst_format[new_cb] ||
           unpack_tile_face_r_dim[old_cb] != unpack_tile_face_r_dim[new_cb] ||
           unpack_tile_num_faces[old_cb] != unpack_tile_num_faces[new_cb];
#else
    return false;
#endif
}

template <uint32_t srca_old_cb, uint32_t srca_new_cb, uint32_t srcb_old_cb, uint32_t srcb_new_cb>
ALWI void sdpa_maybe_reconfig_data_format() {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    if constexpr (
        sdpa_unpack_format_changed<srca_old_cb, srca_new_cb>() ||
        sdpa_unpack_format_changed<srcb_old_cb, srcb_new_cb>()) {
        sdpa_stream_reconfig(srca_old_cb, srca_new_cb, srcb_old_cb, srcb_new_cb);
    }
#endif
}

template <uint32_t srca_old_cb, uint32_t srca_new_cb, uint32_t srcb_old_cb, uint32_t srcb_new_cb>
ALWI void sdpa_maybe_reconfig_data_format(uint32_t runtime_srca_old_cb, uint32_t runtime_srcb_old_cb) {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    if constexpr (
        sdpa_unpack_format_changed<srca_old_cb, srca_new_cb>() ||
        sdpa_unpack_format_changed<srcb_old_cb, srcb_new_cb>()) {
        sdpa_stream_reconfig(runtime_srca_old_cb, srca_new_cb, runtime_srcb_old_cb, srcb_new_cb);
    }
#endif
}

// Keep this out-of-line even on BH: repeated pack-width configuration sites
// inflate SDPA streaming code size more than this call costs in measured cases.
static __attribute__((noinline, noclone)) void configure_pack_width(uint32_t cb, uint32_t pack_width) {
#ifdef SDPA_RECIPE_FP32
    PACK((sdpa_cached_pack_format(cb)));
    PACK(if (sdpa_pack_width == pack_width) { return; })
    PACK(sdpa_pack_width = pack_width;)
#endif
    // Pure MOP refresh: addrmod and packer strides are already configured from
    // the initial pack init, and changing pack_width only requires re-issuing
    // the MOP. Skipping the packer-strides reconfig saves a THCON stall per
    // call on the SDPA streaming hot path.
    PACK((llk_pack_init<
          ckernel::PackMode::Default,
          false /* zero_output */,
          true /* skip_addrmod_config */,
          true /* skip_packer_strides */>(cb, pack_width)));
}

ALWI void configure_single_tile_pack(uint32_t cb) { configure_pack_width(cb, 1); }

ALWI bool configure_row_pack_width(uint32_t cb, uint32_t pack_width) {
    const bool use_blocked_pack_width = should_use_blocked_pack_width(pack_width);
    configure_pack_width(cb, use_blocked_pack_width ? pack_width : 1);
    return use_blocked_pack_width;
}

ALWI void init_sdpa_streaming_semaphores() {
    // reduce_trigger runs the QK row-max reduce as a split MOP gated by a PACK->UNPACK handshake on
    // two T6 tokens (firmware inits neither). FPU_SFPU, posted after pack + mask + push, gates run()#2
    // (and run()#1 on the non-overlap path). UNPACK_MATH_DONE (the first-half token) is borrowed (unused elsewhere
    // in SDPA): PACK posts it early, once the first half is committed, to gate run()#1 on the overlap
    // path so it overlaps the second-half pack.
    PACK((t6_semaphore_init(semaphore::FPU_SFPU, 0, 1)));
    PACK((t6_semaphore_init(semaphore::UNPACK_MATH_DONE, 0, 1)));
}

// Raw pack: caller must have already called configure_row_pack_width(out_cb, pack_width).
// Use this in tight loops after configuring once at the loop boundary.
ALWI void pack_contiguous_rows_nocfg(
    uint32_t out_cb,
    uint32_t row_base,
    uint32_t row_count,
    uint32_t row_stride,
    uint32_t col_base,
    uint32_t pack_width) {
#ifndef SDPA_RECIPE_FP32
    if (out_cb == 8 || out_cb == 9) {
        row_stride *= 2;
    }
#endif
    uint32_t dst_index = 0;
    const bool use_blocked_pack_width = should_use_blocked_pack_width(pack_width);
    for (uint32_t row = 0; row < row_count; ++row) {
        uint32_t out_tile_index = (row_base + row) * row_stride + col_base;
        if (use_blocked_pack_width) {
            sdpa_pack_tile_ooo(dst_index, out_cb, out_tile_index);
            dst_index += pack_width;
        } else {
            for (uint32_t col = 0; col < pack_width; ++col) {
                sdpa_pack_tile_ooo(dst_index++, out_cb, out_tile_index + col);
            }
        }
    }
}

// Safe pack: configures MOP then packs. Use for one-off calls or first-in-group.
ALWI void pack_contiguous_rows(
    uint32_t out_cb,
    uint32_t row_base,
    uint32_t row_count,
    uint32_t row_stride,
    uint32_t col_base,
    uint32_t pack_width) {
    configure_row_pack_width(out_cb, pack_width);
    pack_contiguous_rows_nocfg(out_cb, row_base, row_count, row_stride, col_base, pack_width);
}

/**
 * Blocked subblock matmul with absolute offset packing.
 * Always uses pack_tile<true> at row-major positions in out_cb.
 *
 * noinline on Wormhole: keeps sdpa_inner_loop_step's frame off the TR0 stack to stay within budget
 * for the ring cases (it would otherwise overflow). WH-only to avoid the call overhead elsewhere.
 */
template <bool transpose, uint32_t in1_stride, uint32_t out_num_cols>
void blocked_matmul_and_pack(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t row_subblock_idx,
    uint32_t out_col_offset,
    uint32_t subblock_w,
    uint32_t subblock_h,
    uint32_t inner_dim,
    uint32_t matmul_stride,
#ifdef SDPA_RECIPE_FP32
    bool skip_pack_configure = false) {
#else
    bool skip_pack_configure = false) {
#endif
    SDPA_UTIL_MM_SCOPE();
    tile_regs_acquire();
    uint32_t dst_index = 0;
    uint32_t in0_index = in0_index_start;
    uint32_t in1_index = in1_index_start;
    for (uint32_t inner = 0; inner < inner_dim; ++inner) {
#ifdef SDPA_RECIPE_FP32
#ifndef SDPA_RECIPE_ACCURATE
        if constexpr (transpose) {
            UNPACK((llk_unpack_AB_matmul(in0_cb, in1_cb, in0_index, in1_index, subblock_w, subblock_h, matmul_stride)));
            MATH((llk_math_matmul_no_mop<MathFidelity::HiFi4, MM_THROTTLE>(
                in0_cb, in1_cb, dst_index, subblock_w, subblock_h)));
        } else
#endif
        {
            matmul_block_no_mop(
                in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, matmul_stride);
        }
#else
        matmul_block_no_mop(
            in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, matmul_stride);
#endif
        in0_index++;
        in1_index += in1_stride;
    }
    tile_regs_commit();

    tile_regs_wait();
    if (!skip_pack_configure) {
        configure_row_pack_width(out_cb, subblock_w);
    }
    pack_contiguous_rows_nocfg(
        out_cb, row_subblock_idx * subblock_h, subblock_h, out_num_cols, out_col_offset, subblock_w);
    tile_regs_release();
}

/**
 * Matmul + pack of scores against in-place latent V (V read from K^T: V[sk][vd] == K^T[vd][sk]).
 * Each output column vd is its own matmul chain over K^T row vd (in1 base vd*KT_stride, inner
 * stride 1), so unlike blocked_matmul_and_pack the strided columns can't be folded into one matmul.
 * Batches as many columns as DST holds per acquire/commit/pack to keep the FPU busy, instead of
 * paying the handshake + pack-configure per column (~2/3 FPU idle for the 1-wide path).
 *
 * Loops: outer walks columns in DST-sized batches; middle does one column (= one matmul chain) per
 * DST tile; inner accumulates that chain over inner_dim tiles. Each batch is packed out in one go.
 */
template <uint32_t vDHt, uint32_t dst_size, uint32_t subblock_h>
void inplace_v_matmul_pack_batched(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t inner_dim,
    uint32_t KT_stride) {
    // Each output column is written column-major into DST (column c at c*subblock_h), but the
    // pack below reads DST row-major; the two orderings only coincide when subblock_h==1, which
    // kt_inplace_v guarantees via Sq_chunk_t==1. Enforce it so this can't silently corrupt if
    // reused with multi-tile Q.
    static_assert(subblock_h == 1, "inplace_v_matmul_pack_batched requires single-tile Q (subblock_h==1)");
    // subblock_h DST tiles per output column; batch as many columns as DST holds.
    const uint32_t cols_per_batch = dst_size / subblock_h;
    for (uint32_t vs0 = 0; vs0 < vDHt; vs0 += cols_per_batch) {
        const uint32_t cols = (vDHt - vs0 < cols_per_batch) ? (vDHt - vs0) : cols_per_batch;
        tile_regs_acquire();
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t in0_index = in0_index_start;
            uint32_t in1_index = (vs0 + c) * KT_stride;
            for (uint32_t inner = 0; inner < inner_dim; ++inner) {
                matmul_block_no_mop(
                    in0_cb, in1_cb, in0_index, in1_index, c * subblock_h, false, 1, subblock_h, KT_stride);
                in0_index++;
                in1_index++;
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        configure_row_pack_width(out_cb, cols);
        pack_contiguous_rows_nocfg(out_cb, 0, subblock_h, vDHt, vs0, cols);
        tile_regs_release();
    }
}

/**
 * Per-row-group max reduction with optional eltwise_max against prev values.
 * Reads from in0_cb at row group offset, writes to out_cb sequentially.
 */
template <uint32_t in0_cb, uint32_t scale_cb, uint32_t row_stride>
void reduce_c_row_group(
    uint32_t out_cb,
    uint32_t prev_cb,
    uint32_t row_group_index,
    bool do_eltwise_max,
    uint32_t sbh,
    uint32_t reduce_cols,
    bool respect_trigger = false,
    uint32_t mirror_cb = INVALID_CB,
    bool overlap_first_half = false) {
    const uint32_t group_size = sbh;
    const uint32_t row_start = row_group_index * group_size;

    // row_stride: physical row width in the CB (may exceed cols on the reduced path).
    const uint32_t cumulative_input_tiles = (row_group_index + 1) * group_size * row_stride;
    const uint32_t cumulative_prev_tiles = (row_group_index + 1) * group_size;

    // scale_cb assumed ready (waited once at kernel init)

    tile_regs_acquire();

    if (do_eltwise_max) {
        CircularBuffer(prev_cb).wait_front(cumulative_prev_tiles);
#ifdef SDPA_RECIPE_FP32
        sdpa_stream_reconfig_srca(prev_cb);
#endif
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < group_size; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
    }

    // Deferred: wait for in0_cb just before its first use (reduce_block_max_row).
    // When do_eltwise_max=true, the prev_cb wait + copy_tile work above can overlap
    // with in0_cb data arrival.
    // When respect_trigger=true, the unpack MOP is split into two halves with a
    // HW semaphore wait in between, so we don't need a wait-front here.
    if (!respect_trigger) {
        CircularBuffer(in0_cb).wait_front(cumulative_input_tiles);
    }

#ifdef SDPA_RECIPE_FP32
    sdpa_stream_reconfig(in0_cb, scale_cb);
#endif
    reduce_block_max_row_init_runtime(out_cb, reduce_cols, in0_cb, scale_cb, respect_trigger);
    for (uint32_t i = 0; i < group_size; i++) {
        const uint32_t input_tile_start = (row_start + i) * row_stride;
        reduce_block_max_row_runtime(in0_cb, scale_cb, input_tile_start, i, respect_trigger, overlap_first_half);
    }
    reduce_block_max_row_uninit_runtime(in0_cb, respect_trigger, overlap_first_half);

    tile_regs_commit();
    tile_regs_wait();
#ifdef SDPA_RECIPE_FP32
    configure_single_tile_pack(out_cb);
#endif

    for (uint32_t i = 0; i < group_size; i++) {
        pack_tile<false>(i, out_cb);
    }

    // Dual-write: same DST data to writer's staging CB (e.g. cb_max_out).
    // DST is read non-destructively by pack, so this is safe before tile_regs_release().
    if (mirror_cb != INVALID_CB) {
        for (uint32_t i = 0; i < group_size; i++) {
            pack_tile<false>(i, mirror_cb);
        }
    }

    tile_regs_release();
}

#ifdef SDPA_RECIPE_FP32
#ifdef SDPA_RECIPE_ACCURATE
// Apply the same BF16-derived maximum once to the entire FP32 score row using
// the L1 FP32 adder, then reload four score tiles without a resident maximum.
static void sdpa_subtract_max_l1(uint32_t inout_cb, uint32_t max_cb, uint32_t row, uint32_t cols) {
    tile_regs_acquire();
    sdpa_stream_reconfig(max_cb, max_cb);
    unary_bcast_init<BroadcastType::COL>(max_cb);
    unary_bcast<BroadcastType::COL>(max_cb, row, 0);
    unary_bcast_uninit<BroadcastType::COL>(max_cb);
    tile_regs_commit();
    tile_regs_wait();
    PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_negate_max, 0, VectorMode::None)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    configure_single_tile_pack(inout_cb);
    PACK((llk_pack_relu_config(ReluConfig::none())));
    PACK((llk_pack_reconfig_l1_acc(1)));
    for (uint32_t col = 0; col < cols; ++col) {
        pack_tile<true>(0, inout_cb, row * cols + col);
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    tile_regs_release();
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}
#endif
#endif

/**
 * In-place sub_exp on cb_qkt_im: subtracts max, applies exp with ReLU clamping,
 * writes back to same positions. Accumulates row sums into reduce_cb.
 */
template <bool profiling_enabled, uint32_t scale_fp32>
void sub_exp_block_bcast_cols(
    uint32_t inout_cb,
    uint32_t max_cb,
    uint32_t reduce_cb,
    uint32_t cols_in_row,
    uint32_t q_subblock,
    uint32_t global_col_base,
    uint32_t sbh,
    uint32_t sbw,
    bool skip_pack_configure = false) {
    const uint32_t tiles_per_row = sbh;
    const uint32_t tiles_per_column = sbw;
    const uint32_t max_row_base = q_subblock * tiles_per_row;

#ifdef SDPA_RECIPE_FP32
#ifdef SDPA_RECIPE_ACCURATE
    constexpr uint32_t score_batch = 4;
    // Validated 1x4 QK subblocks and unpadded 512/1024 K chunks have even widths.
    CircularBuffer(max_cb).wait_front((q_subblock + 1) * tiles_per_row);
    if (global_col_base == 0) {
        sdpa_subtract_max_l1(inout_cb, max_cb, max_row_base, cols_in_row);
    }
    UNPACK((get_local_cb_interface(7).fifo_rd_ptr = get_local_cb_interface(inout_cb).fifo_rd_ptr));
    configure_pack_width(inout_cb, score_batch);
    PACK((llk_pack_relu_config(ReluConfig::zero())));
    for (uint32_t i = 0; i < tiles_per_row; ++i) {
        for (uint32_t j = 0; j < tiles_per_column; j += score_batch) {
            const uint32_t index = (max_row_base + i) * cols_in_row + global_col_base + j;
            tile_regs_acquire();
            sdpa_stream_reconfig(7, 7);
            unary_bcast_init<BroadcastType::NONE>(7);
            sdpa_score_unpack_mop(4 * score_batch);
            unary_bcast<BroadcastType::NONE>(7, index, 0);
            // The generic one-tile helper cleared tile 0's zero flags. Preserve
            // the Blackhole unpack-to-DST workaround for tile 1 as well.
            MATH(for (uint32_t tile = 1; tile < score_batch; ++tile) {
                for (uint32_t face = 0; face < 4; ++face) {
                    TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, get_dest_index_in_faces(tile, face));
                }
            })
            sdpa_score_unpack_mop(4);
            unary_bcast_uninit<BroadcastType::NONE>(7);
            tile_regs_commit();
            tile_regs_wait();
            PACK((ckernel::sfpu::restore_sdpa_grid_macro_instructions()));
            PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sdpa_exp_grid_batch,
                (32 * score_batch),
                0,
                VectorMode::None)));
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sdpa_exp_refine_loadmacro,
                (32 * score_batch),
                0,
                VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile<true>(0, inout_cb, index);
            tile_regs_release();
        }
    }
    sdpa_stream_reconfig(inout_cb, max_cb);
    configure_row_pack_width(inout_cb, tiles_per_column);
    PACK((llk_pack_relu_config(ReluConfig::none())));
    PACK((llk_pack_reconfig_l1_acc(0)));
    return;
#endif
#endif

    {
        MaybeDeviceZoneScopedN(profiling_enabled, "SUB_EXP_BLOCK_INIT");
#ifdef SDPA_RECIPE_FP32
        if (skip_pack_configure) {
            sdpa_stream_reconfig_srca(inout_cb);
        } else {
            sdpa_stream_reconfig(inout_cb, max_cb);
        }
#endif
        sub_bcast_cols_init_short_custom(inout_cb, max_cb, tiles_per_column);
    }

    // inout_cb assumed ready (max_cb was already computed from it)
    CircularBuffer(max_cb).wait_front((q_subblock + 1) * tiles_per_row);

    tile_regs_acquire();
    {
        MaybeDeviceZoneScopedN(profiling_enabled, "SUB");
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base;
            sub_tiles_bcast_cols_custom(
                inout_cb, max_cb, in0_tile_index, max_row_base + i, dst_index, tiles_per_column);
            dst_index += tiles_per_column;
        }
    }
    tile_regs_commit();

    tile_regs_wait();
    PACK((llk_pack_relu_config(ReluConfig::zero())));
    {
        MaybeDeviceZoneScopedN(profiling_enabled, "EXP");
        uint32_t dst_index = 0;
        constexpr int iterations = 32;
        constexpr VectorMode vector_mode_exp = VectorMode::None;
#ifdef SDPA_RECIPE_FP32
        if (tiles_per_row * tiles_per_column == 4) {
            PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_grid_batch, (128), 0, vector_mode_exp)));
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_stream_effective, (128), 0, vector_mode_exp)));
        } else
#endif
        {
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                for (uint32_t j = 0; j < tiles_per_column; j++) {
#ifdef SDPA_RECIPE_FP32
                    PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
                    PACK((SFPU_UNARY_CALL(
                        DST_SYNC_MODE,
                        DST_ACCUM_MODE,
                        calculate_sdpa_exp_grid_batch,
                        (iterations),
                        dst_index,
                        vector_mode_exp)));
                    PACK((SFPU_UNARY_CALL(
                        DST_SYNC_MODE,
                        DST_ACCUM_MODE,
                        calculate_sdpa_exp_stream_effective,
                        (iterations),
                        dst_index,
                        vector_mode_exp)));
                    ++dst_index;
#else
                    exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index++, vector_mode_exp);
#endif
                }
            }
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }

    {
        MaybeDeviceZoneScopedN(profiling_enabled, "PACK SUB_EXP");
        // Pack back to inout_cb at the same absolute positions.
        // In Phase 1, the caller pre-configures (cb_qkt_im, actual_sbw) before the kt loop
        // and blocked_matmul_and_pack restores it after each sub_exp. Skip the redundant
        // reconfigure here when the caller guarantees the state.
        if (skip_pack_configure) {
            pack_contiguous_rows_nocfg(
                inout_cb, max_row_base, tiles_per_row, cols_in_row, global_col_base, tiles_per_column);
        } else {
            pack_contiguous_rows(inout_cb, max_row_base, tiles_per_row, cols_in_row, global_col_base, tiles_per_column);
        }
#ifndef SDPA_RECIPE_FP32
        configure_single_tile_pack(reduce_cb);
        {
            uint32_t dst_index = 0;
#pragma GCC unroll 1
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                if (global_col_base > 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                } else {
                    PACK((llk_pack_reconfig_l1_acc(0)));
                }
#pragma GCC unroll 1
                for (uint32_t j = 0; j < tiles_per_column; ++j) {
                    pack_tile<true>(dst_index++, reduce_cb, (max_row_base + i) * sdpa_sum_stride);
                    if (global_col_base == 0 && j == 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
        }
#endif
    }

    tile_regs_release();

    // Restore packer ReLU config after all exp operations complete
    PACK((llk_pack_relu_config(ReluConfig::none())));
    PACK((llk_pack_reconfig_l1_acc(0)));
}

#ifndef SDPA_RECIPE_FP32
#ifdef TRISC_UNPACK
inline uint32_t sdpa_scan_identity_maxima(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t q_subblock, uint32_t tiles_per_row) {
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    uint32_t identical = 1;
    for (uint32_t t = 0; t < tiles_per_row && identical; ++t) {
        auto* old_max =
            reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(in0_cb), global_row_base + t));
        auto* new_max =
            reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(in1_cb), global_row_base + t));
        uint32_t difference = 0;
        uint32_t nonfinite = 0;
        // Same 32 row values, but fixed native-face strides reduce
        // per-row index and branch/control bookkeeping. Checking
        // old's exponent is sufficient once raw old/new bits are equal.
        for (uint32_t face = 0; face < 2; ++face) {
            _Pragma("GCC unroll 16") for (uint32_t r = 0; r < 16; ++r) {
                const uint32_t offset = face * 256 + r * 8;
                const uint32_t a = old_max[offset] & 0xffffu;
                const uint32_t b = new_max[offset] & 0xffffu;
                difference |= a ^ b;
                nonfinite |= ((a & 0x7f80u) == 0x7f80u);
            }
        }
        identical &= ((difference | nonfinite) == 0);
    }

    return identical;
}
#endif

#endif
/**
 * Column-only exp(prev_max - cur_max) for SALAD corrections.
 * Operates on first-column subset of tiles.
 */
template <bool profiling_enabled, uint32_t scale_fp32>
#ifdef SDPA_RECIPE_FP32
bool sub_exp_first_col_blocks(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t sbh) {
#else
bool sub_exp_first_col_blocks(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t q_subblock,
    uint32_t sbh,
    uint32_t precomputed_identity = 0) {
#endif
    const uint32_t tiles_per_row = sbh;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

#ifndef SDPA_RECIPE_FP32
    sub_init(in0_cb, in1_cb);
#endif

    CircularBuffer(in0_cb).wait_front((q_subblock + 1) * tiles_per_row);
    CircularBuffer(in1_cb).wait_front((q_subblock + 1) * tiles_per_row);

    uint32_t identical = 1;
    UNPACK({
#ifdef SDPA_RECIPE_FP32
        for (uint32_t t = 0; t < tiles_per_row && identical; ++t) {
            auto* old_max = reinterpret_cast<volatile uint32_t*>(
                get_tile_l1_byte_address(get_operand_id(in0_cb), global_row_base + t));
            auto* new_max = reinterpret_cast<volatile uint32_t*>(
                get_tile_l1_byte_address(get_operand_id(in1_cb), global_row_base + t));
            // Same 32 BF16 first-column comparisons, four independent loads
            // per loop control iteration. Upper halves remain irrelevant.
            for (uint32_t r = 0; r < 32; r += 4) {
                const uint32_t offset = (r / 16) * 256 + (r % 16) * 8;
                const uint32_t difference =
                    (old_max[offset] ^ new_max[offset]) | (old_max[offset + 8] ^ new_max[offset + 8]) |
                    (old_max[offset + 16] ^ new_max[offset + 16]) | (old_max[offset + 24] ^ new_max[offset + 24]);
                if ((difference & 0xffffu) != 0) {
                    identical = 0;
                    break;
                }
            }
        }
#else
        identical = precomputed_identity;
#endif
        mailbox_write(ckernel::ThreadId::MathThreadId, identical);
        mailbox_write(ckernel::ThreadId::PackThreadId, identical);
    })
    MATH(identical = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(identical = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    if (identical) {
        return true;
    }
#ifdef SDPA_RECIPE_FP32

    // An exact identity correction never consumes either subtraction input.
    // Preserve the comparison and arithmetic, but skip unused operation setup.
    sdpa_stream_reconfig(in0_cb, in1_cb);
    sub_init(in0_cb, in1_cb);
#endif

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
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_correction, (scale_fp32), dst_index, VectorMode::C)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

        configure_single_tile_pack(out_cb);
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            pack_tile<false>(i, out_cb);
        }

        tile_regs_release();
    }
    return false;
}

/**
 * Fused SALAD correction: output correction + sum correction in one init cycle.
 * Folds the sum correction tile(s) into the output correction's last DEST batch
 * when there's room, or appends a minimal extra batch otherwise.
 * Eliminates the separate init + acquire/release overhead for sum correction.
 *
 * ob_q_subblock controls read offset for out_in_cb and bcast_cb (0 when popped row-by-row).
 * sum_q_subblock controls read offset for sum_in_cb (cumulative when not popped per-row).
 */
#ifndef SDPA_RECIPE_FP32
#include "compensated_group.hpp"
#endif

#ifdef SDPA_RECIPE_FP32
template <int pairs = 1, bool first_column = false>
ALWI void sdpa_fp32_state_update(
#else
// The caller fences the current chunk's unpublished L1 writes once per row
// group, then keeps pack in overwrite/single-tile mode for all state updates.
// Copying the next DST half can overlap the current half's PACK-thread SFPU work.
template <int pairs = 1, int state_stride = 1, bool separate_corrections = false>
ALWI void sdpa_compensated_state_update(
#endif
    uint32_t old_cb,
    uint32_t new_cb,
    uint32_t correction_cb,
#ifdef SDPA_RECIPE_FP32
    uint32_t old_index,
    uint32_t new_read_index,
    uint32_t new_write_index,
    uint32_t correction_index,
#else
    uint32_t old_hi,
    uint32_t old_lo,
    uint32_t new_hi,
    uint32_t new_lo,
    uint32_t correction_row,
    uint32_t new_read_hi,
#endif
    bool identity_correction = false) {
    tile_regs_acquire();
#ifdef SDPA_RECIPE_FP32
    sdpa_stream_reconfig(old_cb, old_cb);
    unary_bcast_init<BroadcastType::NONE>(old_cb);
    if constexpr (pairs > 1) {
        // Same FP32 values, ordering and destination indices as individual
        // unary copies. Preserve Blackhole's zero-flag clear for every tile.
        sdpa_score_unpack_mop(4 * pairs);
        unary_bcast<BroadcastType::NONE>(old_cb, old_index, 0);
        MATH(for (uint32_t tile = 1; tile < pairs; ++tile) {
            for (uint32_t face = 0; face < 4; ++face) {
                TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, get_dest_index_in_faces(tile, face));
            }
        })
        sdpa_score_unpack_mop(4);
    } else {
        for (int p = 0; p < pairs; ++p) {
            unary_bcast<BroadcastType::NONE>(old_cb, old_index + p, p);
#else
    copy_init(old_cb);
    if constexpr (pairs == 2 && state_stride == 1 && !separate_corrections) {
        // Same six BF16 tiles, grouped by plane in DST for paired block copy/pack.
        copy_block(old_cb, old_hi, 0, 2);
        copy_block(old_cb, old_lo, 2, 2);
        copy_block(new_cb, new_read_hi, 4, 2);
    } else {
        for (uint32_t b = 0; b < pairs; ++b) {
            copy_tile(old_cb, old_hi + b * state_stride, 3 * b);
            copy_tile(old_cb, old_lo + b * state_stride, 3 * b + 1);
            copy_tile(new_cb, new_read_hi + b * state_stride, 3 * b + 2);
#endif
        }
    }
#ifdef SDPA_RECIPE_FP32
    unary_bcast_uninit<BroadcastType::NONE>(old_cb);
#endif
    if (!identity_correction) {
#ifdef SDPA_RECIPE_FP32
        sdpa_stream_reconfig(correction_cb, correction_cb);
#endif
        unary_bcast_init<BroadcastType::COL>(correction_cb);
#ifdef SDPA_RECIPE_FP32
        unary_bcast<BroadcastType::COL>(correction_cb, correction_index, pairs);
#else
        unary_bcast<BroadcastType::COL>(correction_cb, correction_row, 3 * pairs);
        if constexpr (separate_corrections) {
            static_assert(pairs == 2);
            unary_bcast<BroadcastType::COL>(correction_cb, correction_row + 1, 3 * pairs + 1);
        }
#endif
        unary_bcast_uninit<BroadcastType::COL>(correction_cb);
    }
    tile_regs_commit();
    tile_regs_wait();
#ifdef SDPA_RECIPE_FP32
    if (!identity_correction) {
        if constexpr (first_column) {
            PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, sdpa_state_rescale_first_column, 0, VectorMode::C)));
        } else {
            PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, sdpa_state_rescale, (pairs), 0, VectorMode::None)));
#else
    PACK((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sdpa_identity_state,
        (pairs, separate_corrections),
        0,
        VectorMode::None,
        identity_correction)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    if constexpr (pairs == 2 && state_stride == 1 && !separate_corrections) {
        // Width-two MOP: contiguous high pair, then contiguous low pair.
        pack_tile<true>(0, new_cb, new_hi);
        pack_tile<true>(2, new_cb, new_lo);
    } else {
        for (uint32_t b = 0; b < pairs; ++b) {
            pack_tile<true>(3 * b, new_cb, new_hi + b * state_stride);
            pack_tile<true>(3 * b + 1, new_cb, new_lo + b * state_stride);
#endif
        }
#ifdef SDPA_RECIPE_FP32
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }
    for (int p = 0; p < pairs; ++p) {
        pack_tile<true>(p, new_cb, new_write_index + p);
#endif
    }
    tile_regs_release();
}

template <uint32_t sbh_t, uint32_t sbw_t, uint32_t dst_size>
void salad_correct_fused(
    uint32_t out_in_cb,
    uint32_t sum_in_cb,
    uint32_t bcast_cb,
    uint32_t out_out_cb,
    uint32_t sum_out_cb,
    uint32_t ob_q_subblock,
    uint32_t sum_q_subblock,
    uint32_t write_q_subblock,
    bool current_sum_popped = false,
#ifdef SDPA_RECIPE_FP32
    bool identity_correction = false) {
#else
    bool identity_correction = false,
    bool group_boundary = false,
    bool group_odd = false,
    bool group_has_local = true) {
#endif
    constexpr uint32_t tiles_per_row = sbh_t;
    constexpr uint32_t tiles_per_column = sbw_t;
    constexpr uint32_t col_batch = (dst_size / sbh_t < sbw_t) ? dst_size / sbh_t : sbw_t;
    constexpr uint32_t last_out_cols = (sbw_t % col_batch == 0) ? col_batch : (sbw_t % col_batch);
    constexpr bool can_fuse_last = sdpa_sum_stride == 1 && (last_out_cols * sbh_t + sbh_t <= dst_size);

    // out_in_cb and bcast_cb may be popped row-by-row (ob_q_subblock=0) while
    // sum_in_cb retains cumulative indexing (sum_q_subblock=salad_row).
    const uint32_t ob_row_base = ob_q_subblock * tiles_per_row;
    const uint32_t sum_row_base = sum_q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;

#ifdef SDPA_RECIPE_FP32
    CircularBuffer(out_in_cb).wait_front((ob_q_subblock + 1) * tiles_per_row * tiles_per_column);
    CircularBuffer(sum_in_cb).wait_front((sum_q_subblock + 1) * tiles_per_row);
    CircularBuffer(bcast_cb).wait_front((ob_q_subblock + 1) * tiles_per_row);
    PACK((llk_pack_reconfig_l1_acc(1)));
    configure_single_tile_pack(out_out_cb);
    for (uint32_t i = 0; i < tiles_per_row; ++i) {
        const uint32_t read_row = current_sum_popped ? write_row_base + i : sum_row_base + i;
        if (identity_correction && tiles_per_column == 4) {
            // Identity needs no correction tile, so all four FP32 DST slots
            // hold independent numerator tiles. L1 adds remain unchanged.
            sdpa_fp32_state_update<4>(
                out_in_cb,
                out_out_cb,
                bcast_cb,
                (ob_row_base + i) * tiles_per_column,
                read_row * tiles_per_column,
                (write_row_base + i) * tiles_per_column,
                ob_row_base + i,
                true);
        } else {
            for (uint32_t j = 0; j < tiles_per_column; j += 2) {
                sdpa_fp32_state_update<2>(
                    out_in_cb,
                    out_out_cb,
                    bcast_cb,
                    (ob_row_base + i) * tiles_per_column + j,
                    read_row * tiles_per_column + j,
                    (write_row_base + i) * tiles_per_column + j,
                    ob_row_base + i,
                    identity_correction);
            }
        }
        sdpa_fp32_state_update<1, true>(
            sum_in_cb,
            sum_out_cb,
            bcast_cb,
            sum_row_base + i,
            read_row,
            write_row_base + i,
            ob_row_base + i,
            identity_correction);
    }
    PACK((llk_pack_reconfig_l1_acc(1)));
    return;
#endif

#ifdef SDPA_RECIPE_FP32
    mul_bcast_cols_init(out_in_cb, bcast_cb);
#endif

    CircularBuffer(out_in_cb).wait_front((ob_q_subblock + 1) * tiles_per_row * tiles_per_column * sdpa_out_stride);
    CircularBuffer(sum_in_cb).wait_front((sum_q_subblock + 1) * tiles_per_row * sdpa_sum_stride);
    CircularBuffer(bcast_cb).wait_front((ob_q_subblock + 1) * tiles_per_row);

#ifdef SDPA_RECIPE_FP32
    constexpr uint32_t last_batch_rem = tiles_per_column % col_batch;
    for (uint32_t col_base = 0; col_base < tiles_per_column; col_base += col_batch) {
        const uint32_t cur_cols =
            (col_base + col_batch <= tiles_per_column) ? col_batch : (last_batch_rem > 0 ? last_batch_rem : col_batch);
        const bool is_last_out_batch = (col_base + cur_cols >= tiles_per_column);
        const bool fuse_sum_here = can_fuse_last && is_last_out_batch;
#else
    static_assert(sbh_t == 2 && sbw_t == 4 && dst_size == 8);
    group2_numerator_row(
        out_in_cb,
        out_out_cb,
        bcast_cb,
        current_sum_popped ? write_row_base : sum_row_base,
        sum_row_base,
        current_sum_popped ? write_row_base : sum_row_base,
        write_row_base,
        identity_correction,
        group_boundary,
        group_odd,
        group_has_local);
    PACK((ckernel::sfpu::init_sdpa_compensated_block_macros()));
#endif

#ifdef SDPA_RECIPE_FP32
        tile_regs_acquire();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < cur_cols; j++) {
                uint32_t in0_tile_index = (ob_row_base + i) * tiles_per_column + col_base + j;
                mul_tiles_bcast_cols(out_in_cb, bcast_cb, in0_tile_index, ob_row_base + i, dst_index++);
            }
        }
        if (fuse_sum_here) {
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                mul_tiles_bcast_cols(sum_in_cb, bcast_cb, sum_row_base + i, ob_row_base + i, dst_index++);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_contiguous_rows(out_out_cb, write_row_base, tiles_per_row, tiles_per_column, col_base, cur_cols);
        dst_index = tiles_per_row * cur_cols;
        if (fuse_sum_here) {
            configure_single_tile_pack(sum_out_cb);
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                sdpa_pack_tile_ooo(dst_index++, sum_out_cb, write_row_base + i);
            }
        }
        tile_regs_release();
#else
    configure_single_tile_pack(sum_out_cb);
    if constexpr (tiles_per_row >= 2) {
        PACK((ckernel::sfpu::init_sdpa_compensated_sum_replay()));
#endif
    }
#ifdef SDPA_RECIPE_FP32

    if constexpr (!can_fuse_last && sdpa_sum_stride == 1) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            mul_tiles_bcast_cols(sum_in_cb, bcast_cb, sum_row_base + i, ob_row_base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(sum_out_cb);
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            sdpa_pack_tile_ooo(i, sum_out_cb, write_row_base + i);
        }
        tile_regs_release();
#else
    for (uint32_t i = 0; i + 1 < tiles_per_row; i += 2) {
        sdpa_compensated_state_update<2, 2, true>(
            sum_in_cb,
            sum_out_cb,
            bcast_cb,
            2 * (sum_row_base + i),
            2 * (sum_row_base + i) + 1,
            2 * (write_row_base + i),
            2 * (write_row_base + i) + 1,
            ob_row_base + i,
            2 * (current_sum_popped ? write_row_base + i : sum_row_base + i),
            identity_correction);
#endif
    }
#ifndef SDPA_RECIPE_FP32
    if constexpr (tiles_per_row % 2 != 0) {
        constexpr uint32_t i = tiles_per_row - 1;
        sdpa_compensated_state_update(
            sum_in_cb,
            sum_out_cb,
            bcast_cb,
            2 * (sum_row_base + i),
            2 * (sum_row_base + i) + 1,
            2 * (write_row_base + i),
            2 * (write_row_base + i) + 1,
            ob_row_base + i,
            2 * (current_sum_popped ? write_row_base + i : sum_row_base + i));
    }
    PACK((llk_pack_reconfig_l1_acc(1)));
#endif
}

/**
 * Per-row streaming normalization: matmul_reduce + recip-in-DST + mul_bcast_cols.
 * Consumes (pops) sum and output tiles, writes normalized output.
 * scratch_cb is a 1-tile CB reused for the reciprocal intermediate.
 */
template <
    bool profiling_enabled,
    uint32_t head_dim_t_,
    uint32_t dst_size,
    uint32_t col_identity_cb,
    uint32_t scratch_cb,
    uint32_t normalized_out_cb,
    uint32_t scale_fp32 = 0,
    bool use_attention_sink = false,
    uint32_t cb_attention_sink = INVALID_CB>
static __attribute__((noinline, noclone)) void normalize_row_streaming(
    uint32_t cur_sum_cb,
    uint32_t cur_out_cb,
    uint32_t sbh,
    [[maybe_unused]] uint32_t cur_max_cb_rt = 0,
    [[maybe_unused]] uint32_t sink_row_offset = 0) {
#ifdef SDPA_RECIPE_FP32
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t s = 0; s < sbh; ++s) {
        PACK(sdpa_pack_format_cb = INVALID_CB; sdpa_pack_width = 0;)
        CircularBuffer(cur_sum_cb).wait_front(1);
        CircularBuffer(col_identity_cb).wait_front(1);
        CircularBuffer(scratch_cb).reserve_back(1);
        sdpa_stream_reconfig(cur_sum_cb, cur_sum_cb);
        tile_regs_acquire();
        unary_bcast_init<BroadcastType::NONE>(cur_sum_cb);
        unary_bcast<BroadcastType::NONE>(cur_sum_cb, 0, 0);
        unary_bcast_uninit<BroadcastType::NONE>(cur_sum_cb);
        tile_regs_commit();
        tile_regs_wait();
        PACK(
            (SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, sdpa_state_reciprocal, 0, VectorMode::C)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        configure_single_tile_pack(scratch_cb);
        pack_tile(0, scratch_cb);
        tile_regs_release();
        CircularBuffer(scratch_cb).push_back(1);
        CircularBuffer(cur_sum_cb).pop_front(1);
        CircularBuffer(scratch_cb).wait_front(1);
        CircularBuffer(cur_out_cb).wait_front(head_dim_t_);
        CircularBuffer(normalized_out_cb).reserve_back(head_dim_t_);
        configure_single_tile_pack(normalized_out_cb);
        for (uint32_t j = 0; j < head_dim_t_; ++j) {
            tile_regs_acquire();
            sdpa_stream_reconfig(cur_out_cb, cur_out_cb);
            unary_bcast_init<BroadcastType::NONE>(cur_out_cb);
            unary_bcast<BroadcastType::NONE>(cur_out_cb, j, 0);
            unary_bcast_uninit<BroadcastType::NONE>(cur_out_cb);
            sdpa_stream_reconfig(scratch_cb, scratch_cb);
            unary_bcast_init<BroadcastType::COL>(scratch_cb);
            unary_bcast<BroadcastType::COL>(scratch_cb, 0, 1);
            unary_bcast_uninit<BroadcastType::COL>(scratch_cb);
            tile_regs_commit();
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, sdpa_state_normalize, 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile(0, normalized_out_cb);
            tile_regs_release();
        }
        CircularBuffer(normalized_out_cb).push_back(head_dim_t_);
        CircularBuffer(cur_out_cb).pop_front(head_dim_t_);
        CircularBuffer(scratch_cb).pop_front(1);
    }
    pack_reconfig_data_format(scratch_cb);
    return;
#endif

    // Attention sink: cb_attention_sink holds one raw per-head scalar tile. Broadcast it for each
    // row, compute exp((sink - max)*scale), and fold it into the col-reduced denominator (DST[0]).
    if constexpr (use_attention_sink) {
        CircularBuffer(cb_attention_sink).wait_front(1);
        CircularBuffer(cur_max_cb_rt).wait_front(sink_row_offset + sbh);
    }
    configure_single_tile_pack(scratch_cb);
    for (uint32_t s = 0; s < sbh; s++) {
        // 1+2. Fused matmul_reduce + recip: sum × col_identity → recip → 1/sum in scratch
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "NORM_MATMUL_RECIP");
            constexpr uint32_t N = 1;
            matmul_block_init(cur_sum_cb, col_identity_cb, 0, N, 1, N);
            sdpa_maybe_reconfig_data_format<normalized_out_cb, col_identity_cb, normalized_out_cb, scratch_cb>();
            // Pack format follows scratch_cb for the reciprocal intermediate. The old/new form folds away
            // when scratch and normalized output formats match, and reconfigures after rows that packed output.
            sdpa_maybe_pack_reconfig_data_format<normalized_out_cb, scratch_cb>();

            CircularBuffer(col_identity_cb).wait_front(N);
            CircularBuffer(cur_sum_cb).wait_front(sdpa_sum_stride);

            CircularBuffer(scratch_cb).reserve_back(1);
            tile_regs_acquire();
            matmul_block(cur_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
            if constexpr (sdpa_sum_stride == 2) {
                matmul_block(cur_sum_cb, col_identity_cb, 1, 0, 0, 0, N, 1, N);
            }
            if constexpr (use_attention_sink) {
                // DST[1] = exp((sink[s] - max[row_offset+s]) * scale); DST[0] += DST[1].
                // max - sink with a negated scale is equivalent and avoids expanding the sink
                // scalar to a first-column vector for every Q tile.
                sub_bcast_scalar_init(cur_max_cb_rt, cb_attention_sink);
                sub_tiles_bcast_scalar(cur_max_cb_rt, cb_attention_sink, sink_row_offset + s, 0, 1);
                // The custom first-column exp needs generic unary SFPU addrmod state, but not the
                // Blackhole approximate exp_init macro/replay setup used by exp_tile<true>.
                MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
                constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
                constexpr uint16_t negated_scale_bf16 = scale_bf16 ^ 0x8000;
                MATH((exp_tile_first_column<EXP_APPROX_MODE, negated_scale_bf16>(1)));
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            recip_tile_init<false>();
            MATH((recip_tile<false>(0 /*dst_index*/, VectorMode::C)));
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, scratch_cb);
            tile_regs_release();
            CircularBuffer(scratch_cb).push_back(1);

            CircularBuffer(cur_sum_cb).pop_front(sdpa_sum_stride);
        }

        // 3. Normalize: multiply output tiles by bcast_cols(1/sum)
        // Process in batches of up to dst_size tiles (DST capacity).
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "NORM_MUL_BCAST");
            constexpr uint32_t batch = (head_dim_t_ < dst_size) ? head_dim_t_ : dst_size;
            mul_bcast_cols_init(cur_out_cb, scratch_cb);
            // Pack output to normalized_out_cb; old/new skips when it has the same format as scratch.
            sdpa_maybe_pack_reconfig_data_format<scratch_cb, normalized_out_cb>();
            CircularBuffer(cur_out_cb).wait_front(head_dim_t_ * sdpa_out_stride);
            CircularBuffer(scratch_cb).wait_front(1);

            CircularBuffer(normalized_out_cb).reserve_back(head_dim_t_);
            for (uint32_t base = 0; base < head_dim_t_; base += batch) {
                constexpr uint32_t last_batch = head_dim_t_ % batch;
                const uint32_t cur_batch = (base + batch <= head_dim_t_) ? batch : last_batch;
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
            CircularBuffer(normalized_out_cb).push_back(head_dim_t_);

            CircularBuffer(scratch_cb).pop_front(1);
            CircularBuffer(cur_out_cb).pop_front(head_dim_t_ * sdpa_out_stride);
        }
    }
    // Restore pack format to scratch_cb (im_df = Float16_b) so that subsequent ops
    // (e.g. salad_correct_fused on the next K-chunk's drain row) pack to F16b CBs
    // with the right format. Without this, when normalized_out_cb has a different
    // format (e.g. Bfp8 output dtype), the format register stays Bfp8 and the next
    // pack to a F16b CB writes garbage that's later mis-decoded by F16b unpacks.
    sdpa_maybe_pack_reconfig_data_format<normalized_out_cb, scratch_cb>();
}

// ===================== Streaming SDPA Core Functions =====================

/**
 * L1-accumulate a single mask tile onto one position in out_cb.
 * Minimal primitive used by the lightweight ring mask path.
 */
static inline void l1_acc_single_tile(uint32_t mask_cb, uint32_t tile_idx, uint32_t out_cb, uint32_t out_pos) {
    tile_regs_acquire();
    copy_tile(mask_cb, tile_idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    sdpa_pack_tile_ooo(0, out_cb, out_pos);
    tile_regs_release();
}

/**
 * L1-accumulate a contiguous run of `count` mask tiles (mask_cb[mask_base + i]) onto the matching
 * contiguous out positions (out_cb[out_base + i]). Tiles are processed in DEST_AUTO_LIMIT-sized
 * batches so the tile_regs acquire/release overhead is paid once per batch, not once per tile.
 * Caller brackets with begin/end_mask_l1_accumulate (sets up the copy + L1-accumulate pack state).
 */
static inline void l1_acc_tile_run(
    uint32_t mask_cb, uint32_t mask_base, uint32_t out_cb, uint32_t out_base, uint32_t count) {
    constexpr uint32_t kBatch = compute_kernel_lib::DEST_AUTO_LIMIT;
    for (uint32_t i = 0; i < count; i += kBatch) {
        const uint32_t batch = (count - i) < kBatch ? (count - i) : kBatch;
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; j++) {
            copy_tile(mask_cb, mask_base + i + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; j++) {
            pack_tile<true>(j, out_cb, out_base + i + j);
        }
        tile_regs_release();
    }
}

static inline void l1_acc_neginf_cols(
    uint32_t mask_cb, uint32_t out_cb, uint32_t row_offset, uint32_t start_col, uint32_t end_col, uint32_t neginf_idx) {
    for (uint32_t col = start_col; col < end_col; col++) {
        l1_acc_single_tile(mask_cb, neginf_idx, out_cb, row_offset + col);
    }
}

static inline void l1_acc_causal_col_mask(
    uint32_t mask_cb,
    uint32_t out_cb,
    uint32_t row_offset,
    uint32_t col,
    int32_t q_pos,
    int32_t k_pos,
    uint32_t neginf_idx,
    uint32_t causal_diag_idx) {
    if (k_pos > q_pos) {
        l1_acc_single_tile(mask_cb, neginf_idx, out_cb, row_offset + col);
    } else if (k_pos == q_pos) {
        l1_acc_single_tile(mask_cb, causal_diag_idx, out_cb, row_offset + col);
    }
}

/**
 * Row-local mask stamping primitives for one Q row of a subblock. All column writes clamp to
 * [0, mask_cols), so callers can pass raw (possibly out-of-range) diagonal columns directly.
 */
struct RowMaskStamper {
    uint32_t mask_cb;
    uint32_t out_cb;
    uint32_t neginf_idx;
    uint32_t row_offset;
    uint32_t mask_cols;

    // L1-accumulate diagonal tile `tile_idx` at column `col` (no-op if col out of range).
    inline void stamp_tile_at(int32_t col, uint32_t tile_idx) const {
        if (col >= 0 && static_cast<uint32_t>(col) < mask_cols) {
            l1_acc_single_tile(mask_cb, tile_idx, out_cb, row_offset + static_cast<uint32_t>(col));
        }
    }
    // Fill columns [0, end) with neginf.
    inline void neginf_prefix(int32_t end) const {
        if (end > 0) {
            const uint32_t e = static_cast<uint32_t>(end) < mask_cols ? static_cast<uint32_t>(end) : mask_cols;
            l1_acc_neginf_cols(mask_cb, out_cb, row_offset, 0, e, neginf_idx);
        }
    }
    // Fill columns [start, mask_cols) with neginf (start <= 0 ⇒ whole row).
    inline void neginf_suffix(int32_t start) const {
        const uint32_t s =
            start <= 0 ? 0u : (static_cast<uint32_t>(start) < mask_cols ? static_cast<uint32_t>(start) : mask_cols);
        l1_acc_neginf_cols(mask_cb, out_cb, row_offset, s, mask_cols, neginf_idx);
    }
};

// Stamp the leading (left) sliding-window edge for one Q row: neginf everything before the window,
// then the diagonal edge tile(s). When the edge straddles two K tiles (leading_remainder != 0) the
// previous tile carries the neginf run and both straddle tiles are stamped.
template <uint32_t leading_base_tiles, uint32_t leading_remainder>
static inline void stamp_sliding_leading_edge(
    const RowMaskStamper& stamper,
    int32_t q_pos,
    uint32_t k_start_tile,
    uint32_t sliding_leading_prev_idx,
    uint32_t sliding_leading_idx) {
    const int32_t leading_col = q_pos - static_cast<int32_t>(leading_base_tiles) - static_cast<int32_t>(k_start_tile);
    if constexpr (leading_remainder == 0) {
        // Window edge tile-aligned: single leading diagonal tile.
        stamper.neginf_prefix(leading_col);
        stamper.stamp_tile_at(leading_col, sliding_leading_idx);
    } else {
        // Window edge straddles two K tiles: prev tile carries the neginf run.
        const int32_t leading_prev_col = leading_col - 1;
        stamper.neginf_prefix(leading_prev_col);
        stamper.stamp_tile_at(leading_prev_col, sliding_leading_prev_idx);
        stamper.stamp_tile_at(leading_col, sliding_leading_idx);
    }
}

// Stamp the trailing (right) sliding-window edge for one Q row (non-causal centered windows): the
// diagonal edge tile(s) followed by neginf for everything past the window. When the edge straddles
// into the next K tile (trailing_remainder != 0) both straddle tiles are stamped.
template <uint32_t trailing_base_tiles, uint32_t trailing_remainder>
static inline void stamp_sliding_trailing_edge(
    const RowMaskStamper& stamper,
    int32_t q_pos,
    uint32_t k_start_tile,
    uint32_t trailing_primary_idx,
    uint32_t sliding_trailing_next_idx) {
    const int32_t trailing_col = q_pos + static_cast<int32_t>(trailing_base_tiles) - static_cast<int32_t>(k_start_tile);
    if constexpr (trailing_remainder == 0) {
        // Window edge tile-aligned: single trailing diagonal tile.
        if (trailing_col < 0) {
            stamper.neginf_suffix(0);
        } else if (static_cast<uint32_t>(trailing_col) < stamper.mask_cols) {
            stamper.stamp_tile_at(trailing_col, trailing_primary_idx);
            stamper.neginf_suffix(trailing_col + 1);
        }
    } else if (trailing_col < -1) {
        stamper.neginf_suffix(0);
    } else {
        // Window edge straddles into the next K tile.
        stamper.stamp_tile_at(trailing_col, trailing_primary_idx);
        stamper.stamp_tile_at(trailing_col + 1, sliding_trailing_next_idx);
        stamper.neginf_suffix(trailing_col + 2);
    }
}

/**
 * Combined lightweight mask for streaming ring SDPA. Applies causal, partial, and padded masks.
 * KV-pad rotation reuses the causal path with a compile-time-selected Q row mapping.
 * Caller must set up copy_init and llk_pack_reconfig_l1_acc(1) before calling,
 * and llk_pack_reconfig_l1_acc(0) after calling.
 */
template <
    uint32_t num_cols,
    bool is_causal_sdpa,
    bool kv_pad_rotation_enabled = false,
    uint32_t kv_pad_q_local_padded_Nt = 0,
    uint32_t kv_pad_chunk_size_t = 0,
    uint32_t kv_pad_kv_local_padded_Nt = 0,
    uint32_t sliding_window_size = 0>
static void apply_lightweight_mask_streaming(
    uint32_t mask_cb,
    uint32_t out_cb,
    uint32_t q_subblock,
    uint32_t num_padded,
    bool has_partial,
    uint32_t partial_tile_idx,
    uint32_t sbh,
    bool apply_causal,
    uint32_t neginf_idx,
    uint32_t primary_diag_idx,
    uint32_t sliding_leading_prev_idx,
    uint32_t sliding_leading_idx,
    uint32_t sliding_trailing_next_idx,
    uint32_t q_start_tile,
    uint32_t k_start_tile,
    uint32_t active_Sk,
    bool apply_sliding_window = false,
    uint32_t straddle_col = 0,
    uint32_t straddle_jump = 0,
    const KVPadRotationContext& kv_pad_rotation = {}) {
    // This constrains the inner lightweight-mask path, not the kernel-level causal flag.
    // Chunked prefill re-enables this path when calling sdpa_inner_loop_step.
    static_assert(!kv_pad_rotation_enabled || is_causal_sdpa, "KV-pad rotation mask is causal-only");

    // Caller-owned contract (see function comment): pack state for mask_cb is initialized
    // before entry via copy_init + llk_pack_reconfig_l1_acc(1).
    // Per-row stamp geometry: floor division + remainder, distinct from the ceil-based loop
    // bounds in SlidingWindowLoopGeometry (causal reach here is `window`, not `window - 1`).
    constexpr bool has_sliding_window = sliding_window_size > 0;
    constexpr uint32_t half_window = sliding_window_size / 2;
    constexpr uint32_t leading_base_tiles =
        has_sliding_window ? (is_causal_sdpa ? (sliding_window_size / TILE_HEIGHT) : (half_window / TILE_HEIGHT)) : 0;
    constexpr uint32_t leading_remainder =
        has_sliding_window ? (is_causal_sdpa ? (sliding_window_size % TILE_HEIGHT) : (half_window % TILE_HEIGHT)) : 0;
    constexpr uint32_t trailing_base_tiles = has_sliding_window && !is_causal_sdpa ? (half_window / TILE_HEIGHT) : 0;
    constexpr uint32_t trailing_remainder = has_sliding_window && !is_causal_sdpa ? (half_window % TILE_HEIGHT) : 0;
    for (uint32_t row = 0; row < sbh; row++) {
        uint32_t row_offset = (q_subblock * sbh + row) * num_cols;

        // Causal / sliding mask: per-row diagonal stamps plus full-neginf regions.
        if constexpr (is_causal_sdpa || has_sliding_window) {
            if (apply_causal || apply_sliding_window || kv_pad_rotation_enabled) {
                const uint32_t q_tile = q_subblock * sbh + row;
                const uint32_t q_pos_u32 =
                    q_global_tile_for_mask_row<kv_pad_rotation_enabled>(q_tile, q_start_tile, kv_pad_rotation);
                const uint32_t mask_cols = kv_pad_rotation_enabled ? num_cols : active_Sk;
                if constexpr (kv_pad_rotation_enabled) {
                    if (q_pos_u32 == KV_PAD_ROTATION_INVALID_TILE) {
                        l1_acc_neginf_cols(mask_cb, out_cb, row_offset, 0, mask_cols, neginf_idx);
                        continue;
                    }
                }

                const int32_t q_pos = static_cast<int32_t>(q_pos_u32);
                [[maybe_unused]] const RowMaskStamper stamper{mask_cb, out_cb, neginf_idx, row_offset, mask_cols};

                if constexpr (has_sliding_window) {
                    if (apply_sliding_window) {
                        stamp_sliding_leading_edge<leading_base_tiles, leading_remainder>(
                            stamper, q_pos, k_start_tile, sliding_leading_prev_idx, sliding_leading_idx);
                    }
                }

                if constexpr (kv_pad_rotation_enabled) {
                    for (uint32_t col = 0; col < mask_cols; col++) {
                        const uint32_t local_k_tile = kv_pad_rotation.k_local_start_tile + col;
                        if (local_k_tile >= kv_pad_kv_local_padded_Nt) {
                            l1_acc_single_tile(mask_cb, neginf_idx, out_cb, row_offset + col);
                            continue;
                        }
                        const uint32_t k_pos_u32 =
                            chunked_kv_global_tile_for_local<kv_pad_chunk_size_t, kv_pad_q_local_padded_Nt>(
                                kv_pad_rotation.ring_id, local_k_tile);
                        if (k_pos_u32 >= kv_pad_rotation.logical_tile_count) {
                            l1_acc_single_tile(mask_cb, neginf_idx, out_cb, row_offset + col);
                            continue;
                        }
                        const int32_t k_pos = static_cast<int32_t>(k_pos_u32);
                        l1_acc_causal_col_mask(
                            mask_cb, out_cb, row_offset, col, q_pos, k_pos, neginf_idx, primary_diag_idx);
                    }
                } else if (straddle_col == 0) {
                    // Fast path: K coords contiguous across cols.
                    if (apply_causal) {
                        const int32_t diag_col = q_pos - static_cast<int32_t>(k_start_tile);
                        if (diag_col < 0) {
                            stamper.neginf_suffix(0);
                        } else if (static_cast<uint32_t>(diag_col) < mask_cols) {
                            stamper.stamp_tile_at(diag_col, primary_diag_idx);
                            stamper.neginf_suffix(diag_col + 1);
                        }
                    } else if constexpr (has_sliding_window && !is_causal_sdpa) {
                        if (apply_sliding_window) {
                            stamp_sliding_trailing_edge<trailing_base_tiles, trailing_remainder>(
                                stamper, q_pos, k_start_tile, primary_diag_idx, sliding_trailing_next_idx);
                        }
                    }
                } else {
                    // Chunked-prefill straddle: K coord jumps by straddle_jump at col >= straddle_col
                    // (the K-chunk crosses a slab boundary). Evaluate per-col.
                    if (apply_causal) {
                        for (uint32_t col = 0; col < mask_cols; col++) {
                            int32_t k_pos = static_cast<int32_t>(k_start_tile) + static_cast<int32_t>(col);
                            if (col >= straddle_col) {
                                k_pos += static_cast<int32_t>(straddle_jump);
                            }
                            l1_acc_causal_col_mask(
                                mask_cb, out_cb, row_offset, col, q_pos, k_pos, neginf_idx, primary_diag_idx);
                        }
                    }
                }
            }
        }

        if constexpr (!kv_pad_rotation_enabled) {
            // Padding mask: partial tile + fully-padded columns (unchanged)
            const uint32_t boundary_col = num_cols - num_padded - (has_partial ? 1 : 0);
            if (has_partial) {
                l1_acc_single_tile(mask_cb, partial_tile_idx, out_cb, row_offset + boundary_col);
            }

            uint32_t start = num_cols - num_padded;
            for (uint32_t col = start; col < num_cols; col++) {
                l1_acc_single_tile(mask_cb, neginf_idx, out_cb, row_offset + col);
            }
        }
    }
}

/**
 * Add the dense user mask to one q_subblock row group of a K chunk: QK[r,c] += mask[r,c] for every
 * tile (additive-bias semantics, reproducing any attn_mask exactly), vs the lightweight path's
 * sparse constant-palette stamp.
 *
 * cb_qkt_im is indexed by the absolute row group (out_q_subblock, out_stride=KT_stride); the mask CB
 * is front-relative (the caller pops it per subblock, so this subblock's first row is at the front)
 * with mask_stride=Sk_chunk_t. Each row's active_Sk tiles are contiguous in both CBs, so
 * l1_acc_tile_run batches them. sbh/out_stride/mask_stride are compile-time at every call site, so
 * they are template params and the per-row offset math folds away.
 *
 * Caller brackets with begin_mask_l1_accumulate / end_mask_l1_accumulate (same as the lightweight path).
 */
template <uint32_t sbh, uint32_t out_stride, uint32_t mask_stride>
static void apply_provided_mask_streaming(
    uint32_t mask_cb, uint32_t out_cb, uint32_t out_q_subblock, uint32_t active_Sk) {
    for (uint32_t row = 0; row < sbh; row++) {
        const uint32_t out_offset = (out_q_subblock * sbh + row) * out_stride;
        const uint32_t mask_offset = row * mask_stride;
        l1_acc_tile_run(mask_cb, mask_offset, out_cb, out_offset, active_Sk);
    }
}

/**
 * Open a single-tile L1-accumulate pack onto cb_qkt_im for a mask stamp/apply. MOP is configured
 * for actual_sbw tiles (blocked matmul), so reconfigure pack state for 1 tile per pack and enable
 * L1 accumulation. Close with end_mask_l1_accumulate.
 *
 * reconfig_dt: when true, reconfigure the unpacker srcA data format (fp16 from Q@KT) to the mask
 * format so block-float (bfp8/bfp4) masks are not mis-decoded as fp16; no-op when formats match.
 * The dense provided-mask path needs this; the lightweight palette is always fp16 and does not.
 */
template <bool reconfig_dt>
static inline void begin_mask_l1_accumulate(uint32_t cb_qkt_im, uint32_t cb_mask_in) {
    configure_single_tile_pack(cb_qkt_im);
    if constexpr (reconfig_dt) {
        sdpa_stream_reconfig_srca(cb_qkt_im, cb_mask_in);
        copy_init(cb_mask_in);
    } else {
        copy_init(cb_mask_in);
    }
    PACK((llk_pack_reconfig_l1_acc(1)));
}

static inline void end_mask_l1_accumulate() { PACK((llk_pack_reconfig_l1_acc(0))); }

/**
 * Largest factor of n that is <= max_val.  Picks a QKT subblock width that
 * evenly divides the active K-tile count on the last K-chunk, avoiding
 * partial subblocks and enabling the split-drain path.
 */
constexpr uint32_t largest_factor_le(uint32_t n, uint32_t max_val) {
    for (uint32_t f = max_val; f >= 2; --f) {
        if (n % f == 0) {
            return f;
        }
    }
    return 1;
}

// --- lightweight-mask predicates (single source of truth for "does this op / iter stamp a mask") ---

// Compile-time: does this op configuration ever use the structured lightweight (causal / padding /
// sliding-window) mask? Also selects the lightweight branch via `if constexpr`.
template <bool ring_mode, bool is_causal, bool use_padded_mask, uint32_t sliding_window_size>
constexpr bool sdpa_uses_lightweight_mask() {
    return ring_mode || is_causal || use_padded_mask || sliding_window_size > 0;
}

// Runtime: does *this* K-chunk iteration actually stamp the lightweight mask?
inline bool sdpa_lightweight_mask_stamped(
    bool kv_pad_rotation_enabled, bool causal_applies, bool apply_sliding_window, bool partial_tile_mask) {
    return kv_pad_rotation_enabled || causal_applies || apply_sliding_window || partial_tile_mask;
}

// Can the reduce's first half (run()#1, cols [0, active_Sk/2)) overlap the second-half pack? Only if
// no masked column lands there: either nothing is stamped (no_mask_this_iter), or — on the plain
// contiguous causal path — this q_subblock's lowest diagonal column (first_q_tile = q_subblock *
// qkt_subblock_h) is already in the second half. A causal mask writes only cols >= the diagonal, and
// a fully-visible column's value is its raw score, so reading it pre-mask is correct. Every other
// variant (sliding window, kv-pad rotation, K straddle, padding/partial, provided mask) -> false.
template <
    bool is_causal_sdpa,
    bool kv_pad_rotation_enabled,
    uint32_t sliding_window_size,
    bool use_provided_mask,
    uint32_t Sk_chunk_t>
inline bool sdpa_first_half_unmasked(
    bool no_mask_this_iter,
    bool apply_causal,
    bool apply_sliding_window,
    bool apply_mask,
    uint32_t lw_partial_tile_idx,
    uint32_t mask_straddle_col,
    uint32_t mask_q_start_tile,
    uint32_t mask_k_start_tile,
    uint32_t first_q_tile,
    uint32_t active_Sk) {
    if (no_mask_this_iter) {
        return true;
    }
    constexpr bool has_sliding_window = sliding_window_size > 0;
    if constexpr (is_causal_sdpa && !kv_pad_rotation_enabled && !has_sliding_window && !use_provided_mask) {
        const bool plain_causal_stamp = apply_causal && !apply_sliding_window && mask_straddle_col == 0 &&
                                        active_Sk == Sk_chunk_t && !(apply_mask && lw_partial_tile_idx > 0);
        const int32_t first_diag_col =
            static_cast<int32_t>(mask_q_start_tile + first_q_tile) - static_cast<int32_t>(mask_k_start_tile);
        return plain_causal_stamp && first_diag_col >= static_cast<int32_t>(active_Sk / 2);
    }
    return false;
}

/**
 * One K-chunk iteration of the streaming SDPA algorithm (v2 — no row buffers).
 * Phase 1: Q@KT directly into cb_qkt_im with cb_push_back_hold_wr_ptr, in-place sub_exp.
 * Phase 2: Drain + QKT@V with SALAD corrections, streaming normalization on last K iter.
 */
template <
    bool profiling_enabled,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_h,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_h,
    uint32_t qktv_subblock_w,
    bool use_padded_mask,
    bool ring_mode = false,
    bool is_causal_sdpa = false,
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
    uint32_t KT_stride = Sk_chunk_t,
    bool kv_pad_rotation_enabled = false,
    uint32_t kv_pad_q_local_padded_Nt = 0,
    uint32_t kv_pad_chunk_size_t = 0,
    uint32_t kv_pad_kv_local_padded_Nt = 0,
    uint32_t v_cb_physical_width_t = vDHt,
    bool kt_inplace_v = false,
    uint32_t sliding_window_size = 0,
    bool use_attention_sink = false,
    uint32_t cb_attention_sink = INVALID_CB,
    bool use_provided_mask = false,
    // Compile-time gate for q_base_tiles: only the head-serial ring passes read Q at an offset,
    // and every other caller keeps the original constant-zero index math (and codegen).
    bool has_q_base_tiles = false>
static void sdpa_inner_loop_step(
    AccumulatorHalf& prev,
    AccumulatorHalf& cur,
    const bool is_last_iter,
    const bool is_first_iter,
    [[maybe_unused]] const bool apply_mask = false,
    const uint32_t lw_partial_tile_idx = 0,
    const uint32_t active_Sk = Sk_chunk_t,
    const bool reduce_trigger = false,
    const uint32_t actual_sbw = qkt_subblock_w,
    const uint32_t save_out_cb = INVALID_CB,
    const uint32_t save_max_cb = INVALID_CB,
    const bool apply_causal = false,
    const uint32_t mask_q_start_tile = 0,
    const uint32_t mask_k_start_tile = 0,
    const uint32_t neginf_idx = 0,
    const uint32_t primary_diag_idx = 0,
    const uint32_t sliding_leading_prev_idx = 0,
    const uint32_t sliding_leading_idx = 0,
    const uint32_t sliding_trailing_next_idx = 0,
    const bool apply_sliding_window = false,
    const uint32_t mask_straddle_col = 0,
    const uint32_t mask_straddle_jump = 0,
    const KVPadRotationContext& kv_pad_rotation = {},
// Tile offset of this call's Q chunk from the front of cb_q_in. Non-zero only for head-serial
// ring passes, where cb_q_in holds one resident Q chunk per pass and is popped once at the end.
#ifdef SDPA_RECIPE_FP32
    const uint32_t q_base_tiles = 0) {
    PACK(sdpa_pack_format_cb = INVALID_CB; sdpa_pack_width = 0;)
    sdpa_skip_prev_sum_pop = false;
#else
    const uint32_t q_base_tiles = 0,
    const uint32_t group_k_index = 0,
    bool* group_local_valid = nullptr) {
    static_assert(
        !ring_mode && Sq_chunk_t == 8 && Sk_chunk_t == 16 && vDHt == 4 && qkt_subblock_h == 2 && qktv_subblock_h == 2,
        "Compensated Q256/K512/D128 entry owns four per-Q validity flags");
#endif
    // Callers guarantee active_Sk is evenly divisible by actual_sbw (via largest_factor_le).
    const uint32_t kt_num_full_subblocks = active_Sk / actual_sbw;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t in0_block_w = DHt;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qkt_subblock_h;
    constexpr uint32_t q_subblock_num_tiles = qkt_subblock_h * in0_block_w;
    constexpr uint32_t row_tiles = qkt_subblock_h * KT_stride;  // Use KT_stride for cb_qkt_im row width
    static_assert(!(use_padded_mask && ring_mode), "use_padded_mask and ring_mode are mutually exclusive");

#ifdef SDPA_RECIPE_FP32
    static_assert(qkt_subblock_h == 1, "Early identity scan handles one query tile row per QK subblock");
    uint32_t sdpa_early_identity = !is_first_iter && save_out_cb == INVALID_CB;
    auto scan_identity_row = [&](uint32_t row) {
        UNPACK({
            if (sdpa_early_identity) {
                CircularBuffer(prev.max).wait_front(row + 1);
                CircularBuffer(cur.max).wait_front(row + 1);
                auto* old_max =
                    reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(prev.max), row));
                auto* new_max =
                    reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(cur.max), row));
                for (uint32_t r = 0; r < 32; r += 4) {
                    const uint32_t offset = (r / 16) * 256 + (r % 16) * 8;
                    const uint32_t a0 = old_max[offset], a1 = old_max[offset + 8];
                    const uint32_t a2 = old_max[offset + 16], a3 = old_max[offset + 24];
                    const uint32_t diff = (a0 ^ new_max[offset]) | (a1 ^ new_max[offset + 8]) |
                                          (a2 ^ new_max[offset + 16]) | (a3 ^ new_max[offset + 24]);
                    if ((diff & 0xffffu) || (a0 & 0x7f80u) == 0x7f80u || (a1 & 0x7f80u) == 0x7f80u ||
                        (a2 & 0x7f80u) == 0x7f80u || (a3 & 0x7f80u) == 0x7f80u) {
                        sdpa_early_identity = 0;
                        break;
                    }
                }
            }
        })
    };
#else
    static_assert(
        Sq_chunk_t == 8 && qkt_subblock_h == 2 && qktv_subblock_h == 2 && !kt_inplace_v,
        "Early guard is qualified only for the fixed Q256/K512 materialized-V geometry");
    // Per-K-step flags; only UNPACK fills these. Other threads obtain the
    // matching decision at the original correction mailbox rendezvous.
    uint32_t sdpa_identity_flags[Sq_chunk_t] = {};
#endif
    uint32_t pushed_rows = 0;
    // Q lives at [q_base_tiles, q_base_tiles + Sq_chunk_t*DHt) from the CB front. wait_front counts
    // from the front, so the wait target includes the chunks of earlier passes that stay resident.
    uint32_t q_wait_tiles = (has_q_base_tiles ? q_base_tiles : 0) + q_subblock_num_tiles;
    uint32_t q_index_offset = has_q_base_tiles ? q_base_tiles : 0;
    uint32_t kt_index_offset = 0;

    exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
#ifdef SDPA_RECIPE_FP32
    PACK((ckernel::sfpu::init_sdpa_refine_loadmacros()));
#endif

    // Use KT_stride for cb_qkt_im layout to keep CB pointers aligned across iterations
    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT_stride);

    CircularBuffer(cur.sum).reserve_back(Sq_chunk_t * sdpa_sum_stride);
#ifndef SDPA_RECIPE_FP32
    if (is_first_iter) {
        tile_regs_acquire();
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_zero_sum, 0, VectorMode::None)));
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(cur.sum);
        PACK((llk_pack_reconfig_l1_acc(0)));
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            pack_tile<true>(0, cur.sum, 2 * r + 1);
        }
        tile_regs_release();
    }
#endif
    if (save_max_cb != INVALID_CB) {
        CircularBuffer(save_max_cb).reserve_back(Sq_chunk_t);
    }

    // ========== PHASE 1: Q@KT directly into cb_qkt_im ==========
    // All matmul output goes to cb_qkt_im at absolute offsets via pack_tile<true>.
    // cb_push_back_hold_wr_ptr makes each row visible to UNPACK without advancing wr_ptr.
    CircularBuffer(cb_kt_in).wait_front(DHt * KT_stride);

    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        MaybeDeviceZoneScopedN(profiling_enabled, "Softmax(Q@KT)");
        CircularBuffer(cb_q_in).wait_front(q_wait_tiles);
        kt_index_offset = 0;

        sdpa_maybe_pack_reconfig_data_format<cb_normalized_out, cb_qkt_im>();
#ifdef SDPA_RECIPE_FP32
        sdpa_stream_reconfig(cb_kt_in, cb_q_in);
#else
        sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_identity_scale_in, cb_q_in>();
#endif
        mm_no_mop_init_short(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);
#ifdef SDPA_RECIPE_FP32
#ifndef SDPA_RECIPE_ACCURATE
        MATH((llk_math_matmul_init_no_mop<MathFidelity::HiFi4, MM_THROTTLE>(
            cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h)));
#endif
#endif
        // Configure pack once before the kt loop for cb_qkt_im. Both sub_exp
        // and blocked_matmul_and_pack skip their internal configure (same cb+width).
        // sub_exp's configure_single_tile_pack(reduce_cb) clobbers the global to 1,
        // so blocked_matmul_and_pack must reconfigure when q_subblock > 0.
        // When q_subblock == 0, no sub_exp → global stays set → skip there too.
        configure_row_pack_width(cb_qkt_im, actual_sbw);

        // Mask plan for this q_subblock (single source of truth; reused by the mask stamp below).
        constexpr bool uses_lightweight_mask =
            sdpa_uses_lightweight_mask<ring_mode, is_causal_sdpa, use_padded_mask, sliding_window_size>();
        const bool should_apply_lightweight_mask = sdpa_lightweight_mask_stamped(
            kv_pad_rotation_enabled,
            is_causal_sdpa && apply_causal,
            apply_sliding_window,
            apply_mask && lw_partial_tile_idx > 0);
        const bool no_mask_this_iter = !use_provided_mask && !(uses_lightweight_mask && should_apply_lightweight_mask);

        // run()#1 (the reduce's first half) may overlap the second-half pack only if no masked
        // column lands in [0, active_Sk/2) — see sdpa_first_half_unmasked.
        const bool overlap_first_half = reduce_trigger && sdpa_first_half_unmasked<
                                                              is_causal_sdpa,
                                                              kv_pad_rotation_enabled,
                                                              sliding_window_size,
                                                              use_provided_mask,
                                                              Sk_chunk_t>(
                                                              no_mask_this_iter,
                                                              apply_causal,
                                                              apply_sliding_window,
                                                              apply_mask,
                                                              lw_partial_tile_idx,
                                                              mask_straddle_col,
                                                              mask_q_start_tile,
                                                              mask_k_start_tile,
                                                              q_subblock * qkt_subblock_h,
                                                              active_Sk);
        // PACK posts the first-half token after the subblock covering the last first-half column
        // [active_Sk/2 - 1]; committing a superset of [0, active_Sk/2) is safe (run()#1's cols are a subset).
        const uint32_t first_half_last_sb = (active_Sk / 2 - 1) / actual_sbw;

        {
            for (uint32_t kt_subblock = 0; kt_subblock < kt_num_full_subblocks; ++kt_subblock) {
                if (q_subblock > 0) {
                    uint32_t prev_q_subblock = q_subblock - 1;
#ifndef SDPA_RECIPE_FP32
                    sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
#endif
                    sub_exp_block_bcast_cols<profiling_enabled, scale_fp32>(
                        cb_qkt_im,
                        cur.max,
                        cur.sum,
                        KT_stride,
                        prev_q_subblock,
                        kt_subblock * actual_sbw,
                        qkt_subblock_h,
                        actual_sbw,
                        /*skip_pack_configure=*/true);
                    sdpa_maybe_pack_reconfig_data_format<cb_recip_scratch, cb_qkt_im>();
#ifdef SDPA_RECIPE_FP32
                    sdpa_stream_reconfig_srca(cb_kt_in);
#else
                    sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in>();
#endif
                    mm_no_mop_reinit_short(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);
#ifdef SDPA_RECIPE_FP32
#ifndef SDPA_RECIPE_ACCURATE
                    MATH((llk_math_matmul_reinit_no_mop<MathFidelity::HiFi4, MM_THROTTLE>(
                        cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h)));
#endif
#endif
                }
                {
                    MaybeDeviceZoneScopedN(profiling_enabled, "Q@KT MM+Pack");
                    blocked_matmul_and_pack<true, KT_stride, KT_stride>(
                        cb_q_in,
                        cb_kt_in,
                        cb_qkt_im,
                        q_index_offset,
                        kt_index_offset,
                        q_subblock,
                        kt_subblock * actual_sbw,
                        actual_sbw,
                        qkt_subblock_h,
                        in0_block_w,
                        in0_block_w,
#ifdef SDPA_RECIPE_FP32
                        /*skip_pack_configure=*/true);
                    // The previous row's max was already consumed by sub_exp
                    // above. Scan it after issuing current QK so scalar L1 loads
                    // can overlap the FPU, without a per-row cross-RISC mailbox.
                    if (q_subblock > 0 && kt_subblock == 0) {
                        scan_identity_row(q_subblock - 1);
                    }
#else
                        /*skip_pack_configure=*/q_subblock == 0);
#endif
                    // Signal the first half is committed so run()#1 overlaps the second-half
                    // pack. STALL_PACK drains the L1 writes before the token retires.
                    if (overlap_first_half && kt_subblock == first_half_last_sb) {
                        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::UNPACK_MATH_DONE)));
                    }
#ifndef SDPA_RECIPE_FP32
                    UNPACK({
                        if (!is_first_iter && q_subblock > 0 && kt_subblock == 0) {
                            sdpa_identity_flags[q_subblock - 1] =
                                sdpa_scan_identity_maxima(prev.max, cur.max, q_subblock - 1, qkt_subblock_h);
                        }
                    })
#endif
                    kt_index_offset += actual_sbw;
                }
            }
        }
        // The FP32 max-reduce helper configures its own input views.
#ifndef SDPA_RECIPE_FP32
        sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
#endif

        // Mask stamp/apply: L1-accumulate the mask onto cb_qkt_im for this row group. A dense
        // user-provided mask and the structured lightweight palette are mutually exclusive — the
        // user mask supplies all masking itself (causal/sliding/padding baked in by the caller),
        // so exactly one of these branches is compiled. The only config that stamps nothing is plain
        // non-causal attention with no mask at all (uses_lightweight_mask == false).
        // uses_lightweight_mask hoisted above the kt loop.
        if constexpr (use_provided_mask) {
            // Dense user-provided mask: the full per-position mask, applied on every K chunk (no
            // chunk skipping); the user mask defines the visible region. begin_mask_l1_accumulate
            // reconfigs srcA to the mask format so block-float (bfp8/bfp4) masks decode correctly.
            // The reader streams the mask one Q-tile-row at a time; wait for / pop just this
            // subblock's row group so the wait overlaps the reader, and the mask front then sits at
            // this subblock's first row (apply indexes it mask-base 0).
            constexpr uint32_t mask_subblock_tiles = qkt_subblock_h * Sk_chunk_t;
            CircularBuffer(cb_mask_in).wait_front(mask_subblock_tiles);
            begin_mask_l1_accumulate<true>(cb_qkt_im, cb_mask_in);
            apply_provided_mask_streaming<qkt_subblock_h, KT_stride, Sk_chunk_t>(
                cb_mask_in,
                cb_qkt_im,
                q_subblock,  // cb_qkt_im base: absolute row group
                active_Sk);
            end_mask_l1_accumulate();
            // Restore srcA to fp16 for the following max reduce / next-subblock Q@KT.
            sdpa_stream_reconfig_srca(cb_mask_in, cb_qkt_im);
            CircularBuffer(cb_mask_in).pop_front(mask_subblock_tiles);
        } else if constexpr (uses_lightweight_mask) {
            // Lightweight stamp: causal and/or padding masks. Active for ring, causal non-ring, or
            // non-causal padded with a partial-tile mask (single-chip streaming partial-K case).
            // should_apply_lightweight_mask hoisted above the kt loop.
            if (should_apply_lightweight_mask) {
                begin_mask_l1_accumulate<false>(cb_qkt_im, cb_mask_in);
                apply_lightweight_mask_streaming<
                    KT_stride,
                    is_causal_sdpa,
                    kv_pad_rotation_enabled,
                    kv_pad_q_local_padded_Nt,
                    kv_pad_chunk_size_t,
                    kv_pad_kv_local_padded_Nt,
                    sliding_window_size>(
                    cb_mask_in,
                    cb_qkt_im,
                    q_subblock,
                    kv_pad_rotation_enabled ? 0 : Sk_chunk_t - active_Sk,
                    !kv_pad_rotation_enabled && (apply_mask && lw_partial_tile_idx > 0),
                    kv_pad_rotation_enabled ? 0 : lw_partial_tile_idx,
                    qkt_subblock_h,
                    kv_pad_rotation_enabled || apply_causal,
                    neginf_idx,
                    primary_diag_idx,
                    sliding_leading_prev_idx,
                    sliding_leading_idx,
                    sliding_trailing_next_idx,
                    mask_q_start_tile,
                    mask_k_start_tile,
                    kv_pad_rotation_enabled ? KT_stride : active_Sk,
                    apply_sliding_window,
                    mask_straddle_col,
                    mask_straddle_jump,
                    kv_pad_rotation);
                end_mask_l1_accumulate();
            }
        }

        // Push row (visible for UNPACK reads) but keep wr_ptr stable
        cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles);

        // reduce_trigger barrier. Posted after pack + mask + push so it dominates every
        // cb_qkt_im writer; gates run()#2 (and run()#1 on the non-overlap path). STALL_PACK drains
        // the writes; one post / one uninit get stays balanced (wait_on_zero is non-consuming).
        if (reduce_trigger) {
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::FPU_SFPU)));
        }

        // Max reduce: reads from cb_qkt_im at q_subblock position
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "Reduce max");
            CircularBuffer(cur.max).reserve_back(qkt_subblock_h);
            configure_single_tile_pack(cur.max);
            // Use reduce_trigger to enable early reduce start (before all matmul output is ready).
            // When reduce_trigger=true, the packer signals the unpacker via semaphore after partial output.
            reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT_stride>(
                cur.max,
                prev.max,
                q_subblock,
                !is_first_iter /*do_eltwise_max*/,
                qkt_subblock_h,
                active_Sk,
                reduce_trigger,
                save_max_cb,
                overlap_first_half);
            CircularBuffer(cur.max).push_back(qkt_subblock_h);
            if (save_max_cb != INVALID_CB) {
                CircularBuffer(save_max_cb).push_back(qkt_subblock_h);
            }
        }

        q_index_offset += qkt_subblock_h * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }

    // In-place latent-V reads K^T again in Phase 2, so defer the K^T pop until after the
    // softmax@V matmul (handled where the materialized-V pop would normally fire).
    if constexpr (!kt_inplace_v) {
        CircularBuffer(cb_kt_in).pop_front(DHt * KT_stride);
    }

    // Q is no longer needed after Phase 1. On the last K chunk, pop early so the
    // reader can start fetching the next Q chunk during Phase 2.
    // In ring_mode, is_last_iter is always false — skip entirely.
    if constexpr (!ring_mode) {
        if (is_last_iter) {
            sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
        }
    }

    // Lightweight ring mask tiles are permanently fronted — no pop needed.

    // ========== PHASE 2: Drain last row + QKT@V + SALAD ==========
    // After Phase 1: all rows are pushed (via hold_wr_ptr) in cb_qkt_im.
    // Rows 0..N-2 are softmax'd in-place; row N-1 has raw matmul output.
    {
        constexpr uint32_t qktv_h =
            ttnn::transformer::sdpa::streaming_qktv_h(qktv_subblock_h, qktv_subblock_w, dst_size, Sq_chunk_t);
        constexpr uint32_t qktv_remainder_h = Sq_chunk_t % qktv_h;
        constexpr bool has_qktv_remainder = qktv_remainder_h != 0;
        static_assert(Sq_chunk_t >= qktv_h, "Sq_chunk_t must be at least qktv_h");

        static_assert(vDHt % qktv_subblock_w == 0, "vDHt must be evenly divisible by qktv_subblock_w");
        static_assert(qktv_h * qktv_subblock_w <= dst_size, "qktv subblock must fit in dest register file");
        constexpr uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_h;  // full subblocks only
        constexpr uint32_t qktv_v_num_subblocks = vDHt / qktv_subblock_w;
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * vDHt * sdpa_out_stride;
        // cb_qkt_im row width is KT_stride (for pointer alignment), not Sk_chunk_t
        constexpr uint32_t qktv_in0_row_tiles = qktv_h * KT_stride;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_row_tiles;

        // When save_out_cb is set, V matmul + SALAD write to save_out_cb (cb_out) instead of cur.out.
        // Writer drains save_out_cb row-by-row to DRAM during SALAD. cur.out stays empty.
#ifdef SDPA_RECIPE_FP32
        static_assert(Sq_chunk_t == 8 && Sk_chunk_t == 16 && vDHt == 4 && qktv_h == 1);
        static_assert(!ring_mode && !is_causal_sdpa && !use_attention_sink && !kt_inplace_v);
        uint32_t inplace_numerator = !is_first_iter && save_out_cb == INVALID_CB;
        if (inplace_numerator) {
            CircularBuffer(prev.max).wait_front(Sq_chunk_t);
            CircularBuffer(cur.max).wait_front(Sq_chunk_t);
            UNPACK({
                scan_identity_row(Sq_chunk_t - 1);
                inplace_numerator = sdpa_early_identity;
                mailbox_write(ckernel::ThreadId::MathThreadId, inplace_numerator);
                mailbox_write(ckernel::ThreadId::PackThreadId, inplace_numerator);
            })
            MATH(inplace_numerator = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
            PACK(inplace_numerator = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        }
        if (inplace_numerator) {
            // Both physical banks retain their original 32-tile capacity.
            // The old bank is fully published. Pop it as a whole, preserving
            // its L1 bytes, then reuse it as this chunk's reserved output bank.
            // The alternate bank was empty and remains empty as prev.out.
            CircularBuffer(prev.out).wait_front(qktv_output_num_tiles);
            CircularBuffer(prev.out).pop_front(qktv_output_num_tiles);
            std::swap(prev.out, cur.out);
            // Phase 1's cur.sum reserve did not advance its write pointer
            // or publish tiles. It is safe to leave that empty bank unused.
            CircularBuffer(prev.sum).wait_front(Sq_chunk_t);
            CircularBuffer(prev.sum).pop_front(Sq_chunk_t);
            std::swap(prev.sum, cur.sum);
            CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);
            sdpa_skip_prev_sum_pop = true;
        }
#endif
        const uint32_t out_cb = (save_out_cb != INVALID_CB) ? save_out_cb : cur.out;
#ifndef SDPA_RECIPE_FP32
        // Odd K begins with empty local; put PV directly in its final local plane.
        const uint32_t group_pv_offset = (group_k_index % 2 == 1) ? vDHt : 0;
#endif

        // V wait deferred: don't block here. The sub_exp drain loop below
        // doesn't touch V, so the reader's V DMA can overlap with the drain.
        CircularBuffer(out_cb).reserve_back(qktv_output_num_tiles);
#ifndef SDPA_RECIPE_FP32
        if (is_first_iter) {
            tile_regs_acquire();
            MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_zero_sum, 0, VectorMode::None)));
            tile_regs_commit();
            tile_regs_wait();
            configure_single_tile_pack(out_cb);
            PACK((llk_pack_reconfig_l1_acc(0)));
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                for (uint32_t c = 0; c < vDHt; ++c) {
                    pack_tile<true>(0, out_cb, (2 * r + 1) * vDHt + c);
                }
            }
            tile_regs_release();
        }
#endif

        // q_subblock 0: drain last row's sub_exp in-place + first QKT@V matmul
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "Softmax(Q@KT)@V");
            const uint32_t matmul_inner = actual_sbw;
            const uint32_t drain_subblocks = kt_num_full_subblocks;

            // sub_exp_block_bcast_cols softmaxes the last Q row in place, one column-subblock at a
            // time. The PACK->UNPACK barrier after it makes those in-place pack writes visible to
            // the V-matmul unpack; only needed when q_num_subblocks==1 (Phase 1's hold_wr_ptr
            // didn't sync them).

            if constexpr (!kt_inplace_v) {
                // Split-drain (common, materialized-V path): interleave each column-subblock's
                // sub_exp with its partial V matmul; partial products accumulate across kt_sub via L1.
                for (uint32_t kt_sub = 0; kt_sub < drain_subblocks; ++kt_sub) {
                    sub_exp_block_bcast_cols<profiling_enabled, scale_fp32>(
                        cb_qkt_im,
                        cur.max,
                        cur.sum,
                        KT_stride,
                        q_num_subblocks - 1,
                        kt_sub * matmul_inner,
                        qkt_subblock_h,
                        matmul_inner);
                    if constexpr (q_num_subblocks == 1) {
                        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                    }
#ifdef SDPA_RECIPE_FP32
                    if (kt_sub == 0) {
#else
                    if (kt_sub == 0) {
#endif
                        CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
                        CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                    }
#ifdef SDPA_RECIPE_FP32
                    if (kt_sub > 0 || inplace_numerator) {
#else
                    if (kt_sub > 0) {
#endif
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }

#ifdef SDPA_RECIPE_FP32
                    if (inplace_numerator) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
#endif
                    {
                        MaybeDeviceZoneScopedN(profiling_enabled, "QKT@V MM+Pack");
                        uint32_t v_index_offset = 0;
                        sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                            out_cb, out_cb);
                        // cb_qkt_im rows are laid out at KT_stride even when this kt_sub only consumes a
                        // narrower logical width. Keep unpack init on the physical stride; inner_dim below
                        // still limits how many V rows are multiplied.
                        mm_no_mop_reinit_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h, KT_stride);
                        configure_row_pack_width(out_cb, qktv_subblock_w);
                        for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
#ifdef SDPA_RECIPE_FP32
                            const uint32_t pv_inner_offset = kt_sub * matmul_inner;
                            const uint32_t qktv_in1_index = pv_inner_offset * vDHt + v_index_offset;
#else
                            const uint32_t qktv_in1_index = kt_sub * matmul_inner * vDHt + v_index_offset;
#endif
                            blocked_matmul_and_pack<false, vDHt, vDHt>(
                                cb_qkt_im,
                                cb_v_in,
                                out_cb,
#ifdef SDPA_RECIPE_FP32
                                qktv_in0_index_offset + pv_inner_offset,
#else
                                qktv_in0_index_offset + kt_sub * matmul_inner,
#endif
                                qktv_in1_index,
                                0,
#ifdef SDPA_RECIPE_FP32
                                v_subblock * qktv_subblock_w,
#else
                                v_subblock * qktv_subblock_w + group_pv_offset,
#endif
                                qktv_subblock_w,
                                qktv_h,
                                matmul_inner,
                                KT_stride,
                                /*skip_pack_configure=*/true);
#ifndef SDPA_RECIPE_FP32
                            UNPACK({
                                if (!is_first_iter && kt_sub == 0 && v_subblock == 0) {
                                    sdpa_identity_flags[q_num_subblocks - 1] = sdpa_scan_identity_maxima(
                                        prev.max, cur.max, q_num_subblocks - 1, qkt_subblock_h);
                                }
                            })
#endif
                            v_index_offset += qktv_subblock_w;
                        }
                        sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                    }

#ifdef SDPA_RECIPE_FP32
                    if (kt_sub > 0 || inplace_numerator) {
#else
                    if (kt_sub > 0) {
#endif
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    }
                }
            } else {
                // In-place latent-V full-Sk single pass: softmax the whole row, then one matmul chain
                // per output column over all active_Sk tiles (DST-accumulated, packed once per DST
                // group). Vs split-drain this drops the L1-acc and the per-kt_sub packs/barriers.
                // active_Sk == kt_num_full_subblocks * actual_sbw exactly, so one pass covers the row.
                for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {
                    sub_exp_block_bcast_cols<profiling_enabled, scale_fp32>(
                        cb_qkt_im,
                        cur.max,
                        cur.sum,
                        KT_stride,
                        q_num_subblocks - 1,
                        kt_sub * actual_sbw,
                        qkt_subblock_h,
                        actual_sbw);
                }
                if constexpr (q_num_subblocks == 1) {
                    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                }
                CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
                {
                    MaybeDeviceZoneScopedN(profiling_enabled, "QKT@V MM+Pack");
                    sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                        out_cb, out_cb);
                    mm_no_mop_reinit_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h, KT_stride);
                    inplace_v_matmul_pack_batched<vDHt, dst_size, qktv_h>(
                        cb_qkt_im,
                        cb_v_in,
                        out_cb,
                        qktv_in0_index_offset,
                        /*inner_dim=*/kt_num_full_subblocks * matmul_inner,
                        KT_stride);
                    sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                }
            }
            qktv_in0_index_offset += qktv_h * KT_stride;
            qktv_in0_wait_tiles += qktv_in0_row_tiles;
        }

        // Pack→unpack barrier between Phase 2's q_sub=0 drain and the main V-matmul loop.
        // The drain runs sub_exp in-place on cb_qkt_im at the last q_subblock's positions
        // (PACK writes); the upcoming V matmul (UNPACK reads) targets those same positions.
        // Without an explicit handshake, UNPACK can see stale L1 bytes — observed as wildly
        // wrong V matmul output (rmse > 1) on small-DHt + small-chunk causal shapes.
        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));

#ifdef SDPA_RECIPE_FP32
        // All P rows are committed by the drain barrier above. Use exactly the
        // effective HiFi2 weights in the denominator, without SFPU mantissa masking.
        {
            CircularBuffer(cb_col_identity).wait_front(1);
            sdpa_stream_reconfig(cb_col_identity, cb_qkt_im);
            matmul_block_init(cb_qkt_im, cb_col_identity, 0, 1, 4, KT_stride);
#ifdef SDPA_RECIPE_ACCURATE
            constexpr auto denom_fidelity = MathFidelity::HiFi2;
#else
            constexpr auto denom_fidelity = MathFidelity::LoFi;
#endif
            MATH((llk_math_matmul_init<denom_fidelity, MM_THROTTLE>(cb_qkt_im, cb_col_identity, 0, 1, 4)));
#ifdef SDPA_RECIPE_ACCURATE
            static_assert(denom_fidelity == MathFidelity::HiFi2, "Phase-0/2 denominator requires the two-phase MOP");
            // SrcA is exactly zero/one: its low mantissa is zero. Execute phases
            // 0 and 2, not ordinary HiFi2's 0 and 1, to retain all SrcB bits.
            MATH((addr_mod_t{
                .srca = {.incr = 0, .clr = 1, .cr = 1},
                .srcb = {.incr = 0, .clr = 1, .cr = 1},
                .dest = {.incr = 0, .clr = 1, .cr = 1},
                .fidelity = {.incr = 2, .clr = 0},
            }
                      .set(ADDR_MOD_5)));
#endif
            configure_single_tile_pack(cur.sum);
            PACK((llk_pack_reconfig_l1_acc(inplace_numerator ? 1 : 0)));
            for (uint32_t row = 0; row < Sq_chunk_t; row += 4) {
                tile_regs_acquire();
                for (uint32_t col = 0; col < active_Sk; ++col) {
                    UNPACK(
                        (llk_unpack_AB_matmul(cb_qkt_im, cb_col_identity, row * KT_stride + col, 0, 1, 4, KT_stride)));
                    MATH((llk_math_matmul<denom_fidelity, MM_THROTTLE>(0, 1, 4)));
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < 4; ++j) {
                    pack_tile<true>(j, cur.sum, row + j);
                }
                tile_regs_release();
            }
            MATH((llk_math_matmul_init_no_mop<MATH_FIDELITY, MM_THROTTLE>(
                cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h)));
        }

#endif
        // Per-row normalization lambda — fires on last K chunk (standard or deferred norm).
        // Takes sbh so it works for both full subblocks (qktv_h) and remainder (qktv_remainder_h).
        [[maybe_unused]] uint32_t sink_row_offset = 0;
        [[maybe_unused]] auto normalize_row = [&](uint32_t& pushed, uint32_t sbh) {
            MaybeDeviceZoneScopedN(profiling_enabled, "ROW_NORM");
#ifndef SDPA_RECIPE_FP32
            if (is_first_iter) {
                group2_bootstrap_row(prev.out, out_cb, pushed * sbh, 0);
            }
#endif
            CircularBuffer(cur.sum).push_back(sbh * sdpa_sum_stride);
            CircularBuffer(out_cb).push_back(sbh * vDHt * sdpa_out_stride);
            normalize_row_streaming<
                profiling_enabled,
                vDHt,
                dst_size,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                scale_fp32,
                use_attention_sink,
#ifdef SDPA_RECIPE_FP32
                cb_attention_sink>(cur.sum, out_cb, sbh, cur.max, sink_row_offset);
#else
                cb_attention_sink>(cur.sum, prev.out, sbh, cur.max, sink_row_offset);
            CircularBuffer(out_cb).pop_front(sbh * vDHt * sdpa_out_stride);
#endif
            if constexpr (use_attention_sink) {
                sink_row_offset += sbh;
            }
            pushed++;
        };

        bool identity_corrections[Sq_chunk_t] = {};

        // SALAD correction lambda — works for both full subblocks (sbh=qktv_h) and
        // remainder (sbh=qktv_remainder_h). Normalization is independently guarded at call sites.
        // prev.out is consumed row-by-row: always read from CB front, then pop after use.
        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, uint32_t sbh) {
#ifdef SDPA_RECIPE_FP32
            if (inplace_numerator) {
                CircularBuffer(cb_exp_max_diff).wait_front(sbh);
            }
            if (inplace_numerator) {
                // Preserve the existing correction CB publication/consumption
                // handshake; only its unused arithmetic disappears.
                CircularBuffer(cb_exp_max_diff).pop_front(sbh);
                PACK((llk_pack_reconfig_l1_acc(0)));
                return;
            }
#else
            // Global row-group index is stable even when final normalization pops rows.
            const bool has_local = (group_k_index % 2 == 0) && group_local_valid[salad_row];
#endif
            PACK((llk_pack_reconfig_l1_acc(1)));
            {
                MaybeDeviceZoneScopedN(profiling_enabled, "S_CORR_FUSED");
                // ob_q_subblock=0: prev.out and cb_exp_max_diff are popped row-by-row (read from front).
                // sum_q_subblock=salad_row: prev.sum uses cumulative indexing (not popped per-row).
#ifdef SDPA_RECIPE_FP32
                if (inplace_numerator) {
                    configure_single_tile_pack(cur.sum);
                    sdpa_fp32_state_update<1, true>(
                        prev.sum,
                        cur.sum,
                        cb_exp_max_diff,
                        salad_row,
                        is_last_iter ? w_salad : salad_row,
                        w_salad,
                        0,
                        true);
                } else
#endif
                    if constexpr (has_qktv_remainder) {
                    if (sbh == qktv_remainder_h) {
                        salad_correct_fused<qktv_remainder_h, vDHt, dst_size>(
                            prev.out,
                            prev.sum,
                            cb_exp_max_diff,
                            out_cb,
                            cur.sum,
                            0,
                            salad_row,
                            w_salad,
                            is_last_iter,
#ifdef SDPA_RECIPE_FP32
                            identity_corrections[salad_row]);
#else
                            identity_corrections[salad_row],
                            is_last_iter || (group_k_index % 2 == 0),
                            (group_k_index % 2 == 1),
                            has_local);
#endif
                    } else {
                        salad_correct_fused<qktv_h, vDHt, dst_size>(
                            prev.out,
                            prev.sum,
                            cb_exp_max_diff,
                            out_cb,
                            cur.sum,
                            0,
                            salad_row,
                            w_salad,
                            is_last_iter,
#ifdef SDPA_RECIPE_FP32
                            identity_corrections[salad_row]);
#else
                            identity_corrections[salad_row],
                            is_last_iter || (group_k_index % 2 == 0),
                            (group_k_index % 2 == 1),
                            has_local);
#endif
                    }
                } else {
                    salad_correct_fused<qktv_h, vDHt, dst_size>(
                        prev.out,
                        prev.sum,
                        cb_exp_max_diff,
                        out_cb,
                        cur.sum,
                        0,
                        salad_row,
                        w_salad,
                        is_last_iter,
#ifdef SDPA_RECIPE_FP32
                        identity_corrections[salad_row]);
#else
                        identity_corrections[salad_row],
                        is_last_iter || (group_k_index % 2 == 0),
                        (group_k_index % 2 == 1),
                        has_local);
#endif
                }
            }
#ifndef SDPA_RECIPE_FP32
            group_local_valid[salad_row] = (group_k_index % 2 == 1) && identity_corrections[salad_row] && !is_last_iter;
#endif
            CircularBuffer(cb_exp_max_diff).pop_front(sbh);
#ifdef SDPA_RECIPE_FP32
            if (!inplace_numerator)
                CircularBuffer(prev.out).pop_front(sbh * vDHt * sdpa_out_stride);
#else
            // Protected CB8 remains fronted until final normalization.
#endif
            PACK((llk_pack_reconfig_l1_acc(0)));
        };

        // q_subblock 1..N-1 (+ optional remainder): SALAD(prev) overlapped with matmul(cur)
        // When Sq_chunk_t is not divisible by qktv_h, the last iteration handles the
        // remainder row(s) with a smaller V matmul height.
        constexpr uint32_t total_v_row_groups = qktv_q_num_subblocks + (has_qktv_remainder ? 1 : 0);
        exp_packthread_tile_init<EXP_APPROX_MODE>();
        for (uint32_t q_subblock = 1; q_subblock < total_v_row_groups; ++q_subblock) {
            MaybeDeviceZoneScopedN(profiling_enabled, "Softmax(Q@KT)@V");
            const bool is_remainder_iter = has_qktv_remainder && (q_subblock == qktv_q_num_subblocks);
            const uint32_t cur_h = is_remainder_iter ? qktv_remainder_h : qktv_h;
            uint32_t salad_row = q_subblock - 1;
            uint32_t w_salad = salad_row - pushed_rows;
            // For remainder: convert group index to tile-row index so pack addressing is correct
            // when cur_h < qktv_h (the matmul uses cur_h as subblock_h, so row_subblock_idx * cur_h
            // must equal the actual tile-row offset).
            uint32_t w_q =
                is_remainder_iter ? (qktv_q_num_subblocks - pushed_rows) * qktv_h : (q_subblock - pushed_rows);

            // SALAD for previous group (always a full group, h=qktv_h)
            if (!is_first_iter) {
                CircularBuffer(cb_exp_max_diff).reserve_back(qktv_h);
#ifdef SDPA_RECIPE_FP32
                identity_corrections[salad_row] =
                    inplace_numerator || sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                                             prev.max, cur.max, cb_exp_max_diff, salad_row, qktv_h);
#else
                identity_corrections[salad_row] = sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                    prev.max, cur.max, cb_exp_max_diff, salad_row, qktv_h, sdpa_identity_flags[salad_row]);
#endif
                CircularBuffer(cb_exp_max_diff).push_back(qktv_h);
            }

            // V matmul for current row group — cur_h adapts for remainder
            if (is_remainder_iter) {
                CircularBuffer(cb_qkt_im).wait_front(Sq_chunk_t * KT_stride);
            } else {
                CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
            }
            {
#ifdef SDPA_RECIPE_FP32
                if (inplace_numerator) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
#endif
                MaybeDeviceZoneScopedN(profiling_enabled, "QKT@V MM+Pack");
                uint32_t v_index_offset = 0;
                sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                    out_cb, out_cb);
                // See the q_subblock-0 V matmul above: active_Sk can be narrower than the physical
                // cb_qkt_im row stride, but the unpacker is configured for the physical layout.
                mm_no_mop_reinit_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, cur_h, KT_stride);
                // Configure once before v_subblock loop; skip inside.
                configure_row_pack_width(out_cb, qktv_subblock_w);
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    // Same in-place-vs-materialized V addressing as the q_subblock-0 drain above.
                    // kt_inplace_v is constexpr-true only when Sq_chunk_t == 1, which yields a
                    // single row-group — so this outer q_subblock loop is skipped entirely on that
                    // path and kt_inplace_v is effectively false here. The ternary is kept symmetric
                    // with the q_subblock-0 site so the addressing forms stay paired.
                    const uint32_t qktv_in1_index = kt_inplace_v ? (v_subblock * KT_stride) : v_index_offset;
                    blocked_matmul_and_pack<false, kt_inplace_v ? 1 : vDHt, vDHt>(
                        cb_qkt_im,
                        cb_v_in,
                        out_cb,
                        qktv_in0_index_offset,
                        qktv_in1_index,
                        w_q,
#ifdef SDPA_RECIPE_FP32
                        v_subblock * qktv_subblock_w,
#else
                        v_subblock * qktv_subblock_w + group_pv_offset,
#endif
                        qktv_subblock_w,
                        cur_h,
                        active_Sk,
                        KT_stride,
#ifdef SDPA_RECIPE_FP32
                        /*skip_pack_configure=*/true);
#else
                        /*skip_pack_configure=*/true);
#endif
                    v_index_offset += qktv_subblock_w;
                }
                sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
            }

            // SALAD corrections for previous group (always full, h=qktv_h) + row-by-row push
            if (!is_first_iter) {
                // Last main-loop iteration: hoist drain's sub_exp so both salads
                // (current row and drain row) chain back-to-back with one FPU init.
                if (q_subblock == total_v_row_groups - 1) {
                    constexpr uint32_t drain_h = has_qktv_remainder ? qktv_remainder_h : qktv_h;
                    const uint32_t drain_salad_row =
                        has_qktv_remainder ? (qktv_q_num_subblocks * qktv_h) : (qktv_q_num_subblocks - 1);

                    CircularBuffer(cb_exp_max_diff).reserve_back(drain_h);
#ifdef SDPA_RECIPE_FP32
                    identity_corrections[drain_salad_row] =
                        inplace_numerator || sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                                                 prev.max, cur.max, cb_exp_max_diff, drain_salad_row, drain_h);
#else
                    identity_corrections[drain_salad_row] = sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                        prev.max,
                        cur.max,
                        cb_exp_max_diff,
                        drain_salad_row,
                        drain_h,
                        sdpa_identity_flags[drain_salad_row]);
#endif
                    CircularBuffer(cb_exp_max_diff).push_back(drain_h);

                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(qktv_h * sdpa_sum_stride);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }

                    const uint32_t drain_w = has_qktv_remainder ? ((qktv_q_num_subblocks - pushed_rows) * qktv_h)
                                                                : (qktv_q_num_subblocks - 1 - pushed_rows);
                    salad_correct_row(drain_salad_row, drain_w, drain_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(drain_h * sdpa_sum_stride);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                } else {
                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(qktv_h * sdpa_sum_stride);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                }
            } else if (is_last_iter) {
                normalize_row(pushed_rows, qktv_h);
            } else {
#ifndef SDPA_RECIPE_FP32
                group2_bootstrap_row(prev.out, out_cb, salad_row * qktv_h, salad_row * qktv_h);
#endif
                CircularBuffer(cur.sum).push_back(qktv_h * sdpa_sum_stride);
                CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                pushed_rows++;
            }

            qktv_in0_index_offset += cur_h * KT_stride;
            qktv_in0_wait_tiles += cur_h * KT_stride;
        }

        // Pipeline drain: SALAD for the last group
        {
            constexpr uint32_t drain_h = has_qktv_remainder ? qktv_remainder_h : qktv_h;
            if constexpr (total_v_row_groups == 1) {
                // Single row group: the main loop never ran, so the drain must
                // perform the full SALAD correction (sub_exp + correct) here.
                if (!is_first_iter) {
                    constexpr uint32_t drain_salad_row = 0;
                    CircularBuffer(cb_exp_max_diff).reserve_back(drain_h);
                    identity_corrections[drain_salad_row] = sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
#ifdef SDPA_RECIPE_FP32
                        prev.max, cur.max, cb_exp_max_diff, drain_salad_row, drain_h);
#else
                        prev.max,
                        cur.max,
                        cb_exp_max_diff,
                        drain_salad_row,
                        drain_h,
                        sdpa_identity_flags[drain_salad_row]);
#endif
                    CircularBuffer(cb_exp_max_diff).push_back(drain_h);
                    salad_correct_row(drain_salad_row, 0, drain_h);
                }
                if (is_last_iter) {
                    normalize_row(pushed_rows, drain_h);
                } else {
                    CircularBuffer(cur.sum).push_back(drain_h * sdpa_sum_stride);
                    CircularBuffer(out_cb).push_back(drain_h * vDHt * sdpa_out_stride);
                    pushed_rows++;
                }
            } else {
                // Drain was hoisted into the last main-loop iteration above.
                // For is_first_iter (no SALAD), the drain row still needs push/normalize.
                if (is_first_iter) {
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
#ifndef SDPA_RECIPE_FP32
                        group2_bootstrap_row(
                            prev.out, out_cb, (total_v_row_groups - 1) * qktv_h, (total_v_row_groups - 1) * qktv_h);
#endif
                        CircularBuffer(cur.sum).push_back(drain_h * sdpa_sum_stride);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                }
            }
        }

        // All rows pushed individually — no bulk push needed.

        if constexpr (use_attention_sink) {
            if (is_last_iter) {
                CircularBuffer(cb_attention_sink).pop_front(1);
            }
        }

        // For kt_inplace_v this is the deferred K^T pop: cb_v_in aliases cb_kt_in (v_shares_k_buffer),
        // and v_cb_physical_width_t == DHt, so this pops the same Sk_chunk_t*DHt entry that Phase 1
        // skipped. For the materialized path it pops the V entry as usual. Either way: one entry/chunk.
        CircularBuffer(cb_v_in).pop_front(KT_stride * v_cb_physical_width_t);
        CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT_stride);
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
 * @tparam qkt_subblock_h  QK matmul subblock height (rows processed per DST acquire/release cycle)
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
    uint32_t qkt_subblock_h,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_h,
    uint32_t qktv_subblock_w,
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
    uint32_t sliding_window_size = 0,
    bool is_causal_sdpa = false,
    bool use_attention_sink = false,
    uint32_t cb_attention_sink = INVALID_CB,
    bool use_provided_mask = false,
    bool use_windowed_narrowing = false,
    uint32_t cb_windowed_k_range = INVALID_CB>
void sdpa_standard_v2(
    const uint32_t q_chunks_per_core,
    const uint32_t k_num_chunks,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t local_q_start = 0,
    const uint32_t chunked_q_chunk_offset = 0,
    const LightweightMaskContext& lw_mask = {},
    const uint32_t q_num_chunks = 0,
    const bool use_zigzag_balancing = false) {
    init_sdpa_streaming_semaphores();

    // use_padded_mask + is_causal_sdpa is handled at the host level (mutually exclusive).
    static_assert(
        !(use_padded_mask && is_causal_sdpa), "use_padded_mask and is_causal_sdpa are mutually exclusive in v2");
    // K-loop bound geometry shared with the reader (see sliding_window_geometry.hpp).
    using window_geom = SlidingWindowLoopGeometry<sliding_window_size, is_causal_sdpa, TILE_HEIGHT>;
    constexpr bool has_sliding_window = window_geom::has_sliding_window;
    constexpr uint32_t left_window_tiles = window_geom::left_window_tiles;
    constexpr uint32_t right_window_tiles = window_geom::right_window_tiles;

    // v1 dense-mask scope: a user-provided mask supplies all masking itself (causal/sliding/padding
    // are baked into the tensor by the caller, and the reader neginf-fills padded positions), so it
    // is never combined with the structured stamp paths.
    static_assert(
        !(use_provided_mask && (is_causal_sdpa || has_sliding_window)),
        "use_provided_mask is mutually exclusive with causal/sliding stamping in v2");

    // Neginf tile is permanently fronted by the writer — wait once before any K-chunk loop.
    // Skipped for a user-provided mask: the writer generates no palette; the reader streams the
    // dense mask (with padded positions neginf-filled) per chunk instead.
    constexpr uint32_t padded_k_tiles_inner = (Sk_chunk_t - (Skt % Sk_chunk_t)) % Sk_chunk_t;
    if constexpr (use_padded_mask && padded_k_tiles_inner > 0 && !use_provided_mask) {
        CircularBuffer(cb_mask_in).wait_front(1);
    }

    constexpr uint32_t last_chunk_Sk = Sk_chunk_t - padded_k_tiles_inner;

    for (uint32_t q = 0; q < q_chunks_per_core; q++) {
        AccumulatorHalf prev = {cb_sum_A, cb_max_A, cb_out_im_A};
        AccumulatorHalf cur = {cb_sum_B, cb_max_B, cb_out_im_B};
#ifndef SDPA_RECIPE_FP32
        static_assert(Sq_chunk_t == 8 && vDHt == 4 && !is_causal_sdpa, "Compensated two-chunk noncausal geometry");
        group2_initialize_root(cb_out_im_A);
        uint32_t active_group_index = 0;
        bool group_local_valid[4] = {};  // Fresh per-Q, identical on all three RISCs.
#endif

        // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
        // The unpack MOP is split in half (block_ct_dim / 2), so active_Sk must be even,
        // and we need >1 subblock so the semaphore fires before the reduce's second half.
        constexpr bool can_reduce_trigger = reduce_trigger_supported && (Sk_chunk_t % qkt_subblock_w == 0) &&
                                            (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

        // Pre-compute subblock width: compile-time for full chunks, hoisted for padded last chunk.
        constexpr uint32_t full_sbw = qkt_subblock_w;
        constexpr uint32_t padded_sbw = (last_chunk_Sk < Sk_chunk_t && last_chunk_Sk % qkt_subblock_w != 0)
                                            ? largest_factor_le(last_chunk_Sk, qkt_subblock_w)
                                            : qkt_subblock_w;

        // With largest_factor_le, padded chunks also have evenly-dividing subblocks,
        // so reduce_trigger can be enabled when the same constraints hold for last_chunk_Sk.
        constexpr bool can_reduce_trigger_padded = reduce_trigger_supported && (padded_k_tiles_inner > 0) &&
                                                   (last_chunk_Sk % padded_sbw == 0) &&
                                                   (last_chunk_Sk / padded_sbw > 1) && (last_chunk_Sk % 2 == 0);

        // Optional zigzag Q-chunk remap plus per-Q K-chunk bounds. Causal uses the
        // diagonal upper bound; sliding-window adds a lower bound and, for non-causal
        // centered windows, an upper bound around the Q chunk.
        uint32_t q_chunk_local = local_q_start + q;
        uint32_t q_start_tile = 0;
        uint32_t k_loop_start = 0;
        uint32_t k_loop_end = k_num_chunks;
        if constexpr (is_causal_sdpa || has_sliding_window) {
            // Reader and writer apply the same remap; compute must agree or causal
            // masks and output positions desync. The mod is a no-op when the input is per-head
            // ([0, q_num_chunks)) and extracts the per-head q_chunk when it's a flat global index
            // (global Q scheduling iterates across batches and heads).
            q_chunk_local = remap_q_index(q_chunk_local, q_num_chunks, use_zigzag_balancing) % q_num_chunks;
            // q_chunk_global is the absolute Q chunk index (used for the diagonal);
            // chunked-prefill shifts this via chunked_q_chunk_offset.
            const uint32_t q_chunk_global = q_chunk_local + chunked_q_chunk_offset;
            q_start_tile = q_chunk_global * Sq_chunk_t;
            if constexpr (is_causal_sdpa) {
                const uint32_t limit = (q_start_tile + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t;
                k_loop_end = limit < k_num_chunks ? limit : k_num_chunks;
            }
            if constexpr (has_sliding_window) {
                if (q_start_tile > left_window_tiles) {
                    k_loop_start = (q_start_tile - left_window_tiles) / Sk_chunk_t;
                }
                if constexpr (!is_causal_sdpa) {
                    const uint32_t limit =
                        (q_start_tile + Sq_chunk_t + right_window_tiles + Sk_chunk_t - 1) / Sk_chunk_t;
                    k_loop_end = limit < k_num_chunks ? limit : k_num_chunks;
                }
            }
        }
        // Windowed K-range narrowing: this Q chunk's [k_lo, k_hi) comes from the reader's ctrl CB —
        // read via the UNPACK mailbox so all three TRISCs agree. The reader streams exactly this many
        // K/V chunks and the writer produces exactly this many mask chunks; disagreement deadlocks.
        if constexpr (use_windowed_narrowing) {
            CircularBuffer cb_k_range_obj(cb_windowed_k_range);
            cb_k_range_obj.wait_front(1);
            k_loop_start = ckernel::read_tile_value(cb_windowed_k_range, 0, 0);
            k_loop_end = ckernel::read_tile_value(cb_windowed_k_range, 0, 1);
            cb_k_range_obj.pop_front(1);
        }

        auto call_step = [&](auto profiling_tag,
                             bool is_last,
                             bool is_first,
                             uint32_t active_Sk,
                             bool reduce_trigger,
                             uint32_t sbw,
                             bool apply_causal,
                             uint32_t k_start_tile,
                             bool apply_mask,
                             uint32_t lw_partial_tile_idx,
                             bool apply_sliding_window) {
            sdpa_inner_loop_step<
                decltype(profiling_tag)::value,
                Sq_chunk_t,
                Sk_chunk_t,
                Skt,
                DHt,
                vDHt,
                scale_fp32,
                qkt_subblock_h,
                qkt_subblock_w,
                qktv_subblock_h,
                qktv_subblock_w,
                use_padded_mask,
                false,  // ring_mode
                is_causal_sdpa,
                cb_q_in,
                cb_kt_in,
                cb_v_in,
                cb_qkt_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                cb_mask_in,
                Sk_chunk_t,
                false,
                0,
                0,
                0,
                vDHt,
                false,
                sliding_window_size,
                use_attention_sink,
                cb_attention_sink,
                use_provided_mask>(
                prev,
                cur,
                is_last,
                is_first,
                apply_mask,
                lw_partial_tile_idx,
                active_Sk,
                reduce_trigger,
                sbw,
                INVALID_CB,  // save_out_cb
                INVALID_CB,  // save_max_cb
                apply_causal,
                q_start_tile,
                k_start_tile,
                lw_mask.neginf_tile_idx,
                lw_mask.primary_diag_tile_idx,
                lw_mask.sliding_leading_prev_tile_idx,
                lw_mask.sliding_leading_tile_idx,
                lw_mask.sliding_trailing_next_tile_idx,
#ifdef SDPA_RECIPE_FP32
                apply_sliding_window);
#else
                apply_sliding_window,
                0,
                0,
                {},
                0,
                active_group_index,
                group_local_valid);
#endif
        };

        for (uint32_t k_chunk = k_loop_start; k_chunk < k_loop_end; k_chunk++) {
            bool is_first = (k_chunk == k_loop_start);
            bool is_last = (k_chunk == k_loop_end - 1);

            // Padded path is non-causal only (use_padded_mask && is_causal_sdpa rejected by static_assert).
            // With sliding-window loop narrowing, the loop's last chunk may not be the tensor's final K chunk.
            // A user-provided mask processes the full Sk_chunk_t (padded cols are neginf in the dense
            // mask), so it never takes the partial-tile narrowing.
            const bool is_padded =
                !is_causal_sdpa && !use_provided_mask && (k_chunk == k_num_chunks - 1) && (padded_k_tiles_inner > 0);
            uint32_t chunk_active_Sk = is_padded ? last_chunk_Sk : Sk_chunk_t;
            bool chunk_reduce_trigger = is_padded ? can_reduce_trigger_padded : can_reduce_trigger;
            uint32_t chunk_sbw = is_padded ? padded_sbw : full_sbw;

            // Last-chunk narrowing: causal and non-causal partial-tile K are mutually exclusive
            // (static_assert at function entry), but both shrink chunk_active_Sk on is_last to
            // skip cols past the diag (causal) or past Sk's last partial tile (non-causal partial).
            //
            // Mask side:
            // - Causal: per-row diagonal/trailing-neginf stamp via apply_lightweight_mask_streaming
            //   (no-stamp / partial-stamp / full-neginf decided per row by diag_col).
            // - Non-causal partial: trailing fully-padded tiles → neginf via num_padded; partial
            //   boundary tile → vertical-bar mask tile via apply_mask + lw_partial_tile_idx.
            bool apply_partial_mask = false;
            bool apply_sliding_mask = false;
            bool apply_causal_mask = is_causal_sdpa;
            uint32_t target_active_Sk = chunk_active_Sk;
            if constexpr (use_padded_mask && !is_causal_sdpa && !use_provided_mask) {
                // Stamp the partial-tile vertical mask whenever the tensor's last K chunk carries a
                // partial tile (Sk % TILE != 0), independent of fully padded tiles. When there are no
                // fully padded tiles, global_n_padded_tiles is 0, so target_active_Sk remains
                // Sk_chunk_t and only the vertical mask stamp is added.
                if ((k_chunk == k_num_chunks - 1) && lw_mask.global_n_partial_col > 0) {
                    target_active_Sk = Sk_chunk_t - lw_mask.global_n_padded_tiles;
                    apply_partial_mask = true;
                }
            }
            if constexpr (has_sliding_window) {
                const uint32_t k_start_tile = k_chunk * Sk_chunk_t;
                const uint32_t trailing_range_end =
                    is_causal_sdpa ? (q_start_tile + Sq_chunk_t) : (q_start_tile + Sq_chunk_t + right_window_tiles);
                apply_sliding_mask = true;
                if constexpr (is_causal_sdpa) {
                    const uint32_t k_end_tile = k_start_tile + Sk_chunk_t;
                    apply_causal_mask = (q_start_tile < k_end_tile) && ((q_start_tile + Sq_chunk_t) > k_start_tile);
                }
                if (is_last && trailing_range_end > k_start_tile) {
                    const uint32_t window_active_Sk = trailing_range_end - k_start_tile;
                    target_active_Sk = target_active_Sk < window_active_Sk ? target_active_Sk : window_active_Sk;
                }
            } else if constexpr (is_causal_sdpa) {
                if (is_last) {
                    target_active_Sk = q_start_tile + Sq_chunk_t - k_chunk * Sk_chunk_t;
                }
            }
            if (target_active_Sk < chunk_active_Sk) {
                chunk_active_Sk = target_active_Sk;
                chunk_sbw = largest_factor_le(chunk_active_Sk, qkt_subblock_w);
                // reduce_trigger relies on active_Sk == Sk_chunk_t for the unpack MOP split.
                chunk_reduce_trigger = false;
            }

#ifndef SDPA_RECIPE_FP32
            active_group_index = k_chunk - k_loop_start;
#endif
            call_step(
                std::false_type{},
                is_last,
                is_first,
                chunk_active_Sk,
                chunk_reduce_trigger,
                chunk_sbw,
                apply_causal_mask,
                k_chunk * Sk_chunk_t,
                apply_partial_mask,
                apply_partial_mask ? lw_mask.global_n_partial_tile_idx : 0u,
                apply_sliding_mask);

#ifdef SDPA_RECIPE_FP32
            // Post-iteration cleanup
            // prev.out and cb_exp_max_diff are already popped row-by-row inside salad_correct_row.
#else
            // Recycle the scratch allocation; its local plane survives in L1.
            if (!is_last) {
                sdpa_cb_pop_front_out_of_line(cur.out, Sq_chunk_t * vDHt * sdpa_out_stride);
            }
            // Protected output stays fixed; only max and denominator ping-pong.
#endif
            if (!is_first) {
                sdpa_cb_pop_front_out_of_line(prev.max, Sq_chunk_t);
#ifdef SDPA_RECIPE_FP32
                if (!sdpa_skip_prev_sum_pop)
#endif
                    sdpa_cb_pop_front_out_of_line(prev.sum, Sq_chunk_t * sdpa_sum_stride);
            }

            if (is_last) {
                sdpa_cb_pop_front_out_of_line(cur.max, Sq_chunk_t);
            } else {
                std::swap(prev, cur);
#ifndef SDPA_RECIPE_FP32
                prev.out = cb_out_im_A;
                cur.out = cb_out_im_B;
#endif
            }
        }
        // Q already popped inside sdpa_inner_loop_step after Phase 1 of the last K chunk.
    }
}
