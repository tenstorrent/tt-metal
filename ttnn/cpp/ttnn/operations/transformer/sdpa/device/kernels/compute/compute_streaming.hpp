// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streaming SDPA compute helpers.
// Included only by sdpa.cpp when use_streaming_compute is true.
// Depends on primitives from compute_common.hpp (must be included first).

#pragma once

#include <type_traits>

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#endif
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
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

// --- Outlined out-of-order pack (code-size) ---
// pack_tile<true>() (absolute-address pack) inlines the full
// llk_pack -> program_packer_destination GPR->FLOP address-programming sequence at every
// call site. That inlined sequence is the single largest contributor to the PACK-thread
// (TRISC2) text and overflows the TENSIX kernel-config buffer under watcher on Wormhole.
// Outlining to one noinline copy trades a jal/ret per pack for a large code-size reduction.
//
// The jal/ret is NOT free on the hot inner loop. Empirically (wan2_2 BH perf check), the
// softmax-exp pack in sub_exp_block_bcast_cols is the only perf-critical site: outlining it
// alone caused the full regression (~70% -> ~66% math util), and re-inlining only it fully
// recovers perf. So we keep that one site inlined via pack_tile<true>() directly and outline
// everything else (output/SV drain, salad correction/sum, mask L1-accumulate) through this
// wrapper. That keeps almost all of the Wormhole code-size win (only the exp pack's ~few
// static copies return) with no measurable BH perf cost. On MATH/UNPACK threads pack_tile is
// a no-op, so the outlined wrapper collapses to an empty inline function (zero overhead).
#ifdef TRISC_PACK
__attribute__((noinline, noclone)) static void sdpa_pack_tile_ooo(uint32_t dst, uint32_t cb, uint32_t idx) {
    llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(dst, cb, idx);
}
#else
ALWI void sdpa_pack_tile_ooo(uint32_t, uint32_t, uint32_t) {}
#endif

static __attribute__((noinline, noclone)) void sdpa_cb_push_back_out_of_line(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).push_back(num_tiles);
}

static __attribute__((noinline, noclone)) void sdpa_cb_pop_front_out_of_line(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).pop_front(num_tiles);
}

/**
 * Push tiles to make them visible for UNPACK reads, but rewind wr_ptr so
 * subsequent pack_tile<true> offsets remain relative to a stable base.
 * This eliminates the need for separate row buffers.
 */
ALWI void cb_push_back_hold_wr_ptr(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).push_back(num_tiles);
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

// Ring-streaming lightweight-mask context. Field NAMES match LightweightMaskContext so sdpa_ring_v2's
// generic `lw_mask.<field>` reads work for either type, BUT the 10 compile-time-constant fields are
// template params (static constexpr → zero per-instance storage), leaving only the 3 per-ring-iter
// runtime fields on the stack. Shrinks the live lw_mask object ~56 B → ~12 B on kernel_main's frame.
// Only the ring_joint_sdpa producer uses it; exp_ring keeps the plain LightweightMaskContext —
// sdpa_ring_v2's MaskCtx template param is deduced per caller.
template <
    uint32_t NeginfTileIdx,
    uint32_t CausalDiagTileIdx,
    uint32_t LocalNPaddedTiles,
    uint32_t JointNPaddedTiles,
    uint32_t GlobalNPartialCol,
    uint32_t JointLPartialCol,
    uint32_t GlobalNPartialTileIdx,
    uint32_t JointLPartialTileIdx,
    uint32_t StraddleMaskChunkId>
struct RingStreamingMaskCtx {
    // Per-ring-iter runtime fields (the only ones needing stack storage):
    bool is_causal = false;
    uint32_t global_n_padded_tiles = 0;
    uint32_t straddle_num_padded_tiles = 0;
    // Compile-time-constant fields (no per-instance storage):
    static constexpr uint32_t neginf_tile_idx = NeginfTileIdx;
    static constexpr uint32_t causal_diag_tile_idx = CausalDiagTileIdx;
    static constexpr uint32_t primary_diag_tile_idx = CausalDiagTileIdx;
    static constexpr uint32_t local_n_padded_tiles = LocalNPaddedTiles;
    static constexpr uint32_t joint_n_padded_tiles = JointNPaddedTiles;
    static constexpr uint32_t global_n_partial_col = GlobalNPartialCol;
    static constexpr uint32_t joint_l_partial_col = JointLPartialCol;
    static constexpr uint32_t global_n_partial_tile_idx = GlobalNPartialTileIdx;
    static constexpr uint32_t joint_l_partial_tile_idx = JointLPartialTileIdx;
    static constexpr uint32_t straddle_mask_chunk_id = StraddleMaskChunkId;

    // Materialize a full LightweightMaskContext. Only used on the if-constexpr-discarded v1
    // (sdpa_ring) path so it type-checks; DCE'd on the streaming v2 path (which deduces this type
    // directly and keeps the compact frame).
    operator LightweightMaskContext() const {
        LightweightMaskContext m;
        m.is_causal = is_causal;
        m.neginf_tile_idx = neginf_tile_idx;
        m.causal_diag_tile_idx = causal_diag_tile_idx;
        m.primary_diag_tile_idx = primary_diag_tile_idx;
        m.global_n_padded_tiles = global_n_padded_tiles;
        m.local_n_padded_tiles = local_n_padded_tiles;
        m.joint_n_padded_tiles = joint_n_padded_tiles;
        m.global_n_partial_col = global_n_partial_col;
        m.joint_l_partial_col = joint_l_partial_col;
        m.global_n_partial_tile_idx = global_n_partial_tile_idx;
        m.joint_l_partial_tile_idx = joint_l_partial_tile_idx;
        m.straddle_num_padded_tiles = straddle_num_padded_tiles;
        m.straddle_mask_chunk_id = straddle_mask_chunk_id;
        return m;
    }
};

// Sentinel for "no CB" — beyond the valid 0-31 range.
constexpr uint32_t INVALID_CB = 32;
// BH benefits from blocked pack at width 4; WH keeps the threshold at 8 because
// width-4 blocked-pack reconfiguration costs more than it saves there.
#ifdef ARCH_BLACKHOLE
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 4;
#else
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 8;
#endif
ALWI bool should_use_blocked_pack_width(uint32_t pack_width) { return pack_width >= MIN_BLOCKED_PACK_TILES; }

template <uint32_t old_cb, uint32_t new_cb>
ALWI void sdpa_maybe_pack_reconfig_data_format() {
#ifdef TRISC_PACK
    if constexpr (pack_dst_format[old_cb] != pack_dst_format[new_cb]) {
        pack_reconfig_data_format(old_cb, new_cb);
    }
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
        reconfig_data_format(srca_old_cb, srca_new_cb, srcb_old_cb, srcb_new_cb);
    }
#endif
}

template <uint32_t srca_old_cb, uint32_t srca_new_cb, uint32_t srcb_old_cb, uint32_t srcb_new_cb>
ALWI void sdpa_maybe_reconfig_data_format(uint32_t runtime_srca_old_cb, uint32_t runtime_srcb_old_cb) {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    if constexpr (
        sdpa_unpack_format_changed<srca_old_cb, srca_new_cb>() ||
        sdpa_unpack_format_changed<srcb_old_cb, srcb_new_cb>()) {
        reconfig_data_format(runtime_srca_old_cb, srca_new_cb, runtime_srcb_old_cb, srcb_new_cb);
    }
#endif
}

// Keep this out-of-line even on BH: repeated pack-width configuration sites
// inflate SDPA streaming code size more than this call costs in measured cases.
static __attribute__((noinline, noclone)) void configure_pack_width(uint32_t cb, uint32_t pack_width) {
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

#if defined(TRISC_MATH) && defined(ARCH_WORMHOLE)
ALWI void recip_tile_first_column_wh_idst0_direct() {
    // WH SDPA normalize always operates on DST tile 0. The generic unary-SFPU
    // launcher around recip_tile_first_column() records a larger replay program
    // than this path needs. Keep the first-column reciprocal math identical,
    // but inline only the VectorMode::C traversal so the replay stays within
    // the SFPU replay slots and does not overwrite the no-mop matmul region.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    math::set_addr_mod_base();

#pragma GCC unroll 0
    for (int face = 0; face < 2; face++) {
        ckernel::sfpu::calculate_recip_first_column();
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}
#endif

// Raw pack: caller must have already called configure_row_pack_width(out_cb, pack_width).
// Use this in tight loops after configuring once at the loop boundary.
ALWI void pack_contiguous_rows_nocfg(
    uint32_t out_cb,
    uint32_t row_base,
    uint32_t row_count,
    uint32_t row_stride,
    uint32_t col_base,
    uint32_t pack_width) {
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
#if defined(ARCH_WORMHOLE)
__attribute__((noinline))
#endif
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
    bool skip_pack_configure = false) {
    tile_regs_acquire();
    uint32_t dst_index = 0;
    uint32_t in0_index = in0_index_start;
    uint32_t in1_index = in1_index_start;
    for (uint32_t inner = 0; inner < inner_dim; ++inner) {
        matmul_block_no_mop(
            in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, matmul_stride);
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

    reduce_block_max_row_init_runtime(out_cb, reduce_cols, respect_trigger);
    for (uint32_t i = 0; i < group_size; i++) {
        const uint32_t input_tile_start = (row_start + i) * row_stride;
        reduce_block_max_row_runtime(in0_cb, scale_cb, input_tile_start, i, respect_trigger, overlap_first_half);
    }
    reduce_block_max_row_uninit_runtime(in0_cb, respect_trigger, overlap_first_half);

    tile_regs_commit();
    tile_regs_wait();

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

    {
        MaybeDeviceZoneScopedN(profiling_enabled, "SUB_EXP_BLOCK_INIT");
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
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index++, vector_mode_exp);
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
                    pack_tile<true>(dst_index++, reduce_cb, max_row_base + i);  // HOT: softmax exp, keep inline
                    if (global_col_base == 0 && j == 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
        }
    }

    tile_regs_release();

    // Restore packer ReLU config after all exp operations complete
    PACK((llk_pack_relu_config(ReluConfig::none())));
    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Column-only exp(prev_max - cur_max) for SALAD corrections.
 * Operates on first-column subset of tiles.
 */
template <bool profiling_enabled, uint32_t scale_fp32>
void sub_exp_first_col_blocks(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t sbh) {
    const uint32_t tiles_per_row = sbh;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    sub_tiles_init(in0_cb, in1_cb);

    CircularBuffer(in0_cb).wait_front((q_subblock + 1) * tiles_per_row);
    CircularBuffer(in1_cb).wait_front((q_subblock + 1) * tiles_per_row);

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

        configure_single_tile_pack(out_cb);
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            pack_tile<false>(i, out_cb);
        }

        tile_regs_release();
    }
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
template <uint32_t sbh_t, uint32_t sbw_t, uint32_t dst_size>
void salad_correct_fused(
    uint32_t out_in_cb,
    uint32_t sum_in_cb,
    uint32_t bcast_cb,
    uint32_t out_out_cb,
    uint32_t sum_out_cb,
    uint32_t ob_q_subblock,
    uint32_t sum_q_subblock,
    uint32_t write_q_subblock) {
    constexpr uint32_t tiles_per_row = sbh_t;
    constexpr uint32_t tiles_per_column = sbw_t;
    constexpr uint32_t col_batch = (dst_size / sbh_t < sbw_t) ? dst_size / sbh_t : sbw_t;
    constexpr uint32_t last_out_cols = (sbw_t % col_batch == 0) ? col_batch : (sbw_t % col_batch);
    constexpr bool can_fuse_last = (last_out_cols * sbh_t + sbh_t <= dst_size);

    // out_in_cb and bcast_cb may be popped row-by-row (ob_q_subblock=0) while
    // sum_in_cb retains cumulative indexing (sum_q_subblock=salad_row).
    const uint32_t ob_row_base = ob_q_subblock * tiles_per_row;
    const uint32_t sum_row_base = sum_q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;

    mul_bcast_cols_init_short(out_in_cb, bcast_cb);

    CircularBuffer(out_in_cb).wait_front((ob_q_subblock + 1) * tiles_per_row * tiles_per_column);
    CircularBuffer(sum_in_cb).wait_front((sum_q_subblock + 1) * tiles_per_row);
    CircularBuffer(bcast_cb).wait_front((ob_q_subblock + 1) * tiles_per_row);

    constexpr uint32_t last_batch_rem = tiles_per_column % col_batch;
    for (uint32_t col_base = 0; col_base < tiles_per_column; col_base += col_batch) {
        const uint32_t cur_cols =
            (col_base + col_batch <= tiles_per_column) ? col_batch : (last_batch_rem > 0 ? last_batch_rem : col_batch);
        const bool is_last_out_batch = (col_base + cur_cols >= tiles_per_column);
        const bool fuse_sum_here = can_fuse_last && is_last_out_batch;

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
    }

    if constexpr (!can_fuse_last) {
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
    }
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
    // Attention sink: cb_attention_sink holds raw per-head sink logits (column 0, zeros elsewhere).
    // exp((sink - max)*scale) is computed inline per-row via sub_bcast_cols+exp, then folded into
    // the col-reduced denominator (DST[0]). sbh tiles consumed per call, popped once at end;
    // across all normalize calls for the chunk this consumes exactly Sq_chunk_t tiles.
    if constexpr (use_attention_sink) {
        CircularBuffer(cb_attention_sink).wait_front(sbh);
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
            CircularBuffer(cur_sum_cb).wait_front(1);

            CircularBuffer(scratch_cb).reserve_back(1);
            tile_regs_acquire();
            matmul_block(cur_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
            if constexpr (use_attention_sink) {
                // DST[1] = exp((sink[s] - max[row_offset+s]) * scale); DST[0] += DST[1].
                sub_bcast_cols_init_short(cb_attention_sink, cur_max_cb_rt);
                sub_tiles_bcast_cols(cb_attention_sink, cur_max_cb_rt, s, sink_row_offset + s, 1);
                // The custom first-column exp needs generic unary SFPU addrmod state, but not the
                // Blackhole approximate exp_init macro/replay setup used by exp_tile<true>.
                MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>()));
                constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
                MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(1)));
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
#ifdef ARCH_BLACKHOLE
            recip_tile_init<false>();
            MATH((recip_tile<false>(0 /*dst_index*/, VectorMode::C)));
#else
            recip_tile_init();
            MATH((recip_tile_first_column_wh_idst0_direct()));
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, scratch_cb);
            tile_regs_release();
            CircularBuffer(scratch_cb).push_back(1);

            CircularBuffer(cur_sum_cb).pop_front(1);
        }

        // 3. Normalize: multiply output tiles by bcast_cols(1/sum)
        // Process in batches of up to dst_size tiles (DST capacity).
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "NORM_MUL_BCAST");
            constexpr uint32_t batch = (head_dim_t_ < dst_size) ? head_dim_t_ : dst_size;
            mul_bcast_cols_init_short(cur_out_cb, scratch_cb);
            // Pack output to normalized_out_cb; old/new skips when it has the same format as scratch.
            sdpa_maybe_pack_reconfig_data_format<scratch_cb, normalized_out_cb>();
            CircularBuffer(cur_out_cb).wait_front(head_dim_t_);
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
            CircularBuffer(cur_out_cb).pop_front(head_dim_t_);
        }
    }
    if constexpr (use_attention_sink) {
        CircularBuffer(cb_attention_sink).pop_front(sbh);
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
 * Caller must set up copy_tile_to_dst_init_short and llk_pack_reconfig_l1_acc(1) before calling,
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
    // before entry via copy_tile_to_dst_init_short + llk_pack_reconfig_l1_acc(1).
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
        copy_tile_to_dst_init_short_with_dt(cb_qkt_im, cb_mask_in);
    } else {
        copy_tile_to_dst_init_short(cb_mask_in);
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
    if constexpr (is_causal_sdpa && !kv_pad_rotation_enabled && sliding_window_size == 0 && !use_provided_mask) {
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
    bool use_provided_mask = false>
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
    const KVPadRotationContext& kv_pad_rotation = {}) {
    // Callers guarantee active_Sk is evenly divisible by actual_sbw (via largest_factor_le).
    const uint32_t kt_num_full_subblocks = active_Sk / actual_sbw;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t in0_block_w = DHt;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qkt_subblock_h;
    constexpr uint32_t q_subblock_num_tiles = qkt_subblock_h * in0_block_w;
    constexpr uint32_t row_tiles = qkt_subblock_h * KT_stride;  // Use KT_stride for cb_qkt_im row width

    static_assert(!(use_padded_mask && ring_mode), "use_padded_mask and ring_mode are mutually exclusive");

    uint32_t pushed_rows = 0;
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();

    // Use KT_stride for cb_qkt_im layout to keep CB pointers aligned across iterations
    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT_stride);

    CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);
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
        sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_identity_scale_in, cb_q_in>();
        mm_no_mop_init_short(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);
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

        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_full_subblocks; ++kt_subblock) {
            if (q_subblock > 0) {
                uint32_t prev_q_subblock = q_subblock - 1;
                sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
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
                sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in>();
                mm_no_mop_reinit_short(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);
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
                    /*skip_pack_configure=*/q_subblock == 0);
                // Signal the first half is committed so run()#1 overlaps the second-half
                // pack. STALL_PACK drains the L1 writes before the token retires.
                if (overlap_first_half && kt_subblock == first_half_last_sb) {
                    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::UNPACK_MATH_DONE)));
                }
                kt_index_offset += actual_sbw;
            }
        }
        // Restore float16b for mask/reduce after Q@KT.
        sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();

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
            reconfig_data_format_srca(cb_mask_in, cb_qkt_im);
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
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * vDHt;
        // cb_qkt_im row width is KT_stride (for pointer alignment), not Sk_chunk_t
        constexpr uint32_t qktv_in0_row_tiles = qktv_h * KT_stride;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_row_tiles;

        // When save_out_cb is set, V matmul + SALAD write to save_out_cb (cb_out) instead of cur.out.
        // Writer drains save_out_cb row-by-row to DRAM during SALAD. cur.out stays empty.
        const uint32_t out_cb = (save_out_cb != INVALID_CB) ? save_out_cb : cur.out;

        // V wait deferred: don't block here. The sub_exp drain loop below
        // doesn't touch V, so the reader's V DMA can overlap with the drain.
        CircularBuffer(out_cb).reserve_back(qktv_output_num_tiles);

        // q_subblock 0: drain last row's sub_exp in-place + first QKT@V matmul
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "Softmax(Q@KT)@V");
            const uint32_t matmul_inner = actual_sbw;

            // sub_exp_block_bcast_cols softmaxes the last Q row in place, one column-subblock at a
            // time. The PACK->UNPACK barrier after it makes those in-place pack writes visible to
            // the V-matmul unpack; only needed when q_num_subblocks==1 (Phase 1's hold_wr_ptr
            // didn't sync them).

            if constexpr (!kt_inplace_v) {
                // Split-drain (common, materialized-V path): interleave each column-subblock's
                // sub_exp with its partial V matmul; partial products accumulate across kt_sub via L1.
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
                    if constexpr (q_num_subblocks == 1) {
                        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                    }
                    if (kt_sub == 0) {
                        CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
                        CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                    }
                    if (kt_sub > 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }

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
                            const uint32_t qktv_in1_index = kt_sub * matmul_inner * vDHt + v_index_offset;
                            blocked_matmul_and_pack<false, vDHt, vDHt>(
                                cb_qkt_im,
                                cb_v_in,
                                out_cb,
                                qktv_in0_index_offset + kt_sub * matmul_inner,
                                qktv_in1_index,
                                0,
                                v_subblock * qktv_subblock_w,
                                qktv_subblock_w,
                                qktv_h,
                                matmul_inner,
                                KT_stride,
                                /*skip_pack_configure=*/true);
                            v_index_offset += qktv_subblock_w;
                        }
                        sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                    }

                    if (kt_sub > 0) {
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

        // Per-row normalization lambda — fires on last K chunk (standard or deferred norm).
        // Takes sbh so it works for both full subblocks (qktv_h) and remainder (qktv_remainder_h).
        [[maybe_unused]] uint32_t sink_row_offset = 0;
        [[maybe_unused]] auto normalize_row = [&](uint32_t& pushed, uint32_t sbh) {
            MaybeDeviceZoneScopedN(profiling_enabled, "ROW_NORM");
            CircularBuffer(cur.sum).push_back(sbh);
            CircularBuffer(out_cb).push_back(sbh * vDHt);
            normalize_row_streaming<
                profiling_enabled,
                vDHt,
                dst_size,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                scale_fp32,
                use_attention_sink,
                cb_attention_sink>(cur.sum, out_cb, sbh, cur.max, sink_row_offset);
            if constexpr (use_attention_sink) {
                sink_row_offset += sbh;
            }
            pushed++;
        };

        // SALAD correction lambda — works for both full subblocks (sbh=qktv_h) and
        // remainder (sbh=qktv_remainder_h). Normalization is independently guarded at call sites.
        // prev.out is consumed row-by-row: always read from CB front, then pop after use.
        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, uint32_t sbh) {
            PACK((llk_pack_reconfig_l1_acc(1)));
            {
                MaybeDeviceZoneScopedN(profiling_enabled, "S_CORR_FUSED");
                // ob_q_subblock=0: prev.out and cb_exp_max_diff are popped row-by-row (read from front).
                // sum_q_subblock=salad_row: prev.sum uses cumulative indexing (not popped per-row).
                if constexpr (has_qktv_remainder) {
                    if (sbh == qktv_remainder_h) {
                        salad_correct_fused<qktv_remainder_h, vDHt, dst_size>(
                            prev.out, prev.sum, cb_exp_max_diff, out_cb, cur.sum, 0, salad_row, w_salad);
                    } else {
                        salad_correct_fused<qktv_h, vDHt, dst_size>(
                            prev.out, prev.sum, cb_exp_max_diff, out_cb, cur.sum, 0, salad_row, w_salad);
                    }
                } else {
                    salad_correct_fused<qktv_h, vDHt, dst_size>(
                        prev.out, prev.sum, cb_exp_max_diff, out_cb, cur.sum, 0, salad_row, w_salad);
                }
            }
            CircularBuffer(cb_exp_max_diff).pop_front(sbh);
            CircularBuffer(prev.out).pop_front(sbh * vDHt);
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
                sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                    prev.max, cur.max, cb_exp_max_diff, salad_row, qktv_h);
                CircularBuffer(cb_exp_max_diff).push_back(qktv_h);
            }

            // V matmul for current row group — cur_h adapts for remainder
            if (is_remainder_iter) {
                CircularBuffer(cb_qkt_im).wait_front(Sq_chunk_t * KT_stride);
            } else {
                CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
            }
            {
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
                        v_subblock * qktv_subblock_w,
                        qktv_subblock_w,
                        cur_h,
                        active_Sk,
                        KT_stride,
                        /*skip_pack_configure=*/true);
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
                    sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                        prev.max, cur.max, cb_exp_max_diff, drain_salad_row, drain_h);
                    CircularBuffer(cb_exp_max_diff).push_back(drain_h);

                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(qktv_h);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt);
                        pushed_rows++;
                    }

                    const uint32_t drain_w = has_qktv_remainder ? ((qktv_q_num_subblocks - pushed_rows) * qktv_h)
                                                                : (qktv_q_num_subblocks - 1 - pushed_rows);
                    salad_correct_row(drain_salad_row, drain_w, drain_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(drain_h);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt);
                        pushed_rows++;
                    }
                } else {
                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(qktv_h);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt);
                        pushed_rows++;
                    }
                }
            } else if (is_last_iter) {
                normalize_row(pushed_rows, qktv_h);
            } else {
                CircularBuffer(cur.sum).push_back(qktv_h);
                CircularBuffer(out_cb).push_back(qktv_h * vDHt);
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
                    sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                        prev.max, cur.max, cb_exp_max_diff, drain_salad_row, drain_h);
                    CircularBuffer(cb_exp_max_diff).push_back(drain_h);
                    salad_correct_row(drain_salad_row, 0, drain_h);
                }
                if (is_last_iter) {
                    normalize_row(pushed_rows, drain_h);
                } else {
                    CircularBuffer(cur.sum).push_back(drain_h);
                    CircularBuffer(out_cb).push_back(drain_h * vDHt);
                    pushed_rows++;
                }
            } else {
                // Drain was hoisted into the last main-loop iteration above.
                // For is_first_iter (no SALAD), the drain row still needs push/normalize.
                if (is_first_iter) {
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
                        CircularBuffer(cur.sum).push_back(drain_h);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt);
                        pushed_rows++;
                    }
                }
            }
        }

        // All rows pushed individually — no bulk push needed.

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
    bool use_provided_mask = false>
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

        // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
        // The unpack MOP is split in half (block_ct_dim / 2), so active_Sk must be even,
        // and we need >1 subblock so the semaphore fires before the reduce's second half.
        constexpr bool can_reduce_trigger =
            (Sk_chunk_t % qkt_subblock_w == 0) && (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

        // Pre-compute subblock width: compile-time for full chunks, hoisted for padded last chunk.
        constexpr uint32_t full_sbw = qkt_subblock_w;
        constexpr uint32_t padded_sbw = (last_chunk_Sk < Sk_chunk_t && last_chunk_Sk % qkt_subblock_w != 0)
                                            ? largest_factor_le(last_chunk_Sk, qkt_subblock_w)
                                            : qkt_subblock_w;

        // With largest_factor_le, padded chunks also have evenly-dividing subblocks,
        // so reduce_trigger can be enabled when the same constraints hold for last_chunk_Sk.
        constexpr bool can_reduce_trigger_padded = (padded_k_tiles_inner > 0) && (last_chunk_Sk % padded_sbw == 0) &&
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
                apply_sliding_window);
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

            // Post-iteration cleanup
            // prev.out and cb_exp_max_diff are already popped row-by-row inside salad_correct_row.
            if (!is_first) {
                sdpa_cb_pop_front_out_of_line(prev.max, Sq_chunk_t);
                sdpa_cb_pop_front_out_of_line(prev.sum, Sq_chunk_t);
            }

            if (is_last) {
                sdpa_cb_pop_front_out_of_line(cur.max, Sq_chunk_t);
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
 * @tparam qkt_subblock_h   QK matmul subblock height
 * @tparam cb_max_in      CB for restoring row-max from DRAM (multi Q-chunk, c_6)
 * @tparam cb_max_out     CB for saving row-max to DRAM (multi Q-chunk, c_17)
 * @tparam cb_normalized_out CB for normalized output rows (written by normalize_row_streaming)
 * @tparam cb_sum_out       CB for saving row-sum to DRAM (multi Q-chunk, c_10)
 * @tparam cb_sum_in        CB for restoring row-sum from DRAM (multi Q-chunk, c_11)
 * @tparam local_padded_Nt   Per-device KV padded sequence length in tiles (N_local / TILE_H)
 * @tparam q_local_padded_Nt Per-device Q padded sequence length in tiles. Under chunked-prefill it
 *                           also doubles as the per-chunk K-region stride on this device (one
 *                           chunk's Q lives in one such region); equals local_padded_Nt otherwise.
 * @tparam chunk_size_t      Per-chunk Q/K extent in tiles (chunked-only)
 *
 * @param global_q_start     First global Q chunk index for this core
 * @param global_q_end       One-past-last global Q chunk index for this core
 * @param num_kv_chunks      Total K chunks this ring iter (local + joint if applicable)
 * @param ring_iter          Current ring iteration (0..ring_size-1)
 * @param ring_id            Device ID within the ring that owns this iter's KV shard
 * @param num_local_k_chunks Number of K chunks from the local (non-joint) sequence
 * @param logical_nt         Actual (unpadded) global sequence length in tiles
 * @param acc_state          Persistent accumulator state (prev/cur CB halves for ping-pong)
 * @param is_last_ring_iter  True on the final ring iteration — triggers normalization
 * @param q_per_core         Number of Q chunks per core (1 = L1-only, >1 = DRAM round-trip)
 * @param lw_mask            Lightweight mask context for partial-tile padding
 * @param chunked            Chunked-prefill runtime state (q_start_idx_t, ring_index); ignored when
 * chunked_enabled=false
 * @param is_first_active_iter Set on the first ring iter that does work (decoupled from ring_iter==0 for skipped-chain
 * bounds)
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
    uint32_t cb_normalized_out = 0,
    uint32_t cb_sum_out = 0,
    uint32_t cb_sum_in = 0,
    uint32_t cb_signal = 0,
    bool lightweight_mask_enabled = false,
    bool is_causal_sdpa = false,
    bool is_balanced_sdpa = false,
    bool chunked_enabled = false,
    uint32_t local_padded_Nt = 0,
    uint32_t q_local_padded_Nt = 0,
    uint32_t chunk_size_t = 0,
    bool global_n_mask_enabled = false,
    bool local_n_mask_enabled = false,
    bool joint_n_mask_enabled = false,
    bool straddle_mask_enabled = false,
    bool kv_pad_rotation_enabled = false,
    uint32_t v_cb_physical_width_t = vDHt,
    bool v_shares_k_buffer = false,
    bool kt_inplace_v = false,
    bool sparse_frames_enabled = false,
    uint32_t frame_seqlen_tiles = 0,
    uint32_t num_frames_padded_compile = 0,
    typename MaskCtx = LightweightMaskContext>
void sdpa_ring_v2(
    const uint32_t global_q_start,
    const uint32_t global_q_end,
    const uint32_t num_kv_chunks,
    const uint32_t num_q_chunks,
    const uint32_t ring_iter,
    const uint32_t ring_id,
    const uint32_t num_local_k_chunks,
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
    const MaskCtx& lw_mask = {},
    const bool skip_first_half_q = false,
    const bool use_zigzag_balancing = false,
    const ChunkedContext& chunked = {},
    const bool is_first_active_iter = true,
    const uint32_t* frame_allow_words = nullptr) {
    init_sdpa_streaming_semaphores();

    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    // is_causal: diagonal stamp only on iter 0 (K is local-frame). Chunked: every iter (absolute coords).
    const bool is_causal_iter = (is_causal_sdpa && (ring_iter == 0)) || chunked_enabled;

    // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
    // All conditions are compile-time except the active_Sk == Sk_chunk_t guard (padded chunks).
    constexpr bool can_reduce_trigger =
        (Sk_chunk_t % qkt_subblock_w == 0) && (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

    // Subblock width for the non-padded (common) case — compile-time constant.
    constexpr uint32_t full_sbw = qkt_subblock_w;

    // Pre-compute subblock widths for each mask type (hoisted out of per-chunk loop).
    uint32_t global_n_sbw = full_sbw;
    uint32_t local_n_sbw = full_sbw;
    uint32_t joint_n_sbw = full_sbw;
    uint32_t straddle_sbw = full_sbw;
    if constexpr (global_n_mask_enabled) {
        global_n_sbw = lw_mask.global_n_padded_tiles
                           ? largest_factor_le(Sk_chunk_t - lw_mask.global_n_padded_tiles, qkt_subblock_w)
                           : full_sbw;
    }
    if constexpr (local_n_mask_enabled) {
        local_n_sbw = lw_mask.local_n_padded_tiles
                          ? largest_factor_le(Sk_chunk_t - lw_mask.local_n_padded_tiles, qkt_subblock_w)
                          : full_sbw;
    }
    if constexpr (joint_n_mask_enabled) {
        joint_n_sbw = lw_mask.joint_n_padded_tiles
                          ? largest_factor_le(Sk_chunk_t - lw_mask.joint_n_padded_tiles, qkt_subblock_w)
                          : full_sbw;
    }
    if constexpr (straddle_mask_enabled) {
        straddle_sbw = lw_mask.straddle_num_padded_tiles
                           ? largest_factor_le(Sk_chunk_t - lw_mask.straddle_num_padded_tiles, qkt_subblock_w)
                           : full_sbw;
    }

    uint32_t KV_chunks_processed_in_iter = 0;

    // ---- Q-loop helpers ---------------------------------------------------

    // Non-last ring iter: drain restored staging CBs and skip this Q chunk.
    auto try_balanced_skip = [&](uint32_t q_chunk) -> bool {
        if constexpr (is_balanced_sdpa) {
            if (skip_first_half_q && q_chunk < num_q_chunks / 2 && !is_last_ring_iter) {
                return true;
            }
        }
        return false;
    };

    // Last ring iter: normalize accumulated state and signal writer, then skip K-loop.
    auto try_normalize_only = [&](uint32_t q_chunk) -> bool {
        if constexpr (is_balanced_sdpa) {
            if (skip_first_half_q && q_chunk < num_q_chunks / 2 && is_last_ring_iter) {
                AccumulatorHalf q_prev_norm = acc_state.prev;
                if (q_per_core > 1 && ring_iter > 0) {
                    q_prev_norm = {cb_sum_in, cb_max_in, cb_prev_out};
                }
                constexpr uint32_t norm_dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
                normalize_row_streaming<
                    false,
                    vDHt,
                    norm_dst_size,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_normalized_out>(q_prev_norm.sum, q_prev_norm.out, Sq_chunk_t);
                sdpa_cb_pop_front_out_of_line(q_prev_norm.max, Sq_chunk_t);
                if (q_per_core > 1) {
                    CircularBuffer(cb_signal).reserve_back(1);
                    sdpa_cb_push_back_out_of_line(cb_signal, 1);
                }
                return true;
            }
        }
        return false;
    };

    // ---- K-loop helpers ---------------------------------------------------

    // Skip KV chunks beyond the logical sequence length (padding tiles).
    auto try_skip_oob_kv = [&](uint32_t k_chunk, bool kv_chunk_is_joint) -> bool {
        if (kv_chunk_is_joint) {
            return false;
        }
        return !kv_chunk_starts_before_logical_end<
            kv_pad_rotation_enabled,
            chunked_enabled,
            local_padded_Nt,
            chunk_size_t,
            q_local_padded_Nt>(ring_id, k_chunk * Sk_chunk_t, logical_nt);
    };

    // Causal skip: K chunks fully above the diagonal — drain K/V from CBs and skip.
    auto try_skip_causal_above_diag = [&](uint32_t k_chunk, uint32_t causal_k_limit) -> bool {
        if constexpr (is_causal_sdpa) {
            if (is_causal_iter && k_chunk >= causal_k_limit) {
                CircularBuffer(cb_kt_in).wait_front(DHt * Sk_chunk_t);
                sdpa_cb_pop_front_out_of_line(cb_kt_in, DHt * Sk_chunk_t);
                // In-place latent-V never pushes a V entry, so only K^T needs draining.
                if constexpr (!kt_inplace_v) {
                    CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                    sdpa_cb_pop_front_out_of_line(cb_v_in, Sk_chunk_t * v_cb_physical_width_t);
                }
                KV_chunks_processed_in_iter++;
                return true;
            }
        }
        return false;
    };

    // Sparse-frames skip: this (q_frame, k_frame) pair is disallowed by the packed bitmap.
    // Drain K/V from CBs (reader pushes K/V for every k_chunk in an active ring iter; compute
    // must consume) and skip. `q_frame_for_chunk` is set per Q chunk before the k_chunk loop
    // (uniform within a chunk because Sq_chunk_t == frame_seqlen_tiles). k_frame comes from the
    // K chunk's absolute global tile position in the padded sequence.
    //
    //   bit_idx = q_frame * num_frames_padded_compile + k_frame
    //   allowed = (frame_allow_words[bit_idx / 32] >> (bit_idx % 32)) & 1
    auto try_skip_sparse_frames = [&](uint32_t k_chunk, uint32_t q_frame_for_chunk, bool kv_chunk_is_joint) -> bool {
        if constexpr (sparse_frames_enabled) {
            if (kv_chunk_is_joint) {
                return false;  // joint K is always attended (no frame semantics)
            }
            // K chunk's global tile position along the padded sequence (all sp shards concatenated).
            const uint32_t k_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
            const uint32_t k_frame = k_global_start_tile / frame_seqlen_tiles;
            const uint32_t bit_idx = q_frame_for_chunk * num_frames_padded_compile + k_frame;
            const uint32_t word = frame_allow_words[bit_idx >> 5];
            const uint32_t bit = (word >> (bit_idx & 31u)) & 1u;
            if (bit == 0u) {
                CircularBuffer(cb_kt_in).wait_front(DHt * Sk_chunk_t);
                sdpa_cb_pop_front_out_of_line(cb_kt_in, DHt * Sk_chunk_t);
                if constexpr (!kt_inplace_v) {
                    CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                    sdpa_cb_pop_front_out_of_line(cb_v_in, Sk_chunk_t * v_cb_physical_width_t);
                }
                KV_chunks_processed_in_iter++;
                return true;
            }
        }
        return false;
    };

    // -----------------------------------------------------------------------

    for (uint32_t q = global_q_start; q < global_q_end; q++) {
        // Compute Q chunk index (with optional zigzag remapping for causal balancing).
        // num_q_chunks is total per-head chunks (local + joint), matching the divisor the
        // writer/reader use to flatten (batch, head, q_chunk) — see ring_joint_sdpa.cpp.
        uint32_t q_chunk = remap_q_index(q, num_q_chunks, use_zigzag_balancing) % num_q_chunks;

        // Causal K-chunk limit and Q start tile for this Q chunk
        uint32_t causal_k_limit = num_kv_chunks;
        uint32_t q_start_tile = 0;
        if constexpr (is_causal_sdpa) {
            if (is_causal_iter) {
                q_start_tile = q_chunk * Sq_chunk_t;
                causal_k_limit = (q_start_tile + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t;
            }
        } else if constexpr (chunked_enabled) {
            // Absolute Q tile row. Diag stamp masks K past Q's range; logical_n skip handles K past the cache.
            q_start_tile = chunked.q_start_idx_t + chunked.ring_index * q_local_padded_Nt + q_chunk * Sq_chunk_t;
        }

        if (try_balanced_skip(q_chunk)) {
            continue;
        }
        if (try_normalize_only(q_chunk)) {
            continue;
        }

        // Per-Q pre-scan: count K chunks that will actually be processed.
        // Placed after balanced-skip guards so skipped Q chunks don't pay for the scan.
        // Note: this scan does NOT drain CBs (reader hasn't pushed yet); it only mirrors the
        // skip predicates so `is_last_k` fires on the right chunk in the main loop below.
        //
        // Sparse-frames: q_frame is uniform within a Q chunk when the pattern is enabled (host
        // asserts frame_seqlen_tiles % Sq_chunk_t == 0, so no Q chunk straddles a frame). Chunks
        // may be smaller than a frame; the integer division below still maps every chunk to
        // exactly one q_frame. Computed once here and reused inline.
        uint32_t q_frame_for_this_chunk = 0;
        if constexpr (sparse_frames_enabled) {
            // q_start_tile was set above under `is_causal_sdpa || chunked_enabled`; for the
            // sparse-frames path (mutually exclusive with those), derive it from q_chunk.
            q_frame_for_this_chunk = (q_chunk * Sq_chunk_t) / frame_seqlen_tiles;
        }
        uint32_t per_q_valid_kv = 0;
        for (uint32_t k = 0; k < num_kv_chunks; ++k) {
            const bool is_joint = k >= num_local_k_chunks;
            if (try_skip_oob_kv(k, is_joint)) {
                continue;
            }
            if constexpr (is_causal_sdpa) {
                if (is_causal_iter && k >= causal_k_limit) {
                    continue;
                }
            }
            if constexpr (sparse_frames_enabled) {
                if (!is_joint) {
                    const uint32_t k_global = local_padded_Nt * ring_id + k * Sk_chunk_t;
                    const uint32_t k_frame = k_global / frame_seqlen_tiles;
                    const uint32_t bit_idx = q_frame_for_this_chunk * num_frames_padded_compile + k_frame;
                    const uint32_t word = frame_allow_words[bit_idx >> 5];
                    if (((word >> (bit_idx & 31u)) & 1u) == 0u) {
                        continue;
                    }
                }
            }
            per_q_valid_kv++;
        }

        // Use persistent accumulator state from caller (single Q-chunk)
        // or restore from DRAM (multi Q-chunk).
        AccumulatorHalf q_prev = acc_state.prev, q_cur = acc_state.cur;

        const bool is_first_kv_for_this_q = is_first_active_iter;

        // Multi Q-chunk restore: K0 reads prev accumulators directly from staging buffers
        // (cb_prev_out, cb_max_in, cb_sum_in) — no copy_block needed.
        // After K0's swap, reset q_cur to original accumulator CBs for normal ping-pong.
        const AccumulatorHalf original_prev = q_prev;
        const bool restore_from_staging = (q_per_core > 1 && !is_first_active_iter);
        if (restore_from_staging) {
            q_prev = {cb_sum_in, cb_max_in, cb_prev_out};
        }

        uint32_t KV_chunks_processed = 0;

        for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
            const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
            if (try_skip_oob_kv(k_chunk, kv_chunk_is_joint)) {
                continue;
            }
            if (try_skip_causal_above_diag(k_chunk, causal_k_limit)) {
                continue;
            }
            if (try_skip_sparse_frames(k_chunk, q_frame_for_this_chunk, kv_chunk_is_joint)) {
                continue;
            }

            KV_chunks_processed++;
            KV_chunks_processed_in_iter++;

            const bool is_first = is_first_kv_for_this_q && (KV_chunks_processed == 1);
            const bool is_last_k = (KV_chunks_processed == per_q_valid_kv);

            // Last K chunk of last ring_iter triggers per-row normalization
            const bool is_last_k_of_last_ring_iter = is_last_ring_iter && is_last_k;

            // Signal writer that last K-chunk is starting (for row-by-row DMA save/restore).
            if (is_last_k && q_per_core > 1) {
                CircularBuffer(cb_signal).reserve_back(1);
                sdpa_cb_push_back_out_of_line(cb_signal, 1);
            }

            // Determine if this K chunk needs masking (partial tile within a tile boundary)
            bool is_global_n_mask_chunk = false;
            bool is_local_n_mask_chunk = false;
            bool is_joint_n_mask_chunk = false;
            if constexpr (global_n_mask_enabled) {
                is_global_n_mask_chunk = ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id;
            }
            if constexpr (local_n_mask_enabled) {
                is_local_n_mask_chunk = local_n_needs_masking && k_chunk == local_n_mask_chunk_id;
            }
            if constexpr (joint_n_mask_enabled) {
                is_joint_n_mask_chunk = ring_iter_needs_joint_n_mask && kv_chunk_is_joint &&
                                        (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id;
            }
            // Straddle chunk (rix > rid balanced-causal with k_chunk_size ∤ coarse_chunk_size):
            // only the early-half columns attend; late-half columns must be dropped. Narrow
            // active_Sk like local_n; no partial-tile stamp needed for tile-aligned straddle.
            bool is_straddle_mask_chunk = false;
            if constexpr (straddle_mask_enabled) {
                is_straddle_mask_chunk =
                    lw_mask.straddle_num_padded_tiles > 0 && k_chunk == lw_mask.straddle_mask_chunk_id;
            }

            bool apply_mask = kv_pad_rotation_enabled;
            if constexpr (global_n_mask_enabled) {
                apply_mask = apply_mask || is_global_n_mask_chunk;
            }
            if constexpr (joint_n_mask_enabled) {
                apply_mask = apply_mask || is_joint_n_mask_chunk;
            }

            // Resolve lightweight mask params for partial tile masking
            uint32_t lw_partial_tile_idx = 0;
            if constexpr (lightweight_mask_enabled) {
                if (apply_mask) {
                    bool partial_tile_selected = false;
                    if constexpr (global_n_mask_enabled) {
                        if (is_global_n_mask_chunk) {
                            lw_partial_tile_idx = lw_mask.global_n_partial_tile_idx;
                            partial_tile_selected = true;
                        }
                    }
                    if constexpr (joint_n_mask_enabled) {
                        if (!partial_tile_selected && is_joint_n_mask_chunk) {
                            lw_partial_tile_idx = lw_mask.joint_l_partial_tile_idx;
                        }
                    }
                }
            }

            // Tile-level matmul skip for global_n, local_n, or joint_n padding.
            // Runtime reduce narrows to active_Sk; matmul/sub_exp/V also need narrowing.
            // Also select pre-computed subblock width for this chunk's mask type.
            uint32_t active_Sk_param = Sk_chunk_t;
            uint32_t chunk_sbw = full_sbw;
            bool narrowed_by_mask = false;
            if constexpr (global_n_mask_enabled) {
                if (is_global_n_mask_chunk) {
                    active_Sk_param = Sk_chunk_t - lw_mask.global_n_padded_tiles;
                    chunk_sbw = global_n_sbw;
                    narrowed_by_mask = true;
                }
            }
            if constexpr (local_n_mask_enabled) {
                if (!narrowed_by_mask && is_local_n_mask_chunk) {
                    active_Sk_param = Sk_chunk_t - lw_mask.local_n_padded_tiles;
                    chunk_sbw = local_n_sbw;
                    narrowed_by_mask = true;
                }
            }
            if constexpr (joint_n_mask_enabled) {
                if (!narrowed_by_mask && is_joint_n_mask_chunk) {
                    active_Sk_param = Sk_chunk_t - lw_mask.joint_n_padded_tiles;
                    chunk_sbw = joint_n_sbw;
                    narrowed_by_mask = true;
                }
            }
            if constexpr (straddle_mask_enabled) {
                if (!narrowed_by_mask && is_straddle_mask_chunk) {
                    active_Sk_param = Sk_chunk_t - lw_mask.straddle_num_padded_tiles;
                    chunk_sbw = straddle_sbw;
                }
            }

            // Causal narrowing on the diagonal-crossing K-chunk: cols past the last Q-row's
            // diag tile are -inf for every row in the Q-chunk, so skip matmul/sub_exp/V there.
            // Composes with any prior padding narrowing via min().
            if constexpr (is_causal_sdpa && !kv_pad_rotation_enabled) {
                if (is_causal_iter && k_chunk == causal_k_limit - 1) {
                    const uint32_t causal_active = q_start_tile + Sq_chunk_t - k_chunk * Sk_chunk_t;
                    if (causal_active < active_Sk_param) {
                        active_Sk_param = causal_active;
                        chunk_sbw = largest_factor_le(active_Sk_param, qkt_subblock_w);
                    }
                }
            }

            // On last K-chunk of non-last ring iters (multi-Q), redirect output, sum, and max
            // to writer-staging CBs, eliminating post-loop copy_block calls.
            // Writer drains cb_out row-by-row during SALAD; cb_sum_out and cb_max_out bulk after.
            const bool save_to_staging = is_last_k && !is_last_ring_iter && q_per_core > 1;
            const uint32_t step_save_out_cb = save_to_staging ? cb_out : INVALID_CB;
            const uint32_t step_save_max_cb = save_to_staging ? cb_max_out : INVALID_CB;
            if (save_to_staging) {
                q_cur.sum = cb_sum_out;
            }

            // K start tile fed to diag stamp must share Q's coord frame (local for is_causal, global for chunked).
            const uint32_t step_k_start_tile =
                chunked_enabled ? kv_global_tile_for_local<true, local_padded_Nt, chunk_size_t, q_local_padded_Nt>(
                                      ring_id, k_chunk * Sk_chunk_t)
                                : (k_chunk * Sk_chunk_t);

            // Chunked-prefill straddle. Background: each device's K cache holds the per-chunk
            // K region for every chunk back-to-back, q_local_padded_Nt tiles per region. When
            // k_chunk_size does not divide q_local_padded_Nt, a single K-chunk can begin in
            // one region (chunk j) and end in the next (chunk j+1). Because adjacent regions
            // map to *non-adjacent* global K positions (jumping by chunk_size_t between them),
            // the global K coord is no longer contiguous across the K-chunk's columns. We
            // signal this to the diag stamp via:
            //   - straddle_col: column index at which the jump happens (= tiles remaining in
            //     region j from this K-chunk's start)
            //   - straddle_jump: the global-K increment at that boundary
            //     (= chunk_size_t - q_local_padded_Nt, i.e. the gap between region j's end and
            //     region j+1's start in global K).
            // When straddle_col > 0 the stamp evaluates the diagonal per column instead of
            // per row, applying the jump for columns >= straddle_col.
            uint32_t step_straddle_col = 0;
            uint32_t step_straddle_jump = 0;
            if constexpr (chunked_enabled) {
                if (q_local_padded_Nt > 0) {
                    const uint32_t local_start = k_chunk * Sk_chunk_t;
                    const uint32_t slab_end_local = (local_start / q_local_padded_Nt + 1) * q_local_padded_Nt;
                    if (local_start + Sk_chunk_t > slab_end_local) {
                        step_straddle_col = slab_end_local - local_start;
                        step_straddle_jump = chunk_size_t - q_local_padded_Nt;
                    }
                }
            }
            KVPadRotationContext step_kv_pad_rotation = chunked.kv_pad_rotation;
            step_kv_pad_rotation.k_local_start_tile = k_chunk * Sk_chunk_t;
            step_kv_pad_rotation.ring_id = ring_id;
            step_kv_pad_rotation.logical_tile_count = logical_nt;

            sdpa_inner_loop_step<
                false,  // profiling_enabled
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
                false,                              // use_padded_mask — ring uses ring mask instead
                true,                               // ring_mode
                is_causal_sdpa || chunked_enabled,  // chunked re-enables causal masking with absolute coords
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
                kv_pad_rotation_enabled,
                q_local_padded_Nt,
                chunk_size_t,
                local_padded_Nt,
                v_cb_physical_width_t,
                kt_inplace_v>(
                q_prev,
                q_cur,
                is_last_k_of_last_ring_iter,
                is_first,
                apply_mask,
                lw_partial_tile_idx,
                active_Sk_param,
                can_reduce_trigger && (active_Sk_param == Sk_chunk_t),
                chunk_sbw,
                step_save_out_cb,
                step_save_max_cb,
                is_causal_iter,
                kv_pad_rotation_enabled ? q_chunk * Sq_chunk_t : q_start_tile,
                step_k_start_tile,
                lw_mask.neginf_tile_idx,
                lw_mask.causal_diag_tile_idx,
                0,      // sliding_leading_prev_idx (not used in ring)
                0,      // sliding_leading_idx (not used in ring)
                0,      // sliding_trailing_next_idx (not used in ring)
                false,  // apply_sliding_window
                step_straddle_col,
                step_straddle_jump,
                step_kv_pad_rotation);

            // Post-iteration cleanup: pop previous values and swap aliases
            // prev.out and cb_exp_max_diff are already popped row-by-row inside salad_correct_row.
            if (!is_first) {
                sdpa_cb_pop_front_out_of_line(q_prev.max, Sq_chunk_t);
                sdpa_cb_pop_front_out_of_line(q_prev.sum, Sq_chunk_t);
            }

            if (is_last_k_of_last_ring_iter) {
                // Normalization consumed cur.sum and cur.out; pop cur.max.
                sdpa_cb_pop_front_out_of_line(q_cur.max, Sq_chunk_t);
            } else {
                std::swap(q_prev, q_cur);
                // After K0's swap, q_cur holds staging buffers. Reset to original accumulator CBs.
                if (restore_from_staging && KV_chunks_processed == 1) {
                    q_cur = {original_prev.sum, original_prev.max, original_prev.out};
                }
            }
        }

        // Pop Q — not popped inside step since ring_mode gates the early Q pop.
        // When q_per_core == 1, Q is identical across ring iterations so we keep it
        // fronted in the CB and only pop on the last iteration to avoid redundant DRAM re-reads.
        if (q_per_core > 1 || is_last_ring_iter) {
            sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
        }

        // Persist or save accumulators for next ring iteration
        if (q_per_core == 1) {
            // Single Q-chunk: persist in L1 (no DRAM round-trip)
            acc_state.prev = q_prev;
            acc_state.cur = q_cur;
        } else if (!is_last_ring_iter) {
            // Multi Q-chunk: save raw accumulators to DRAM via writer CBs.
            // Out tiles already saved row-by-row via cb_out during last K-chunk SALAD.
            // Sum already in cb_sum_out (redirected via q_cur.sum on last K-chunk).
            // Max already in cb_max_out (dual-write from DST during reduce on last K-chunk).
            // Pop the accumulator CB for max — dual-write doesn't replace the alias,
            // so the accumulator CB still has tiles from the last K-chunk's reduce.
            // (Sum doesn't need this because planting replaced the alias entirely.)
            sdpa_cb_pop_front_out_of_line(q_prev.max, Sq_chunk_t);
        }
        // On last ring_iter: normalized output already in cb_out from normalize_row_streaming
    }

    // Dummy KV pop for CB write-pointer phase alignment across chained reader cores.
    for (uint32_t dummy_chunk = 0; dummy_chunk < dummy_kv_chunks_for_phase_alignment<v_shares_k_buffer, kt_inplace_v>(
                                                     KV_chunks_processed_in_iter);
         ++dummy_chunk) {
        CircularBuffer(cb_kt_in).wait_front(DHt * Sk_chunk_t);
        sdpa_cb_pop_front_out_of_line(cb_kt_in, DHt * Sk_chunk_t);
        // In-place latent-V never pushes a V entry, so there is nothing extra to drain.
        if constexpr (!kt_inplace_v) {
            CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
            sdpa_cb_pop_front_out_of_line(cb_v_in, Sk_chunk_t * v_cb_physical_width_t);
        }
    }
}
