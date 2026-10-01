// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streaming SDPA compute helpers.
// Included by sdpa_recipe.cpp for the explicit Blackhole numerical recipes.
// Depends on primitives from compute_common.hpp (must be included first).

#pragma once

#ifndef SDPA_RECIPE_FP32
#include "compensated_sfpu.hpp"
#endif

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp"
#include "lofi_scaling.hpp"

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

// Blackhole's pack-to-unpack handshake overlaps row-max reduction with QK packing.
constexpr bool reduce_trigger_supported = true;

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

#include "sdpa_cb_protocol.hpp"
#ifdef SDPA_RECIPE_FP32
#include "fp32_state.hpp"
#endif

struct AccumulatorHalf {
    uint32_t sum, max, out;
};

// LoFi no-MOP matmuls bake the reuse-side source clear into the recorded replay image
// (reuse A when ct_dim >= rt_dim, else reuse B); mm_no_mop_reinit_short does not re-record it.
// When a reinit changes the reuse side (a one-tile-wide QK subblock followed by a wider PV
// subblock, or a single-row tail), MATH would clear the wrong source and UNPACK would wait
// forever. Reuse-B shapes (only new narrow geometries) always re-record, since other matmul
// inits (e.g. normalization) may have replaced the image; reuse-A shapes re-record only after a
// reuse-B image, so reuse-A-only builds skip the re-record.
// HiFi replays clear outside the image and need none of this.
#ifdef SDPA_RECIPE_LOFI
static bool recipe_mm_reuse_a = true;
#endif
ALWI void recipe_mm_init(uint32_t in0, uint32_t in1, bool transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
    mm_no_mop_init_short(in0, in1, transpose, ct, rt, kt);
#ifdef SDPA_RECIPE_LOFI
    recipe_mm_reuse_a = ct >= rt;
#endif
}
ALWI void recipe_mm_reinit(uint32_t in0, uint32_t in1, bool transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
#ifdef SDPA_RECIPE_LOFI
    if (ct < rt || !recipe_mm_reuse_a) {
        recipe_mm_init(in0, in1, transpose, ct, rt, kt);
        return;
    }
#endif
    mm_no_mop_reinit_short(in0, in1, transpose, ct, rt, kt);
}

// Q chunks up to 1024 rows (32 tiles); row pairs bound the per-state validity flags.
constexpr uint32_t kRecipeMaxQTiles = 32;
constexpr uint32_t kRecipeMaxRowGroups = kRecipeMaxQTiles / 2;
// Any tile-aligned K chunk: the QK subblock width (SDPA_RECIPE_QK_W, host-chosen to divide
// it) sets the reduction pieces; the early max-reduce overlap needs at least two subblocks.
template <uint32_t Sk_chunk_t>
constexpr bool kRecipeValidKTiles = Sk_chunk_t >= 1;
#ifndef SDPA_RECIPE_QK_W
#define SDPA_RECIPE_QK_W 4
#endif

struct RecipeAccumulatorState {
    AccumulatorHalf prev, cur;
    uint32_t processed_chunks = 0;
#ifndef SDPA_RECIPE_FP32
    bool group_local_valid[kRecipeMaxRowGroups] = {};
#endif
};

// Sentinel for "no CB" — beyond the valid 0-31 range.
constexpr uint32_t INVALID_CB = 32;
#ifdef SDPA_RECIPE_FP32
static bool sdpa_skip_prev_sum_pop = false;
#endif
// Blackhole benefits from blocked packing at width four.
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

#ifdef SDPA_RECIPE_FP32
static sdpa::streaming::Fp32PackConfig sdpa_pack;
#ifdef TRISC_PACK
ALWI void sdpa_cached_pack_format(uint32_t cb) {
    if (sdpa_pack.format_cb == INVALID_CB || pack_src_format[sdpa_pack.format_cb] != pack_src_format[cb] ||
        pack_dst_format[sdpa_pack.format_cb] != pack_dst_format[cb]) {
        pack_reconfig_data_format(cb);
    }
    sdpa_pack.format_cb = cb;
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
    PACK(if (sdpa_pack.width == pack_width) { return; })
    PACK(sdpa_pack.width = pack_width;)
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
    bool skip_pack_configure = false) {
    tile_regs_acquire();
    uint32_t dst_index = 0;
    uint32_t in0_index = in0_index_start;
    uint32_t in1_index = in1_index_start;
#if defined(SDPA_KO_QKMM) || defined(SDPA_KO_PVMM)
#ifdef SDPA_KO_QKMM
    if constexpr (transpose) {
        inner_dim = 0;
    }
#endif
#ifdef SDPA_KO_PVMM
    if constexpr (!transpose) {
        inner_dim = 0;
    }
#endif
#endif
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
#ifdef SDPA_RECIPE_K_PRIMARY_ROWS
    if constexpr (transpose) {
        mask_recipe_tail(out_col_offset, subblock_w, subblock_h);
    }
#endif
    if (!skip_pack_configure) {
        configure_row_pack_width(out_cb, subblock_w);
    }
#ifdef SDPA_KO_SPACK
    if constexpr (!transpose)
#endif
    pack_contiguous_rows_nocfg(
        out_cb, row_subblock_idx * subblock_h, subblock_h, out_num_cols, out_col_offset, subblock_w);
    tile_regs_release();
}

// Row maxima combine the current QK block with the previous online maximum.
// Perf research knockouts (not for merge): each SDPA_KO_* define removes one piece of work; output is wrong.
//
// SDPA_PROTO_PA (B/E, perf research): reference-max numerator. P uses the first K chunk's row max for the
// whole Q chunk, so every later chunk is an identity correction; PV accumulates across chunks in plane 0 of
// the scratch CB (Float32, set by the host) with packer L1 accumulate; no numerator fold. The denominator
// keeps its compensated fold.
#if defined(SDPA_PROTO_PA) && !defined(SDPA_RECIPE_FP32)
#define SDPA_PA 1
#ifndef SDPA_PA_DETECT_ULPS
#define SDPA_PA_DETECT_ULPS 256u  // perf research stand-in for the tau threshold (BF16 ordered-key distance)
#endif
// Set per K step: the Float32 denominator accumulates onto earlier chunks after the first.
inline bool sdpa_pa_sum_acc = false;
#ifndef SDPA_PA_DBG
#define SDPA_PA_DBG 0
#endif
#define PA_PV_BEGIN(pv_cb, acc)                     \
    do {                                            \
        pack_reconfig_data_format(pv_cb);           \
        PACK((llk_pack_reconfig_l1_acc((((acc) || (SDPA_PA_DBG & 8)) && !(SDPA_PA_DBG & 1)) ? 1 : 0))); \
    } while (0)
#define PA_PV_END(restore_cb)                 \
    do {                                      \
        PACK((llk_pack_reconfig_l1_acc(0)));  \
        pack_reconfig_data_format(restore_cb); \
    } while (0)
#endif
#if defined(SDPA_PA) && defined(SDPA_PA_SAFE)
#define PA_SUM_CB(cb) 12  // chunk-local BF16 row sums
#else
#define PA_SUM_CB(cb) (cb)
// Defaults for builds without the safe P-A (defined in Phase 2 when it is on).
#define PA_FIRST_FOLD(rows)
#define PA_PV_OFF(g) group_pv_offset
#define PA_PV_ACC(g) (!is_first_iter)
#endif
#ifdef SDPA_PERF_ZONES
constexpr bool sdpa_perf_zones = true;
#else
constexpr bool sdpa_perf_zones = false;
#endif

#ifdef SDPA_PA
#ifndef SDPA_PA_DBG
#define SDPA_PA_DBG 0
#endif
#ifndef SDPA_PROTO_PA_TAU
// 28 ln 2: shifts the fast exp's INT16 grid by exactly 28 octaves (256 * 28 = 7168 steps), so P is
// 2^-28 times the unshifted P bit for bit and the exp's error pattern is unchanged.
#define SDPA_PROTO_PA_TAU (28.0f * 0.69314718055994531f)
#endif
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// Reference max bias: m_ref = m + tau / scale, so exp(scale * (s - m_ref)) stays in the fast exp's range
// (its INT16 grid saturates about 0.72 above zero) until a row max grows by more than tau.
template <uint32_t scale_fp32>
inline void calculate_sdpa_pa_bias() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float bias = (SDPA_PA_DBG & 128) ? SDPA_PROTO_PA_TAU / __builtin_bit_cast(float, scale_fp32) : 0.0f;
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat m = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = m + bias;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
#endif

#if defined(SDPA_PA) && defined(SDPA_PA_SAFE)
#ifndef SDPA_PA_THETA
// Rescale threshold: keep the reference max until the row max grows by more than theta (scaled units).
// theta + 0.72 must stay below tau (the exp's headroom), so P never saturates.
#define SDPA_PA_THETA (16.0f * 0.69314718055994531f)
#endif
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// dest tile 0: max(m_ref, rowmax) from the reduce; tile `group`: m_ref. Keep m_ref unless exceeded by theta.
template <uint32_t scale_fp32, int group>
inline void calculate_sdpa_pa_select() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float theta = SDPA_PA_THETA / __builtin_bit_cast(float, scale_fp32);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat m = sfpi::dst_reg[0];
        sfpi::vFloat r = sfpi::dst_reg[32 * group];
        v_if(m <= r + theta) { sfpi::dst_reg[0] = r; }
        v_endif;
        sfpi::dst_reg++;
    }
}
// dest tiles [0, w): state; [w, 2w): this chunk's term; tile 2w: column-broadcast correction c.
// state = state * c + term.
template <int w>
inline void calculate_sdpa_pa_rescale_add() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat c = sfpi::dst_reg[32 * 2 * w];
#pragma GCC unroll 2
        for (int t = 0; t < w; ++t) {
            sfpi::vFloat x = sfpi::dst_reg[32 * t];
            sfpi::vFloat y = sfpi::dst_reg[32 * (w + t)];
            sfpi::dst_reg[32 * t] = x * c + y;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
#endif

#if defined(SDPA_PA) && (defined(SDPA_PA_DETECT) || defined(SDPA_PA_PCHECK))
// Overflow detection result (perf research: computed, not yet acted on).
static volatile uint32_t sdpa_pa_overflow = 0;
#endif

template <uint32_t in0_cb, uint32_t scale_cb, uint32_t row_stride, uint32_t pa_scale_fp32 = 0, bool pa_force_reduce = false>
void reduce_c_row_group(
    uint32_t out_cb,
    uint32_t prev_cb,
    uint32_t row_group_index,
    bool do_eltwise_max,
    uint32_t sbh,
    uint32_t reduce_cols,
    bool respect_trigger = false,
    bool overlap_first_half = false) {
    const uint32_t group_size = sbh;
    const uint32_t row_start = row_group_index * group_size;

    // row_stride: physical row width in the CB (may exceed cols on the reduced path).
    const uint32_t cumulative_input_tiles = (row_group_index + 1) * group_size * row_stride;
    const uint32_t cumulative_prev_tiles = (row_group_index + 1) * group_size;

    // scale_cb assumed ready (waited once at kernel init)

#ifdef SDPA_KO_REDUCE
    CircularBuffer(in0_cb).wait_front(cumulative_input_tiles);
    tile_regs_acquire();
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < group_size; i++) {
        pack_tile<false>(i, out_cb);
    }
    tile_regs_release();
    return;
#endif
    tile_regs_acquire();

#if (defined(SDPA_PA) || (defined(SDPA_PROTO_PA32) && defined(SDPA_RECIPE_FP32))) && !defined(SDPA_PA_REDUCE_PROBE) && \
    !defined(SDPA_PA_SAFE)
    if (do_eltwise_max && !pa_force_reduce) {
        // Reference max: carry the previous maximum unchanged (bitwise), skipping the reduce. A plain copy:
        // the reduce's seeding copy transposes within faces for the reduce's dest layout.
        CircularBuffer(prev_cb).wait_front(cumulative_prev_tiles);
#ifdef SDPA_RECIPE_FP32
        sdpa_stream_reconfig_srca(prev_cb);
#endif
        copy_init(prev_cb);
        for (uint32_t i = 0; i < group_size; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
#ifdef SDPA_RECIPE_FP32
        configure_single_tile_pack(out_cb);
#endif
        for (uint32_t i = 0; i < group_size; i++) {
            pack_tile<false>(i, out_cb);
        }
        tile_regs_release();
        return;
    }
#endif
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
#if defined(SDPA_PA) && defined(SDPA_PA_SAFE)
    // The reference (m_ref) beside the true max, for the conditional rescale below.
    if (do_eltwise_max) {
        copy_init(prev_cb);
        for (uint32_t i = 0; i < group_size; i++) {
            copy_tile(prev_cb, row_start + i, group_size + i);
        }
    }
#endif
#ifdef SDPA_PA_REDUCE_PROBE
    // Cost probe for overflow detection: the true max is computed above; carry the reference max.
    if (do_eltwise_max) {
#ifdef SDPA_RECIPE_FP32
        sdpa_stream_reconfig_srca(prev_cb);
#endif
        copy_init(prev_cb);
        for (uint32_t i = 0; i < group_size; i++) {
            copy_tile(prev_cb, row_start + i, group_size + i);
        }
    }
#endif

    tile_regs_commit();
    tile_regs_wait();
#if defined(SDPA_PA) && defined(SDPA_PA_SAFE)
    if (do_eltwise_max) {
        static_assert(true);
        for (uint32_t i = 0; i < group_size; i++) {
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_pa_select, (pa_scale_fp32, 2), i, VectorMode::None)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }
#endif
#if defined(SDPA_PA) && (SDPA_PA_DBG & 128)
    // First K chunk only (later chunks return above with the carried reference).
    for (uint32_t i = 0; i < group_size; i++) {
        PACK((SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_pa_bias, (pa_scale_fp32), i, VectorMode::None)));
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
#ifdef SDPA_RECIPE_FP32
    configure_single_tile_pack(out_cb);
#endif

    for (uint32_t i = 0; i < group_size; i++) {
#ifdef SDPA_PA_REDUCE_PROBE
        pack_tile<false>(do_eltwise_max ? group_size + i : i, out_cb);
#else
        pack_tile<false>(i, out_cb);
#endif
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
    // One batch per QK subblock column group (the subblock width divides the K chunk).
#ifdef SDPA_I4
    // I4 (perf research): s - m on the SFPU in FP32. Two score tiles per batch leave a dest slot for m.
    constexpr uint32_t score_batch = SDPA_RECIPE_QK_W > 2 ? 2 : SDPA_RECIPE_QK_W;
#else
    constexpr uint32_t score_batch = SDPA_RECIPE_QK_W;
#endif
    static_assert(score_batch == 1 || score_batch == 2 || score_batch == 4);
    CircularBuffer(max_cb).wait_front((q_subblock + 1) * tiles_per_row);
#ifndef SDPA_I4
    if (global_col_base == 0) {
#ifndef SDPA_KO_SUBL1
        sdpa_subtract_max_l1(inout_cb, max_cb, max_row_base, cols_in_row);
#endif
    }
#endif
    cb_alias_read_ptr(7, inout_cb);
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
#ifdef SDPA_I4
            // The row maximum (BF16, exact in FP32) broadcast along columns into the slot after the scores.
            sdpa_stream_reconfig(max_cb, max_cb);
            unary_bcast_init<BroadcastType::COL>(max_cb);
            unary_bcast<BroadcastType::COL>(max_cb, max_row_base + i, score_batch);
            unary_bcast_uninit<BroadcastType::COL>(max_cb);
#endif
            tile_regs_commit();
            tile_regs_wait();
#ifdef SDPA_I4
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_sub_bcast_max, (score_batch), 0, VectorMode::None)));
#endif
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
                (32 * score_batch, true),
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
#ifdef SDPA_KO_SUB
        for (uint32_t i = 0; i < 0; i++) {
#else
        for (uint32_t i = 0; i < tiles_per_row; i++) {
#endif
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
#ifndef SDPA_KO_EXP
#ifdef SDPA_RECIPE_FP32
        if (tiles_per_row * tiles_per_column == 4) {
            PACK((ckernel::sfpu::restore_sdpa_grid_macro_instructions()));
            PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_grid_batch, (128), 0, vector_mode_exp)));
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_refine_loadmacro, (128, false), 0, vector_mode_exp)));
        } else
#endif
        {
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                for (uint32_t j = 0; j < tiles_per_column; j++) {
#ifdef SDPA_RECIPE_FP32
                    PACK((ckernel::sfpu::restore_sdpa_grid_macro_instructions()));
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
                        calculate_sdpa_exp_refine_loadmacro,
                        (iterations, false),
                        dst_index,
                        vector_mode_exp)));
                    ++dst_index;
#elif defined(SDPA_I6)
                    // I6: exp tile t, then pack it; the next tile's SFPU exp overlaps this pack.
                    if (i == 0 && j == 0) {
                        configure_single_tile_pack(inout_cb);
                    }
                    exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index, vector_mode_exp);
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    pack_tile<true>(dst_index, inout_cb, (max_row_base + i) * cols_in_row + global_col_base + j);
                    ++dst_index;
#else
                    exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index++, vector_mode_exp);
#endif
                }
            }
        }
#endif
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }

    {
        MaybeDeviceZoneScopedN(profiling_enabled, "PACK SUB_EXP");
        // Pack back to inout_cb at the same absolute positions.
        // In Phase 1, the caller pre-configures (cb_qkt_im, actual_sbw) before the kt loop
        // and blocked_matmul_and_pack restores it after each sub_exp. Skip the redundant
        // reconfigure here when the caller guarantees the state.
#if !defined(SDPA_KO_PPACK) && !(defined(SDPA_I6) && !defined(SDPA_RECIPE_FP32))
        if (skip_pack_configure) {
            pack_contiguous_rows_nocfg(
                inout_cb, max_row_base, tiles_per_row, cols_in_row, global_col_base, tiles_per_column);
        } else {
            pack_contiguous_rows(inout_cb, max_row_base, tiles_per_row, cols_in_row, global_col_base, tiles_per_column);
        }
#endif
#if !defined(SDPA_RECIPE_FP32) && !defined(SDPA_KO_SUMPACK) && !defined(SDPA_PA_DENOM)
#if defined(SDPA_PA) && !(SDPA_PA_DBG & 16) && !defined(SDPA_PA_SAFE)
        pack_reconfig_data_format(reduce_cb);
#endif
        configure_single_tile_pack(reduce_cb);
        {
            uint32_t dst_index = 0;
#pragma GCC unroll 1
            for (uint32_t i = 0; i < tiles_per_row; i++) {
#if defined(SDPA_PA) && !defined(SDPA_PA_SAFE)
                if (global_col_base > 0 || sdpa_pa_sum_acc) {
#else
                if (global_col_base > 0) {
#endif
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
#if defined(SDPA_PA) && !(SDPA_PA_DBG & 16) && !defined(SDPA_PA_SAFE)
        pack_reconfig_data_format(inout_cb);
#endif
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
#ifdef SDPA_KO_IDENT
    identical = 1;
#endif
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

#ifndef SDPA_RECIPE_FP32
// The caller fences the current chunk's unpublished L1 writes once per row
// group, then keeps pack in overwrite/single-tile mode for all state updates.
// Copying the next DST half can overlap the current half's PACK-thread SFPU work.
ALWI void sdpa_compensated_sum_update(
    uint32_t old_cb,
    uint32_t new_cb,
    uint32_t correction_cb,
    uint32_t old_hi,
    uint32_t old_lo,
    uint32_t new_hi,
    uint32_t new_lo,
    uint32_t correction_row,
    uint32_t new_read_hi,
    bool identity_correction = false,
    uint32_t rows = 2) {
    // A single remaining row runs the paired SFPU program with row 0 duplicated
    // into the second slot. SFPU lanes are independent; only row 0 is packed.
    tile_regs_acquire();
    copy_init(old_cb);
    for (uint32_t b = 0; b < 2; ++b) {
        const uint32_t r = b < rows ? b : 0;
        SDPA_FOLD_COPY(copy_tile(old_cb, old_hi + r * 2, 3 * b));
        SDPA_FOLD_COPY(copy_tile(old_cb, old_lo + r * 2, 3 * b + 1));
        SDPA_FOLD_COPY(copy_tile(new_cb, new_read_hi + r * 2, 3 * b + 2));
    }
    if (!identity_correction) {
        unary_bcast_init<BroadcastType::COL>(correction_cb);
        SDPA_FOLD_COPY(unary_bcast<BroadcastType::COL>(correction_cb, correction_row, 6));
        SDPA_FOLD_COPY(unary_bcast<BroadcastType::COL>(correction_cb, correction_row + (rows > 1 ? 1 : 0), 7));
        unary_bcast_uninit<BroadcastType::COL>(correction_cb);
    }
    tile_regs_commit();
    tile_regs_wait();
    SDPA_FOLD_SFPU(PACK((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sdpa_identity_state,
        (2, true),
        0,
        VectorMode::None,
        identity_correction))));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    for (uint32_t b = 0; b < rows; ++b) {
        pack_tile<true>(3 * b, new_cb, new_hi + b * 2);
        pack_tile<true>(3 * b + 1, new_cb, new_lo + b * 2);
    }
    tile_regs_release();
}

#endif

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
        if (identity_correction && tiles_per_column == 4) {
            // Identity needs no correction tile, so all four FP32 DST slots
            // hold independent numerator tiles. L1 adds remain unchanged.
            sdpa::streaming::rescale_and_accumulate<4>(
                out_in_cb,
                out_out_cb,
                bcast_cb,
                (ob_row_base + i) * tiles_per_column,
                (write_row_base + i) * tiles_per_column,
                ob_row_base + i,
                true);
        } else {
            for (uint32_t j = 0; j + 1 < tiles_per_column; j += 2) {
                sdpa::streaming::rescale_and_accumulate<2>(
                    out_in_cb,
                    out_out_cb,
                    bcast_cb,
                    (ob_row_base + i) * tiles_per_column + j,
                    (write_row_base + i) * tiles_per_column + j,
                    ob_row_base + i,
                    identity_correction);
            }
            if (tiles_per_column % 2 != 0) {
                // Odd head-dim width: the last numerator column on its own.
                const uint32_t j = tiles_per_column - 1;
                sdpa::streaming::rescale_and_accumulate<1>(
                    out_in_cb,
                    out_out_cb,
                    bcast_cb,
                    (ob_row_base + i) * tiles_per_column + j,
                    (write_row_base + i) * tiles_per_column + j,
                    ob_row_base + i,
                    identity_correction);
            }
        }
        sdpa::streaming::rescale_and_accumulate<1, true>(
            sum_in_cb,
            sum_out_cb,
            bcast_cb,
            sum_row_base + i,
            write_row_base + i,
            ob_row_base + i,
            identity_correction);
    }
    PACK((llk_pack_reconfig_l1_acc(1)));
#else

#ifndef SDPA_PA
    CircularBuffer(out_in_cb).wait_front((ob_q_subblock + 1) * tiles_per_row * tiles_per_column * sdpa_out_stride);
#endif
    CircularBuffer(sum_in_cb).wait_front((sum_q_subblock + 1) * tiles_per_row * sdpa_sum_stride);
    CircularBuffer(bcast_cb).wait_front((ob_q_subblock + 1) * tiles_per_row);

    static_assert((sbh_t == 1 || sbh_t == 2) && sbw_t >= 1 && dst_size == 8);
#ifdef SDPA_KO_FOLD
    PACK((llk_pack_reconfig_l1_acc(1)));
    return;
#endif
#if !defined(SDPA_KO_NUMFOLD) && !defined(SDPA_PA)
    group2_numerator_row(
        out_in_cb,
        out_out_cb,
        bcast_cb,
        current_sum_popped ? write_row_base : sum_row_base,
        sum_row_base,
        current_sum_popped ? write_row_base : sum_row_base,
        identity_correction,
        group_boundary,
        group_odd,
        group_has_local,
        tiles_per_row,
        tiles_per_column);
#endif
    PACK((ckernel::sfpu::init_sdpa_compensated_block_macros()));
#ifdef SDPA_KO_SUMFOLD
    PACK((llk_pack_reconfig_l1_acc(1)));
    return;
#endif

    configure_single_tile_pack(sum_out_cb);
    PACK((ckernel::sfpu::init_sdpa_compensated_sum_replay()));
    for (uint32_t i = 0; i < tiles_per_row; i += 2) {
        sdpa_compensated_sum_update(
            sum_in_cb,
            sum_out_cb,
            bcast_cb,
            2 * (sum_row_base + i),
            2 * (sum_row_base + i) + 1,
            2 * (write_row_base + i),
            2 * (write_row_base + i) + 1,
            ob_row_base + i,
            2 * (current_sum_popped ? write_row_base + i : sum_row_base + i),
            identity_correction,
            tiles_per_row - i < 2 ? 1 : 2);
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
    uint32_t normalized_out_cb>
static __attribute__((noinline, noclone)) void normalize_row_streaming(
    uint32_t cur_sum_cb, uint32_t cur_out_cb, uint32_t sbh) {
#ifdef SDPA_RECIPE_FP32
    sdpa::streaming::normalize_rows<head_dim_t_, col_identity_cb>(
        cur_sum_cb, cur_out_cb, scratch_cb, normalized_out_cb, sbh, sdpa_pack);
#else

    configure_single_tile_pack(scratch_cb);
    for (uint32_t s = 0; s < sbh; s++) {
#if defined(SDPA_PA) && defined(SDPA_PA_DENOM)
        // P-A matmul denominator: l is already a column (P * ones column) in column 0 of the Float32 tile;
        // its other columns are not row sums, so no cross-column reduce. 1/l straight from column 0.
        {
            CircularBuffer(cur_sum_cb).wait_front(sdpa_sum_stride);
            reconfig_data_format_srca(cur_sum_cb);
            copy_init(cur_sum_cb);
            CircularBuffer(scratch_cb).reserve_back(1);
            tile_regs_acquire();
            copy_tile(cur_sum_cb, 0, 0);
            recip_tile_init();
            MATH((recip_tile(0 /*dst_index*/, VectorMode::C)));
            tile_regs_commit();
            tile_regs_wait();
            configure_single_tile_pack(scratch_cb);
            pack_tile(0, scratch_cb);
            tile_regs_release();
            CircularBuffer(scratch_cb).push_back(1);
            CircularBuffer(cur_sum_cb).pop_front(sdpa_sum_stride);
        }
#else
        // 1+2. Fused matmul_reduce + recip: sum × col_identity → recip → 1/sum in scratch
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "NORM_MATMUL_RECIP");
            constexpr uint32_t N = 1;
#ifdef SDPA_PA
            // The matmul unpack path does not read Float32 into a 16-bit dest: round the accumulated
            // denominator to BF16 through dest into the spare CB 12 (the unused ping-pong bank) first.
#ifdef SDPA_PA_SAFE
            constexpr uint32_t pa_sum_bf16_cb = 8;
#else
            constexpr uint32_t pa_sum_bf16_cb = 12;
#endif
            CircularBuffer(cur_sum_cb).wait_front(sdpa_sum_stride);
            reconfig_data_format_srca(cur_sum_cb);
            copy_init(cur_sum_cb);
            CircularBuffer(pa_sum_bf16_cb).reserve_back(1);
            tile_regs_acquire();
            copy_tile(cur_sum_cb, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(pa_sum_bf16_cb);
            configure_single_tile_pack(pa_sum_bf16_cb);
            pack_tile(0, pa_sum_bf16_cb);
            tile_regs_release();
            CircularBuffer(pa_sum_bf16_cb).push_back(1);
            reconfig_data_format_srca(pa_sum_bf16_cb);
            configure_single_tile_pack(scratch_cb);
            const uint32_t norm_sum_cb = pa_sum_bf16_cb;
#else
            const uint32_t norm_sum_cb = cur_sum_cb;
#endif
            matmul_block_init(norm_sum_cb, col_identity_cb, 0, N, 1, N);
            sdpa_maybe_reconfig_data_format<normalized_out_cb, col_identity_cb, normalized_out_cb, scratch_cb>();
            // Pack format follows scratch_cb for the reciprocal intermediate. The old/new form folds away
            // when scratch and normalized output formats match, and reconfigures after rows that packed output.
            sdpa_maybe_pack_reconfig_data_format<normalized_out_cb, scratch_cb>();

            CircularBuffer(col_identity_cb).wait_front(N);
            CircularBuffer(norm_sum_cb).wait_front(1);

            CircularBuffer(scratch_cb).reserve_back(1);
            tile_regs_acquire();
            matmul_block(norm_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
#ifndef SDPA_PA
            if constexpr (sdpa_sum_stride == 2) {
                matmul_block(cur_sum_cb, col_identity_cb, 1, 0, 0, 0, N, 1, N);
            }
#endif

            recip_tile_init();
            MATH((recip_tile(0 /*dst_index*/, VectorMode::C)));
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, scratch_cb);
            tile_regs_release();
            CircularBuffer(scratch_cb).push_back(1);

#ifdef SDPA_PA
            CircularBuffer(norm_sum_cb).pop_front(1);
#endif
            CircularBuffer(cur_sum_cb).pop_front(sdpa_sum_stride);
        }
#endif

        // 3. Normalize: multiply output tiles by bcast_cols(1/sum)
        // Process in batches of up to dst_size tiles (DST capacity).
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "NORM_MUL_BCAST");
            constexpr uint32_t batch = (head_dim_t_ < dst_size) ? head_dim_t_ : dst_size;
#ifdef SDPA_PA
            reconfig_data_format_srca(cur_out_cb);
#endif
#if defined(SDPA_PA) && (SDPA_PA_DBG & 4)
            copy_init(cur_out_cb);
#else
            recipe_output_scale_init(cur_out_cb, scratch_cb);
#endif
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
#if defined(SDPA_PA) && (SDPA_PA_DBG & 4)
                    copy_tile(cur_out_cb, base + j, j);  // debug: unnormalized numerator
#else
                    recipe_output_scale_tile(cur_out_cb, scratch_cb, base + j, 0, j);
#endif
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
#ifdef SDPA_PA
            reconfig_data_format_srca(normalized_out_cb);
#endif
        }
    }
    // Restore pack format to scratch_cb (im_df = Float16_b) so that subsequent ops
    // (e.g. salad_correct_fused on the next K-chunk's drain row) pack to F16b CBs
    // with the right format. Without this, when normalized_out_cb has a different
    // format (e.g. Bfp8 output dtype), the format register stays Bfp8 and the next
    // pack to a F16b CB writes garbage that's later mis-decoded by F16b unpacks.
    sdpa_maybe_pack_reconfig_data_format<normalized_out_cb, scratch_cb>();
#endif
}

#ifdef SDPA_RECIPE_MASK
// Dense additive attn_mask (legacy SDPA contract): L1-accumulate one QK row group's mask tiles onto
// the packed scores in cb_qkt_im, before the row max. The reader streams one Q chunk x K chunk of
// mask per K chunk into CB 15, one Q tile row at a time. The add runs in the packer's L1
// accumulator (the same adder legacy SDPA uses); no recipe SFPU/FPU arithmetic changes, and
// unmasked builds compile this out.
constexpr uint32_t kRecipeMaskCb = 15;
// The reader always pushes whole group_rows groups (an odd chunk's single-row tail group carries a
// zero padding row), so waits/pops stay group-aligned in a one- or two-group CB.
template <uint32_t cb_qkt_im, uint32_t k_tiles, uint32_t group_rows>
static __attribute__((noinline, noclone)) void recipe_add_attn_mask(uint32_t row_start, uint32_t rows) {
    constexpr uint32_t batch = compute_kernel_lib::DEST_AUTO_LIMIT;
    const uint32_t tiles = rows * k_tiles;
    CircularBuffer(kRecipeMaskCb).wait_front(group_rows * k_tiles);
    configure_single_tile_pack(cb_qkt_im);
    sdpa_stream_reconfig_srca(kRecipeMaskCb);
    copy_init(kRecipeMaskCb);
    PACK((llk_pack_reconfig_l1_acc(1)));
    const uint32_t out_base = row_start * k_tiles;
    for (uint32_t i = 0; i < tiles; i += batch) {
        const uint32_t n = tiles - i < batch ? tiles - i : batch;
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            copy_tile(kRecipeMaskCb, i + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile<true>(j, cb_qkt_im, out_base + i + j);
        }
        tile_regs_release();
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    CircularBuffer(kRecipeMaskCb).pop_front(group_rows * k_tiles);
    // Callers expect srcA configured for the score CB (max reduce / next row group).
    sdpa_stream_reconfig_srca(cb_qkt_im);
}
#endif

/**
 * One K-chunk iteration of the streaming SDPA algorithm (v2 — no row buffers).
 * Phase 1: Q@KT directly into cb_qkt_im with cb_push_back_hold_wr_ptr, in-place sub_exp.
 * Phase 2: Drain + QKT@V with SALAD corrections, streaming normalization on last K iter.
 */
template <
    bool profiling_enabled,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
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
    uint32_t cb_normalized_out,
    bool independent_q_release = false>
static void sdpa_inner_loop_step(
    AccumulatorHalf& prev,
    AccumulatorHalf& cur,
    bool is_last_iter,
    bool is_first_iter
#ifndef SDPA_RECIPE_FP32
    ,
    uint32_t group_k_index,
    bool* group_local_valid
#endif
    ,
    bool release_q = false) {
    constexpr uint32_t KT_stride = Sk_chunk_t;
    constexpr uint32_t active_Sk = Sk_chunk_t;
    constexpr uint32_t actual_sbw = qkt_subblock_w;
#if defined(SDPA_KO_REDUCE) || defined(SDPA_KO_NOSPLIT) || (defined(SDPA_PA) && !defined(SDPA_PA_SAFE)) || \
    (defined(SDPA_PROTO_PA32) && defined(SDPA_RECIPE_FP32))
    constexpr bool reduce_trigger = false &&
#else
    constexpr bool reduce_trigger = reduce_trigger_supported && Sk_chunk_t % qkt_subblock_w == 0 &&
#endif
                                    Sk_chunk_t / qkt_subblock_w > 1 && Sk_chunk_t % 2 == 0;
#ifdef SDPA_RECIPE_FP32
    PACK(sdpa_pack.format_cb = INVALID_CB; sdpa_pack.width = 0;)
    sdpa_skip_prev_sum_pop = false;
#else
    // Row groups pair two query tile rows; an odd Q chunk ends with a single-row group.
    static_assert(
        Sq_chunk_t >= 1 && Sq_chunk_t <= kRecipeMaxQTiles && kRecipeValidKTiles<Sk_chunk_t> && vDHt >= 1 &&
        qkt_subblock_h == 2 && qktv_subblock_h == 2);
#endif
    const uint32_t kt_num_full_subblocks = active_Sk / actual_sbw;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t in0_block_w = DHt;
    // An odd Q chunk (BF16 recipes, two-row subblocks) ends with a one-row subblock. Helpers
    // address rows as index * height, so that subblock uses row index 2 * (n - 1) and height 1.
    constexpr bool q_tail_row = Sq_chunk_t % qkt_subblock_h != 0;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qkt_subblock_h + (q_tail_row ? 1 : 0);
    constexpr uint32_t q_subblock_num_tiles = qkt_subblock_h * in0_block_w;
    auto qk_rows = [](uint32_t q_subblock) -> uint32_t {
        return q_tail_row && q_subblock == q_num_subblocks - 1 ? 1 : qkt_subblock_h;
    };
    auto qk_index = [](uint32_t q_subblock) -> uint32_t {
        return q_tail_row && q_subblock == q_num_subblocks - 1 ? q_subblock * qkt_subblock_h : q_subblock;
    };

#ifdef SDPA_RECIPE_FP32
    static_assert(qkt_subblock_h == 1, "Early identity scan handles one query tile row per QK subblock");
    uint32_t sdpa_early_identity = !is_first_iter;
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
        qkt_subblock_h == 2 && qktv_subblock_h == 2, "Early guard pairs query tile rows (odd chunks end with one row)");
    // Per-K-step flags; only UNPACK fills these. Other threads obtain the
    // matching decision at the original correction mailbox rendezvous.
    uint32_t sdpa_identity_flags[Sq_chunk_t] = {};
#endif
    uint32_t pushed_rows = 0;
    // Q lives at [q_base_tiles, q_base_tiles + Sq_chunk_t*DHt) from the CB front. wait_front counts
    // from the front, so the wait target includes the chunks of earlier passes that stay resident.
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
#ifdef SDPA_RECIPE_FP32
    PACK((ckernel::sfpu::init_sdpa_refine_loadmacros()));
#endif
#if defined(SDPA_PA) && !(SDPA_PA_DBG & 2)
    // Reference max headroom: P = exp(scale * (s - m_ref) - tau). The fast exp computes
    // y = 256*log2(e)*scale*x + (B - C) as INT16 (saturating ~0.72 above zero); lowering the constant by
    // 256*log2(e)*tau moves saturation to tau + 0.72 while s - m_ref stays small (precise in BF16).
    PACK({
        constexpr float pa_exp_c = 32500.818359375f - 256.0f * 28.0f;  // tau = 28 ln 2, exact in the grid
        constexpr uint32_t pa_exp_bits = __builtin_bit_cast(uint32_t, pa_exp_c);
        TTI_SFPLOADI(0, 0xA, pa_exp_bits & 0xFFFF);
        TTI_SFPLOADI(0, 0x8, pa_exp_bits >> 16);
        TTI_SFPCONFIG(0, 13, 0);
    })
#endif

    // Use KT_stride for cb_qkt_im layout to keep CB pointers aligned across iterations
    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT_stride);

    CircularBuffer(cur.sum).reserve_back(Sq_chunk_t * sdpa_sum_stride);
#ifdef SDPA_PA
    sdpa_pa_sum_acc = !is_first_iter;
#endif
#if defined(SDPA_PA) && defined(SDPA_PA_SAFE)
    CircularBuffer(PA_SUM_CB(cur.sum)).reserve_back(Sq_chunk_t * sdpa_sum_stride);
#endif
#ifndef SDPA_RECIPE_FP32
    if (is_first_iter) {
        tile_regs_acquire();
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_zero_sum, 0, VectorMode::None)));
        tile_regs_commit();
        tile_regs_wait();
#if defined(SDPA_PA) && !(SDPA_PA_DBG & 32)
        pack_reconfig_data_format(cur.sum);
#endif
        configure_single_tile_pack(cur.sum);
        PACK((llk_pack_reconfig_l1_acc(0)));
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            pack_tile<true>(0, cur.sum, 2 * r + 1);
        }
#if defined(SDPA_PA) && !(SDPA_PA_DBG & 32)
        pack_reconfig_data_format(cb_qkt_im);
#endif
        tile_regs_release();
    }
#endif

    // ========== PHASE 1: Q@KT directly into cb_qkt_im ==========
    // All matmul output goes to cb_qkt_im at absolute offsets via pack_tile<true>.
    // cb_push_back_hold_wr_ptr makes each row visible to UNPACK without advancing wr_ptr.
    CircularBuffer(cb_kt_in).wait_front(DHt * KT_stride);

    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        MaybeDeviceZoneScopedN(profiling_enabled, "Softmax(Q@KT)");
        const uint32_t cur_qk_h = qk_rows(q_subblock);
        const uint32_t cur_qk_index = qk_index(q_subblock);
        CircularBuffer(cb_q_in).wait_front(q_wait_tiles - (qkt_subblock_h - cur_qk_h) * in0_block_w);
        kt_index_offset = 0;

        sdpa_maybe_pack_reconfig_data_format<cb_normalized_out, cb_qkt_im>();
#ifdef SDPA_RECIPE_FP32
        sdpa_stream_reconfig(cb_kt_in, cb_q_in);
#else
        sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_identity_scale_in, cb_q_in>();
#endif
        recipe_mm_init(cb_q_in, cb_kt_in, true, actual_sbw, cur_qk_h, in0_block_w);
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

#ifdef SDPA_RECIPE_MASK
        // The first-half max reduce must not read scores before the mask lands on them.
        constexpr bool overlap_first_half = false;
#else
        const bool overlap_first_half = reduce_trigger;
#endif
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
                        PA_SUM_CB(cur.sum),
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
                    recipe_mm_reinit(cb_q_in, cb_kt_in, true, actual_sbw, cur_qk_h, in0_block_w);
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
                        cur_qk_index,
                        kt_subblock * actual_sbw,
                        actual_sbw,
                        cur_qk_h,
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

#ifdef SDPA_RECIPE_MASK
        recipe_add_attn_mask<cb_qkt_im, KT_stride, qkt_subblock_h>(q_subblock * qkt_subblock_h, cur_qk_h);
#endif

        // Push row (visible for UNPACK reads) but keep wr_ptr stable
        cb_push_back_hold_wr_ptr(cb_qkt_im, cur_qk_h * KT_stride);

        // reduce_trigger barrier. Posted after pack + push so it dominates every
        // cb_qkt_im writer; gates run()#2 (and run()#1 on the non-overlap path). STALL_PACK drains
        // the writes; one post / one uninit get stays balanced (wait_on_zero is non-consuming).
        if (reduce_trigger) {
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::FPU_SFPU)));
        }

        // Max reduce: reads from cb_qkt_im at q_subblock position
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "Reduce max");
            CircularBuffer(cur.max).reserve_back(cur_qk_h);
            configure_single_tile_pack(cur.max);
            // Use reduce_trigger to enable early reduce start (before all matmul output is ready).
            // When reduce_trigger=true, the packer signals the unpacker via semaphore after partial output.
            reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT_stride, scale_fp32>(
                cur.max,
                prev.max,
                cur_qk_index,
                !is_first_iter /*do_eltwise_max*/,
                cur_qk_h,
                active_Sk,
                reduce_trigger,
                overlap_first_half);
            CircularBuffer(cur.max).push_back(cur_qk_h);
#if defined(SDPA_PA) && defined(SDPA_PA_DETECT)
            // Overflow detection off the exp's critical path: the carried reference max is already published;
            // compute the true max into the check CB and compare column 0 on UNPACK.
            if (!is_first_iter) {
                constexpr uint32_t pa_check_cb = 8;
                CircularBuffer(pa_check_cb).reserve_back(cur_qk_h);
                configure_single_tile_pack(pa_check_cb);
                reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT_stride, scale_fp32, true>(
                    pa_check_cb, prev.max, cur_qk_index, true, cur_qk_h, active_Sk, false, false);
                CircularBuffer(pa_check_cb).push_back(cur_qk_h);
                CircularBuffer(pa_check_cb).wait_front(cur_qk_h);
                UNPACK({
                    uint32_t grew = 0;
                    for (uint32_t t = 0; t < cur_qk_h; ++t) {
                        auto* m_new = reinterpret_cast<volatile uint32_t*>(
                            get_tile_l1_byte_address(get_operand_id(pa_check_cb), t));
                        auto* m_ref = reinterpret_cast<volatile uint32_t*>(
                            get_tile_l1_byte_address(get_operand_id(prev.max), cur_qk_index * cur_qk_h + t));
                        for (uint32_t face = 0; face < 2; ++face) {
                            for (uint32_t r = 0; r < 16; ++r) {
                                const uint32_t offset = face * 256 + r * 8;
                                // Order-preserving BF16 keys (sign-magnitude to unsigned).
                                uint32_t a = m_new[offset] & 0xffffu, b = m_ref[offset] & 0xffffu;
                                a = (a & 0x8000u) ? (~a & 0xffffu) : (a | 0x8000u);
                                b = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
                                grew |= a > b + SDPA_PA_DETECT_ULPS;
                            }
                        }
                    }
                    sdpa_pa_overflow = sdpa_pa_overflow | grew;
                })
                CircularBuffer(pa_check_cb).pop_front(cur_qk_h);
            }
#endif
        }

        q_index_offset += qkt_subblock_h * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }

    {
        CircularBuffer(cb_kt_in).pop_front(DHt * KT_stride);
    }

    // Release after the last local use, independently of final normalization
    // when a ring caller needs to reload Q for the next pass.
    {
        if (independent_q_release ? release_q : is_last_iter) {
            sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
        }
    }

    // ========== PHASE 2: Drain last row + QKT@V + SALAD ==========
    // After Phase 1: all rows are pushed (via hold_wr_ptr) in cb_qkt_im.
    // Rows 0..N-2 are softmax'd in-place; row N-1 has raw matmul output.
    {
#ifdef SDPA_PERF_PHASE_ZONES
        DeviceZoneScopedN("P2");
#endif
#ifdef SDPA_RECIPE_FP32
        // FP32 recipes use single-row PV groups (their in-place numerator and row-wise FP32
        // state updates assume one row per group), including narrower D64 subblocks.
        constexpr uint32_t qktv_h = qktv_subblock_h;
#else
        constexpr uint32_t qktv_h =
            ttnn::transformer::sdpa::streaming_qktv_h(qktv_subblock_h, qktv_subblock_w, dst_size, Sq_chunk_t);
#endif
        constexpr uint32_t qktv_remainder_h = Sq_chunk_t % qktv_h;
        // QK and PV row groups coincide; an odd BF16 chunk ends with one single-row group.
        static_assert(qktv_h == qkt_subblock_h && qktv_remainder_h <= 1 && Sq_chunk_t >= 1);

        static_assert(vDHt % qktv_subblock_w == 0, "vDHt must be evenly divisible by qktv_subblock_w");
        static_assert(qktv_h * qktv_subblock_w <= dst_size, "qktv subblock must fit in dest register file");
        constexpr uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_h + (qktv_remainder_h ? 1 : 0);
        constexpr uint32_t last_group = qktv_q_num_subblocks - 1;
        constexpr uint32_t last_h = qktv_remainder_h ? qktv_remainder_h : qktv_h;
        // Group g's row index in units of its own height.
        auto pv_rows = [](uint32_t group) -> uint32_t { return group == last_group ? last_h : qktv_h; };
        auto pv_index = [](uint32_t group) -> uint32_t {
            return group == last_group && last_h != qktv_h ? group * qktv_h : group;
        };
        constexpr uint32_t qktv_v_num_subblocks = vDHt / qktv_subblock_w;
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * vDHt * sdpa_out_stride;
        // cb_qkt_im row width is KT_stride (for pointer alignment), not Sk_chunk_t
        constexpr uint32_t qktv_in0_row_tiles = qktv_h * KT_stride;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_row_tiles;

#ifdef SDPA_RECIPE_FP32
        static_assert(
            Sq_chunk_t >= 1 && Sq_chunk_t <= kRecipeMaxQTiles && kRecipeValidKTiles<Sk_chunk_t> && vDHt >= 1 &&
            qktv_h == 1);
        // FP32 recipes use single-row QK and PV groups, so odd Q chunks need no remainder group.
        uint32_t inplace_numerator = !is_first_iter;
        if (inplace_numerator) {
            CircularBuffer(prev.max).wait_front(Sq_chunk_t);
            CircularBuffer(cur.max).wait_front(Sq_chunk_t);
            UNPACK({
                scan_identity_row(Sq_chunk_t - 1);
#ifdef SDPA_KO_IDENT
                inplace_numerator = 1;
#else
                inplace_numerator = sdpa_early_identity;
#endif
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
        const uint32_t out_cb = cur.out;
#ifndef SDPA_RECIPE_FP32
        // Odd K begins with empty local; put PV directly in its final local plane.
#ifdef SDPA_PA
        constexpr uint32_t group_pv_offset = 0;
#else
        const uint32_t group_pv_offset = (group_k_index % 2 == 1) ? vDHt : 0;
#endif
#if defined(SDPA_PA) && defined(SDPA_PA_SAFE)
        // Groups whose reference max is unchanged (bitwise, UNPACK's Phase-1 scans), sent to MATH and PACK.
        uint32_t pa_ident = 0;
        if (!is_first_iter) {
            CircularBuffer(prev.max).wait_front(Sq_chunk_t);
            CircularBuffer(cur.max).wait_front(Sq_chunk_t);
            UNPACK({
                sdpa_identity_flags[q_num_subblocks - 1] = sdpa_scan_identity_maxima(
                    prev.max, cur.max, qk_index(q_num_subblocks - 1), qk_rows(q_num_subblocks - 1));
                for (uint32_t g = 0; g < q_num_subblocks; ++g) {
                    pa_ident |= (sdpa_identity_flags[g] ? 1u : 0u) << g;
                }
                mailbox_write(ckernel::ThreadId::MathThreadId, pa_ident);
                mailbox_write(ckernel::ThreadId::PackThreadId, pa_ident);
            })
            MATH(pa_ident = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
            PACK(pa_ident = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        }
        auto pa_is_ident = [&](uint32_t g) -> bool { return !is_first_iter && ((pa_ident >> g) & 1u); };
        // Changed groups put this chunk's PV in plane 1 and fold it below; others accumulate in plane 0.
        auto pa_pv_offset = [&](uint32_t g) -> uint32_t { return (is_first_iter || pa_is_ident(g)) ? 0 : vDHt; };
        auto pa_pv_acc = [&](uint32_t g) -> bool { return pa_is_ident(g); };
        // Fold group g (rows Q tile rows) into the Float32 state: l (CB 13) from the chunk-local sums
        // (CB 12), and for changed groups O = O * c + PV (plane 1). In the last chunk, normalization pops
        // each group, so reads index from the current front.
        auto pa_fold = [&](uint32_t g, uint32_t rows, bool first, bool ident) {
            constexpr uint32_t lsum_cb = 12;
            const uint32_t read_row = is_last_iter ? 0 : g * rows;
            CircularBuffer(lsum_cb).push_back(rows * sdpa_sum_stride);
            CircularBuffer(lsum_cb).wait_front(rows * sdpa_sum_stride);
            if (first || ident) {
                reconfig_data_format_srca(lsum_cb);
                copy_init(lsum_cb);
                tile_regs_acquire();
                for (uint32_t r = 0; r < rows; ++r) {
                    copy_tile(lsum_cb, r * sdpa_sum_stride, r);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cur.sum);
                configure_single_tile_pack(cur.sum);
                PACK((llk_pack_reconfig_l1_acc(first ? 0 : 1)));
                for (uint32_t r = 0; r < rows; ++r) {
                    pack_tile<true>(r, cur.sum, r * sdpa_sum_stride);
                }
                PACK((llk_pack_reconfig_l1_acc(0)));
                pack_reconfig_data_format(cb_qkt_im);
                tile_regs_release();
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cur.sum);
                    copy_init(cur.sum);
                    copy_tile(cur.sum, (read_row + r) * sdpa_sum_stride, 0);
                    reconfig_data_format_srca(lsum_cb);
                    copy_init(lsum_cb);
                    copy_tile(lsum_cb, r * sdpa_sum_stride, 1);
                    unary_bcast_init<BroadcastType::COL>(cb_exp_max_diff);
                    unary_bcast<BroadcastType::COL>(cb_exp_max_diff, r, 2);
                    unary_bcast_uninit<BroadcastType::COL>(cb_exp_max_diff);
                    tile_regs_commit();
                    tile_regs_wait();
                    PACK((SFPU_UNARY_CALL(
                        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_pa_rescale_add, (1), 0, VectorMode::None)));
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    pack_reconfig_data_format(cur.sum);
                    configure_single_tile_pack(cur.sum);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    pack_tile<true>(0, cur.sum, r * sdpa_sum_stride);
                    pack_reconfig_data_format(cb_qkt_im);
                    tile_regs_release();
                }
                for (uint32_t r = 0; r < rows; ++r) {
                    for (uint32_t j = 0; j < vDHt; j += 2) {
                        const bool pair = j + 1 < vDHt;
                        tile_regs_acquire();
                        reconfig_data_format_srca(out_cb);
                        copy_init(out_cb);
                        const uint32_t rbase = (read_row + r) * 2 * vDHt + j;
                        copy_tile(out_cb, rbase, 0);
                        if (pair) {
                            copy_tile(out_cb, rbase + 1, 1);
                        }
                        copy_tile(out_cb, rbase + vDHt, pair ? 2 : 1);
                        if (pair) {
                            copy_tile(out_cb, rbase + vDHt + 1, 3);
                        }
                        reconfig_data_format_srca(cb_exp_max_diff);
                        unary_bcast_init<BroadcastType::COL>(cb_exp_max_diff);
                        unary_bcast<BroadcastType::COL>(cb_exp_max_diff, r, pair ? 4 : 2);
                        unary_bcast_uninit<BroadcastType::COL>(cb_exp_max_diff);
                        tile_regs_commit();
                        tile_regs_wait();
                        if (pair) {
                            PACK((SFPU_UNARY_CALL(
                                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_pa_rescale_add, (2), 0, VectorMode::None)));
                        } else {
                            PACK((SFPU_UNARY_CALL(
                                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_pa_rescale_add, (1), 0, VectorMode::None)));
                        }
                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                        pack_reconfig_data_format(out_cb);
                        configure_single_tile_pack(out_cb);
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        const uint32_t wbase = r * 2 * vDHt + j;
                        pack_tile<true>(0, out_cb, wbase);
                        if (pair) {
                            pack_tile<true>(1, out_cb, wbase + 1);
                        }
                        pack_reconfig_data_format(cb_qkt_im);
                        tile_regs_release();
                    }
                }
            }
            CircularBuffer(lsum_cb).pop_front(rows * sdpa_sum_stride);
            reconfig_data_format_srca(cb_qkt_im);
        };
#define PA_FIRST_FOLD(rows)                  \
    if (is_first_iter) {                     \
        pa_fold(0, rows, true, false);       \
    }
#define PA_PV_OFF(g) pa_pv_offset(g)
#define PA_PV_ACC(g) pa_pv_acc(g)
#endif
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
#if defined(SDPA_PA) && (SDPA_PA_DBG & 8)
            // Debug: zero the Float32 numerator plane explicitly; every PV pack then accumulates.
            pack_reconfig_data_format(out_cb);
            configure_single_tile_pack(out_cb);
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                for (uint32_t c = 0; c < vDHt; ++c) {
                    pack_tile<true>(0, out_cb, 2 * r * vDHt + c);
                }
            }
            pack_reconfig_data_format(cb_qkt_im);
#else
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                for (uint32_t c = 0; c < vDHt; ++c) {
                    pack_tile<true>(0, out_cb, (2 * r + 1) * vDHt + c);
                }
            }
#endif
            tile_regs_release();
        }
#endif

        // q_subblock 0: drain last row's sub_exp in-place + first QKT@V matmul
        {
            MaybeDeviceZoneScopedN(profiling_enabled, "Softmax(Q@KT)@V");
            const uint32_t matmul_inner = actual_sbw;
            const uint32_t drain_subblocks = kt_num_full_subblocks;
            // Group 0 is short only when the whole chunk is one single-row group (a one-tile
            // BF16 chunk); cb_qkt_im then holds one row, not qktv_h.
            constexpr uint32_t first_h = last_group == 0 ? last_h : qktv_h;

            // sub_exp_block_bcast_cols softmaxes the last Q row in place, one column-subblock at a
            // time. The PACK->UNPACK barrier after it makes those in-place pack writes visible to
            // the V-matmul unpack; only needed when q_num_subblocks==1 (Phase 1's hold_wr_ptr
            // didn't sync them).

            {
                // Split-drain (common, materialized-V path): interleave each column-subblock's
                // sub_exp with its partial V matmul; partial products accumulate across kt_sub via L1.
                for (uint32_t kt_sub = 0; kt_sub < drain_subblocks; ++kt_sub) {
                    sub_exp_block_bcast_cols<profiling_enabled, scale_fp32>(
                        cb_qkt_im,
                        cur.max,
                        PA_SUM_CB(cur.sum),
                        KT_stride,
                        qk_index(q_num_subblocks - 1),
                        kt_sub * matmul_inner,
                        qk_rows(q_num_subblocks - 1),
                        matmul_inner);
                    if constexpr (q_num_subblocks == 1) {
                        // Phase 1 published no earlier row, so make the in-place exponential
                        // visible to the PV unpack explicitly.
                        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                    }

                    if (kt_sub == 0) {
                        CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles - (qktv_h - first_h) * KT_stride);
                        CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * vDHt);
                    }
#ifdef SDPA_RECIPE_FP32
                    if (kt_sub > 0 || inplace_numerator) {
#elif defined(SDPA_PA)
                    PA_PV_BEGIN(out_cb, kt_sub > 0 || PA_PV_ACC(0));
                    if (false) {
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
                        recipe_mm_reinit(cb_qkt_im, cb_v_in, false, qktv_subblock_w, first_h, KT_stride);
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
                                v_subblock * qktv_subblock_w + PA_PV_OFF(0),
#endif
                                qktv_subblock_w,
                                first_h,
                                matmul_inner,
                                KT_stride,
                                /*skip_pack_configure=*/true);
#ifndef SDPA_RECIPE_FP32
                            UNPACK({
                                if (!is_first_iter && kt_sub == 0 && v_subblock == 0) {
                                    sdpa_identity_flags[q_num_subblocks - 1] = sdpa_scan_identity_maxima(
                                        prev.max,
                                        cur.max,
                                        qk_index(q_num_subblocks - 1),
                                        qk_rows(q_num_subblocks - 1));
                                }
                            })
#endif
                            v_index_offset += qktv_subblock_w;
                        }
                        sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                    }

#ifdef SDPA_RECIPE_FP32
                    if (kt_sub > 0 || inplace_numerator) {
#elif defined(SDPA_PA)
                    PA_PV_END(cb_qkt_im);
                    if (false) {
#else
                    if (kt_sub > 0) {
#endif
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    }
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
#if defined(SDPA_PA) && defined(SDPA_PA_PCHECK)
        // Overflow detection cost probe: rowmax(P) = exp(growth - tau), reduced after the drain barrier
        // (all P published) into CB 8; UNPACK scans it at the end of the step.
        if (!is_first_iter) {
            constexpr uint32_t pa_check_cb = 8;
            CircularBuffer(pa_check_cb).reserve_back(Sq_chunk_t);
            for (uint32_t g = 0; g < qktv_q_num_subblocks; ++g) {
                configure_single_tile_pack(pa_check_cb);
                reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT_stride, scale_fp32, true>(
                    pa_check_cb, prev.max, g, false, qktv_h, active_Sk, false, false);
            }
            CircularBuffer(pa_check_cb).push_back(Sq_chunk_t);
        }
#endif
#if defined(SDPA_PA) && defined(SDPA_PA_DENOM)
// MATH_FIDELITY exists only in the math-thread build, so this is used inside MATH(...) only.
#define PA_DENOM_FIDELITY (MATH_FIDELITY == MathFidelity::LoFi ? MathFidelity::HiFi2 : MATH_FIDELITY)
        // P-A denominator: l_chunk = P * 1 on the FPU (at least HiFi2: LoFi truncates P and biases l low), Float32
        // L1-accumulated into the denominator bank. Replaces the per-tile row-sum packs.
        {
            CircularBuffer(cb_col_identity).wait_front(1);
            matmul_block_init(cb_qkt_im, cb_col_identity, 0, 1, 4, KT_stride);
            MATH((llk_math_matmul_init<PA_DENOM_FIDELITY, MM_THROTTLE>(cb_qkt_im, cb_col_identity, 0, 1, 4)));
            pack_reconfig_data_format(cur.sum);
            configure_single_tile_pack(cur.sum);
            PACK((llk_pack_reconfig_l1_acc(is_first_iter ? 0 : 1)));
            for (uint32_t row = 0; row < Sq_chunk_t; row += 4) {
                const uint32_t rows = Sq_chunk_t - row < 4 ? Sq_chunk_t - row : 4;
                if (rows != 4) {
                    UNPACK((llk_unpack_AB_matmul_init(cb_qkt_im, cb_col_identity, 0, 1, rows, KT_stride)));
                    MATH((llk_math_matmul_init<PA_DENOM_FIDELITY, MM_THROTTLE>(cb_qkt_im, cb_col_identity, 0, 1, rows)));
                }
                tile_regs_acquire();
                for (uint32_t col = 0; col < active_Sk; ++col) {
                    UNPACK((llk_unpack_AB_matmul(
                        cb_qkt_im, cb_col_identity, row * KT_stride + col, 0, 1, rows, KT_stride)));
                    MATH((llk_math_matmul<PA_DENOM_FIDELITY, MM_THROTTLE>(0, 1, rows)));
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t r = 0; r < rows; ++r) {
                    pack_tile<true>(r, cur.sum, (row + r) * sdpa_sum_stride);
                }
                tile_regs_release();
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            pack_reconfig_data_format(cb_qkt_im);
            // Full no-MOP PV init on all threads: the MOP matmul above replaced unpack and math state that
            // the short PV reinit does not restore.
            recipe_mm_init(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, KT_stride);
        }
#endif

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
            // Rows are batched four at a time (the FP32 half-sync dest capacity). A Q
            // chunk that is not a multiple of four finishes with a two-row batch;
            // batching only changes which rows share a dest pass, not any element's sum.
            auto denominator_init = [&](uint32_t rows) {
                MATH((llk_math_matmul_init<denom_fidelity, MM_THROTTLE>(cb_qkt_im, cb_col_identity, 0, 1, rows)));
#ifdef SDPA_RECIPE_ACCURATE
                static_assert(
                    denom_fidelity == MathFidelity::HiFi2, "Phase-0/2 denominator requires the two-phase MOP");
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
            };
            denominator_init(4);
            configure_single_tile_pack(cur.sum);
            PACK((llk_pack_reconfig_l1_acc(inplace_numerator ? 1 : 0)));
            for (uint32_t row = 0; row < Sq_chunk_t; row += 4) {
                const uint32_t rows = Sq_chunk_t - row < 4 ? Sq_chunk_t - row : 4;
                if (rows != 4) {
                    UNPACK((llk_unpack_AB_matmul_init(cb_qkt_im, cb_col_identity, 0, 1, rows, KT_stride)));
                    denominator_init(rows);
                }
                tile_regs_acquire();
#ifndef SDPA_KO_DENOM
                for (uint32_t col = 0; col < active_Sk; ++col) {
                    UNPACK((llk_unpack_AB_matmul(
                        cb_qkt_im, cb_col_identity, row * KT_stride + col, 0, 1, rows, KT_stride)));
                    MATH((llk_math_matmul<denom_fidelity, MM_THROTTLE>(0, 1, rows)));
                }
#endif
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < rows; ++j) {
                    pack_tile<true>(j, cur.sum, row + j);
                }
                tile_regs_release();
            }
            MATH((llk_math_matmul_init_no_mop<MATH_FIDELITY, MM_THROTTLE>(
                cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h)));
        }

#endif
        auto normalize_row = [&](uint32_t& pushed, uint32_t sbh) {
            MaybeDeviceZoneScopedN(profiling_enabled, "ROW_NORM");
#ifndef SDPA_RECIPE_FP32
            if (is_first_iter) {
                group2_bootstrap_row(prev.out, out_cb, pushed * qktv_h, 0, sbh, vDHt);
            }
#endif
            PA_FIRST_FOLD(sbh)
            CircularBuffer(cur.sum).push_back(sbh * sdpa_sum_stride);
            CircularBuffer(out_cb).push_back(sbh * vDHt * sdpa_out_stride);
            normalize_row_streaming<
                profiling_enabled,
                vDHt,
                dst_size,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out>(
#if defined(SDPA_RECIPE_FP32) || defined(SDPA_PA)
                cur.sum, out_cb, sbh);
#else
                cur.sum, prev.out, sbh);
            CircularBuffer(out_cb).pop_front(sbh * vDHt * sdpa_out_stride);
#endif

            pushed++;
        };

        bool identity_corrections[Sq_chunk_t] = {};

        // Correct the previous row; normalization is guarded separately.
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
#if defined(SDPA_PA) && defined(SDPA_PA_SAFE)
            CircularBuffer(cb_exp_max_diff).wait_front(sbh);
            pa_fold(salad_row, sbh, false, pa_is_ident(salad_row));
            CircularBuffer(cb_exp_max_diff).pop_front(sbh);
            return;
#elif defined(SDPA_PA)
            // Reference max: nothing to fold; numerator and denominator accumulate in Float32 L1.
            CircularBuffer(cb_exp_max_diff).wait_front(sbh);
            CircularBuffer(cb_exp_max_diff).pop_front(sbh);
            return;
#endif
            PACK((llk_pack_reconfig_l1_acc(1)));
            {
                MaybeDeviceZoneScopedN(profiling_enabled, "S_CORR_FUSED");
                // ob_q_subblock=0: prev.out and cb_exp_max_diff are popped row-by-row (read from front).
                // sum_q_subblock=salad_row: prev.sum uses cumulative indexing (not popped per-row).
                {
#ifdef SDPA_RECIPE_FP32
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
                        identity_corrections[salad_row]);
#else
                    if (qktv_remainder_h == 0 || sbh == qktv_h) {
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
                            identity_corrections[salad_row],
                            is_last_iter || (group_k_index % 2 == 0),
                            (group_k_index % 2 == 1),
                            has_local);
                    } else if constexpr (qktv_remainder_h != 0) {
                        // Single-row tail group: row indices in units of one row.
                        salad_correct_fused<1, vDHt, dst_size>(
                            prev.out,
                            prev.sum,
                            cb_exp_max_diff,
                            out_cb,
                            cur.sum,
                            0,
                            salad_row * qktv_h,
                            w_salad * qktv_h,
                            is_last_iter,
                            identity_corrections[salad_row],
                            is_last_iter || (group_k_index % 2 == 0),
                            (group_k_index % 2 == 1),
                            has_local);
                    }
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

        // Rows 1..N-1: correction of the previous row overlaps the current PV matmul.
        constexpr uint32_t total_v_row_groups = qktv_q_num_subblocks;
        // No score exp follows. This restores the generic SFPU state (config register, ADDR_MOD_7 with no
        // DST increment, counters) that the SFPI correction exp assumes: the BALANCED/ACCURATE refine
        // leaves ADDR_MOD_7 at a DST increment of 4, and the FP32 correction does not reset it itself.
        exp_packthread_tile_init<EXP_APPROX_MODE>();
        for (uint32_t q_subblock = 1; q_subblock < total_v_row_groups; ++q_subblock) {
            MaybeDeviceZoneScopedN(profiling_enabled, "Softmax(Q@KT)@V");
#ifdef SDPA_TRISC2_SPIN
            // Perf research: RISC-only busy time on the pack thread, where TRISC2 RVV column math would run.
            PACK({
                const uint32_t spin_start = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
                while (reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - spin_start < SDPA_TRISC2_SPIN) {
                }
            })
#endif
            const uint32_t cur_h = pv_rows(q_subblock);
            uint32_t salad_row = q_subblock - 1;
            uint32_t w_salad = salad_row - pushed_rows;
            uint32_t w_q = q_subblock - pushed_rows;

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

            // V matmul for the current row group.
            {
                CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles - (qktv_h - cur_h) * KT_stride);
            }
            {
#ifdef SDPA_RECIPE_FP32
                if (inplace_numerator) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
#endif
#ifdef SDPA_PA
                PA_PV_BEGIN(out_cb, PA_PV_ACC(q_subblock));
#endif
                MaybeDeviceZoneScopedN(profiling_enabled, "QKT@V MM+Pack");
                uint32_t v_index_offset = 0;
                sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                    out_cb, out_cb);
                // See the q_subblock-0 V matmul above: active_Sk can be narrower than the physical
                // cb_qkt_im row stride, but the unpacker is configured for the physical layout.
                recipe_mm_reinit(cb_qkt_im, cb_v_in, false, qktv_subblock_w, cur_h, KT_stride);
                // Configure once before v_subblock loop; skip inside.
                configure_row_pack_width(out_cb, qktv_subblock_w);
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    const uint32_t qktv_in1_index = v_index_offset;
                    blocked_matmul_and_pack<false, vDHt, vDHt>(
                        cb_qkt_im,
                        cb_v_in,
                        out_cb,
                        qktv_in0_index_offset,
                        qktv_in1_index,
                        cur_h == qktv_h ? w_q : w_q * qktv_h,
#ifdef SDPA_RECIPE_FP32
                        v_subblock * qktv_subblock_w,
#else
                        v_subblock * qktv_subblock_w + PA_PV_OFF(q_subblock),
#endif
                        qktv_subblock_w,
                        cur_h,
                        active_Sk,
                        KT_stride,
                        /*skip_pack_configure=*/true);
                    v_index_offset += qktv_subblock_w;
                }
                sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
#ifdef SDPA_PA
                PA_PV_END(cb_qkt_im);
#endif
            }

            // SALAD corrections for previous group (always full, h=qktv_h) + row-by-row push
            if (!is_first_iter) {
                // Last main-loop iteration: hoist drain's sub_exp so both salads
                // (current row and drain row) chain back-to-back with one FPU init.
                if (q_subblock == total_v_row_groups - 1) {
                    constexpr uint32_t drain_h = last_h;
                    const uint32_t drain_salad_row = last_group;

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
                        pv_index(drain_salad_row),
                        drain_h,
                        sdpa_identity_flags[drain_salad_row]);
#endif
                    CircularBuffer(cb_exp_max_diff).push_back(drain_h);

                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        PA_FIRST_FOLD(qktv_h)
                        CircularBuffer(cur.sum).push_back(qktv_h * sdpa_sum_stride);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }

                    const uint32_t drain_w = last_group - pushed_rows;
                    salad_correct_row(drain_salad_row, drain_w, drain_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, drain_h);
                    } else {
                        PA_FIRST_FOLD(drain_h)
                        CircularBuffer(cur.sum).push_back(drain_h * sdpa_sum_stride);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                } else {
                    salad_correct_row(salad_row, w_salad, qktv_h);
                    if (is_last_iter) {
                        normalize_row(pushed_rows, qktv_h);
                    } else {
                        PA_FIRST_FOLD(qktv_h)
                        CircularBuffer(cur.sum).push_back(qktv_h * sdpa_sum_stride);
                        CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                }
            } else if (is_last_iter) {
                normalize_row(pushed_rows, qktv_h);
            } else {
#ifndef SDPA_RECIPE_FP32
                group2_bootstrap_row(prev.out, out_cb, salad_row * qktv_h, salad_row * qktv_h, qktv_h, vDHt);
#endif
                PA_FIRST_FOLD(qktv_h)
                CircularBuffer(cur.sum).push_back(qktv_h * sdpa_sum_stride);
                CircularBuffer(out_cb).push_back(qktv_h * vDHt * sdpa_out_stride);
                pushed_rows++;
            }

            qktv_in0_index_offset += cur_h * KT_stride;
            qktv_in0_wait_tiles += cur_h * KT_stride;
        }

        // Pipeline drain: SALAD for the last group
        {
            constexpr uint32_t drain_h = last_h;
            if constexpr (total_v_row_groups == 1) {
                // Single row group: the main loop never ran, so the drain performs the
                // full correction (as the legacy streaming kernel does).
                if (!is_first_iter) {
                    CircularBuffer(cb_exp_max_diff).reserve_back(drain_h);
#ifdef SDPA_RECIPE_FP32
                    identity_corrections[0] =
                        inplace_numerator || sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                                                 prev.max, cur.max, cb_exp_max_diff, pv_index(0), drain_h);
#else
                    identity_corrections[0] = sub_exp_first_col_blocks<profiling_enabled, scale_fp32>(
                        prev.max, cur.max, cb_exp_max_diff, pv_index(0), drain_h, sdpa_identity_flags[0]);
#endif
                    CircularBuffer(cb_exp_max_diff).push_back(drain_h);
                    salad_correct_row(0, 0, drain_h);
                }
                if (is_last_iter) {
                    normalize_row(pushed_rows, drain_h);
                } else {
#ifndef SDPA_RECIPE_FP32
                    if (is_first_iter) {
                        group2_bootstrap_row(prev.out, out_cb, 0, 0, drain_h, vDHt);
                    }
#endif
                    PA_FIRST_FOLD(drain_h)
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
                            prev.out, out_cb, last_group * qktv_h, last_group * qktv_h, drain_h, vDHt);
#endif
                        PA_FIRST_FOLD(drain_h)
                        CircularBuffer(cur.sum).push_back(drain_h * sdpa_sum_stride);
                        CircularBuffer(out_cb).push_back(drain_h * vDHt * sdpa_out_stride);
                        pushed_rows++;
                    }
                }
            }
        }

        // All rows pushed individually — no bulk push needed.

#if defined(SDPA_PA) && defined(SDPA_PA_PCHECK)
        if (!is_first_iter) {
            constexpr uint32_t pa_check_cb = 8;
            CircularBuffer(pa_check_cb).wait_front(Sq_chunk_t);
            UNPACK({
                uint32_t grew = 0;
                for (uint32_t t = 0; t < Sq_chunk_t; ++t) {
                    auto* p_max = reinterpret_cast<volatile uint32_t*>(
                        get_tile_l1_byte_address(get_operand_id(pa_check_cb), t));
                    for (uint32_t face = 0; face < 2; ++face) {
                        for (uint32_t r = 0; r < 16; ++r) {
                            // P >= 0: BF16 bits order like the values. 0x3380 = 2^-24 (growth > 4 octaves).
                            grew |= (p_max[face * 256 + r * 8] & 0x7fffu) > 0x3380u;
                        }
                    }
                }
                sdpa_pa_overflow = sdpa_pa_overflow | grew;
            })
            CircularBuffer(pa_check_cb).pop_front(Sq_chunk_t);
        }
#endif
        CircularBuffer(cb_v_in).pop_front(KT_stride * vDHt);
        CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT_stride);
    }
}

// One Q chunk over k_num_chunks K chunks: the online-softmax loop for recipes B-E. Called once per Q chunk
// by sdpa_standard_v2 (dense and joint). A caller may split one Q chunk's K range into several segments,
// setting final_segment on the last.
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
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
    uint32_t cb_normalized_out,
    bool independent_q_release = false>
ALWI void sdpa_segment_v2(RecipeAccumulatorState& state, uint32_t k_num_chunks, bool final_segment, bool release_q) {
    static_assert(
        Sq_chunk_t >= 1 && Sq_chunk_t <= kRecipeMaxQTiles && kRecipeValidKTiles<Sk_chunk_t> && DHt == vDHt &&
        DHt >= 1);
    ASSERT(k_num_chunks > 0);
    auto& prev = state.prev;
    auto& cur = state.cur;
#ifndef SDPA_RECIPE_FP32
    const uint32_t cb_out_im_A = prev.out;
    const uint32_t cb_out_im_B = cur.out;
#ifndef SDPA_PA
    if (state.processed_chunks == 0) {
        group2_initialize_root(cb_out_im_A, Sq_chunk_t * vDHt * sdpa_out_stride);
    }
#endif
#endif
    for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
#ifdef SDPA_PERF_PHASE_ZONES
        DeviceZoneScopedN("STEP");
#endif
#ifdef SDPA_RECIPE_K_PRIMARY_ROWS
        recipe_k_tile_offset = (state.processed_chunks + k_chunk) * Sk_chunk_t;
#endif
        const bool is_first = state.processed_chunks == 0 && k_chunk == 0;
        const bool last_local = k_chunk == k_num_chunks - 1;
        const bool is_last = final_segment && last_local;
        sdpa_inner_loop_step<
            sdpa_perf_zones,
            Sq_chunk_t,
            Sk_chunk_t,
            DHt,
            vDHt,
            scale_fp32,
            qkt_subblock_h,
            qkt_subblock_w,
            qktv_subblock_h,
            qktv_subblock_w,
            cb_q_in,
            cb_kt_in,
            cb_v_in,
            cb_qkt_im,
            cb_identity_scale_in,
            cb_exp_max_diff,
            cb_col_identity,
            cb_recip_scratch,
            cb_normalized_out,
            independent_q_release>(
            prev,
            cur,
            is_last,
            is_first
#ifndef SDPA_RECIPE_FP32
            ,
            state.processed_chunks + k_chunk,
            state.group_local_valid
#endif
            ,
            release_q && last_local);
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
#ifndef SDPA_PA
                sdpa_cb_pop_front_out_of_line(prev.sum, Sq_chunk_t * sdpa_sum_stride);
#endif
        }
#ifdef SDPA_PA
        // One Float32 denominator bank for the whole Q chunk: recycle it (its bytes survive in L1).
        if (!is_last) {
            sdpa_cb_pop_front_out_of_line(cur.sum, Sq_chunk_t * sdpa_sum_stride);
        }
#endif

        if (is_last) {
            sdpa_cb_pop_front_out_of_line(cur.max, Sq_chunk_t);
        } else {
            std::swap(prev, cur);
#ifndef SDPA_RECIPE_FP32
            prev.out = cb_out_im_A;
            cur.out = cb_out_im_B;
#endif
#ifdef SDPA_PA
            std::swap(prev.sum, cur.sum);
#endif
        }
    }
    state.processed_chunks += k_num_chunks;
}

template <uint32_t... Configuration>
void sdpa_standard_v2(
    uint32_t q_chunks_per_core,
    uint32_t k_num_chunks,
    uint32_t cb_out_im_A,
    uint32_t cb_out_im_B,
    uint32_t cb_max_A,
    uint32_t cb_max_B,
    uint32_t cb_sum_A,
    uint32_t cb_sum_B) {
    init_sdpa_streaming_semaphores();
    for (uint32_t q = 0; q < q_chunks_per_core; ++q) {
        RecipeAccumulatorState state{{cb_sum_A, cb_max_A, cb_out_im_A}, {cb_sum_B, cb_max_B, cb_out_im_B}};
        sdpa_segment_v2<Configuration...>(state, k_num_chunks, true, true);
    }
}
