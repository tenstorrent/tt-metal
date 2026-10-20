// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streaming SDPA compute helpers.
// Included by sdpa_recipe.cpp for the explicit Blackhole numerical recipes.
// Depends on primitives from compute_common.hpp (must be included first).

#pragma once

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp"
#include "recipe_config.hpp"
#include "lofi_scaling.hpp"

#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

// FP32 recipes: one numerator plane per query-tile row. BF16 recipes: each query-tile row holds the Float32
// numerator (plane 0), then a changed group's chunk PV (plane 1).
constexpr uint32_t sdpa_out_stride = recipe::fp32 ? 1 : 2;
// BALANCED's QK fidelity (its PV fidelity is the compute config's). On BF16 Q and K, HiFi3 measures the same
// error as HiFi4; HiFi2 drops Q's eighth significant bit.
constexpr auto kBalancedQkFidelity = MathFidelity::HiFi3;
// BALANCED runs its QK at kBalancedQkFidelity; ACCURATE and the BF16 recipes at the compute config's.
constexpr bool kBalancedQk = recipe::fp32 && !recipe::accurate;

// Blackhole's pack-to-unpack handshake overlaps row-max reduction with QK packing.
constexpr bool reduce_trigger_supported = true;

#include "sdpa_cb_protocol.hpp"
#include "fp32_state.hpp"

struct AccumulatorHalf {
    uint32_t sum, max, out;
};

// PACK -> UNPACK handshake: UNPACK continues once PACK's preceding L1 writes are visible.
ALWI void recipe_pack_to_unpack_barrier() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

// LoFi no-MOP matmuls bake the reuse-side source clear into the recorded replay image
// (reuse A when ct_dim >= rt_dim, else reuse B); mm_no_mop_reinit_short does not re-record it.
// When a reinit changes the reuse side (a one-tile-wide QK subblock followed by a wider PV
// subblock, or a single-row tail), MATH would clear the wrong source and UNPACK would wait
// forever. Reuse-B shapes (only new narrow geometries) always re-record, since other matmul
// inits (e.g. normalization) may have replaced the image; reuse-A shapes re-record only after a
// reuse-B image, so reuse-A-only builds skip the re-record.
// HiFi replays clear outside the image and need none of this.
static bool recipe_mm_reuse_a = true;  // FAST only
ALWI void recipe_mm_init(uint32_t in0, uint32_t in1, bool transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
    mm_no_mop_init_short(in0, in1, transpose, ct, rt, kt);
    if constexpr (recipe::lofi) {
        recipe_mm_reuse_a = ct >= rt;
    }
}
ALWI void recipe_mm_reinit(uint32_t in0, uint32_t in1, bool transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
    if constexpr (recipe::lofi && recipe::ring) {
        // FAST's ring kernels always re-record: one matmul init in MATH fits them in the kernel config buffer,
        // and their pack-bound chunks leave MATH the time.
        recipe_mm_init(in0, in1, transpose, ct, rt, kt);
    } else {
        if constexpr (recipe::lofi) {
            if (ct < rt || !recipe_mm_reuse_a) {
                recipe_mm_init(in0, in1, transpose, ct, rt, kt);
                return;
            }
        }
        mm_no_mop_reinit_short(in0, in1, transpose, ct, rt, kt);
    }
}

// Workaround for an SFPI compiler ICE (sfpi 7.85, gimple-rvtt-synth.cc:485 in rvtt_synth_renumber, MATH's
// blocked_matmul_and_pack): FAST with an odd Q chunk (single-row tail group) and QK/PV subblock widths 4/2 or 2/4
// (D64 with K128, D128 with K64, ...). Hiding the unfused PV loop's row count from constant propagation avoids it
// (no ICE over a Q 1-15 x K 1-30 x D 1-8 tile sweep); even chunks compile unchanged.
template <bool tail>
ALWI void recipe_opaque_tail_rows(uint32_t& rows) {
    if constexpr (tail) {
        MATH(asm volatile("" : "+r"(rows));)
    }
}

// Q chunks up to 1024 rows (32 tiles). Any tile-aligned K chunk: the QK subblock width (recipe::qk_w,
// host-chosen to divide it) sets the reduction pieces.
constexpr uint32_t kRecipeMaxQTiles = 32;

struct RecipeAccumulatorState {
    AccumulatorHalf prev, cur;
    uint32_t processed_chunks = 0;
};

// Sentinel for "no CB" — beyond the valid 0-31 range.
constexpr uint32_t INVALID_CB = 32;
static bool sdpa_skip_prev_sum_pop = false;  // FP32 recipes only
// The FP32 recipes keep their row maxima in FP32 tiles (CBs 10, 11): BF16 maxima of FP32 scores can sit
// beyond the score exp's range (about 21.5 below and 0.7 above 0) once |scale * s| exceeds about 1e4, which
// zeroes a whole row of P. The max reduce reads scores through srcA (19 bits), so the maxima are exact on
// that grid, and every FPU read of them (S - m, m_old - m_new) is exact as well.
// Word offset of row r's maximum (column 0) in an FP32 tile: 16 x 16 word faces, rows 16-31 in face 2.
constexpr uint32_t sdpa_max_word(uint32_t r) { return (r / 16) * 512 + (r % 16) * 16; }
// Blackhole benefits from blocked packing at width four.
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 4;
ALWI bool should_use_blocked_pack_width(uint32_t pack_width) { return pack_width >= MIN_BLOCKED_PACK_TILES; }

template <typename... Args>
ALWI void sdpa_stream_reconfig(Args... args) {
    if constexpr (recipe::fp32) {
        reconfig_data_format_skip_int8(args...);
    } else {
        reconfig_data_format(args...);
    }
}

template <typename... Args>
ALWI void sdpa_stream_reconfig_srca(Args... args) {
    if constexpr (recipe::fp32) {
        reconfig_data_format_srca_skip_int8(args...);
    } else {
        reconfig_data_format_srca(args...);
    }
}

static sdpa::streaming::Fp32PackConfig sdpa_pack;  // FP32 recipes only
// PACK thread only (pack_src_format / pack_dst_format exist in the PACK build).
ALWI void sdpa_cached_pack_format(uint32_t cb) {
    PACK({
        if (sdpa_pack.format_cb == INVALID_CB || pack_src_format[sdpa_pack.format_cb] != pack_src_format[cb] ||
            pack_dst_format[sdpa_pack.format_cb] != pack_dst_format[cb]) {
            pack_reconfig_data_format(cb);
        }
        sdpa_pack.format_cb = cb;
    })
}

template <uint32_t old_cb, uint32_t new_cb>
ALWI void sdpa_maybe_pack_reconfig_data_format() {
    if constexpr (recipe::fp32) {
        PACK((sdpa_cached_pack_format(new_cb)));
    } else {
        PACK({
            if constexpr (pack_dst_format[old_cb] != pack_dst_format[new_cb]) {
                pack_reconfig_data_format(old_cb, new_cb);
            }
        })
    }
}

// The unpack format tables exist in the UNPACK and MATH builds only; PACK never reconfigures the unpacker.
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
    if constexpr (
        sdpa_unpack_format_changed<srca_old_cb, srca_new_cb>() ||
        sdpa_unpack_format_changed<srcb_old_cb, srcb_new_cb>()) {
        sdpa_stream_reconfig(srca_old_cb, srca_new_cb, srcb_old_cb, srcb_new_cb);
    }
}

template <uint32_t srca_old_cb, uint32_t srca_new_cb, uint32_t srcb_old_cb, uint32_t srcb_new_cb>
ALWI void sdpa_maybe_reconfig_data_format(uint32_t runtime_srca_old_cb, uint32_t runtime_srcb_old_cb) {
    if constexpr (
        sdpa_unpack_format_changed<srca_old_cb, srca_new_cb>() ||
        sdpa_unpack_format_changed<srcb_old_cb, srcb_new_cb>()) {
        sdpa_stream_reconfig(runtime_srca_old_cb, srca_new_cb, runtime_srcb_old_cb, srcb_new_cb);
    }
}

// Keep this out-of-line even on BH: repeated pack-width configuration sites
// inflate SDPA streaming code size more than this call costs in measured cases.
static __attribute__((noinline, noclone)) void configure_pack_width(uint32_t cb, uint32_t pack_width) {
    if constexpr (recipe::fp32) {
        PACK((sdpa_cached_pack_format(cb)));
        PACK(if (sdpa_pack.width == pack_width) { return; })
        PACK(sdpa_pack.width = pack_width;)
    }
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
    if constexpr (!recipe::fp32) {
        if (out_cb == 8 || out_cb == 9) {
            row_stride *= 2;
        }
    }
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
    for (uint32_t inner = 0; inner < inner_dim; ++inner) {
        if constexpr (kBalancedQk && transpose) {
            UNPACK((llk_unpack_AB_matmul(in0_cb, in1_cb, in0_index, in1_index, subblock_w, subblock_h, matmul_stride)));
            MATH((llk_math_matmul_no_mop<kBalancedQkFidelity, MM_THROTTLE>(
                in0_cb, in1_cb, dst_index, subblock_w, subblock_h)));
        } else {
            matmul_block_no_mop(
                in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, matmul_stride);
        }
        in0_index++;
        in1_index += in1_stride;
    }
    tile_regs_commit();

    tile_regs_wait();
    if constexpr (transpose && recipe::ring) {
        if (recipe_k_valid_rows < recipe_k_chunk_rows) {
            mask_recipe_tail(out_col_offset, subblock_w, subblock_h);
        }
    } else if constexpr (transpose && recipe::k_tail) {
        mask_recipe_tail(out_col_offset, subblock_w, subblock_h);
    }
    if (!skip_pack_configure) {
        configure_row_pack_width(out_cb, subblock_w);
    }
    pack_contiguous_rows_nocfg(
        out_cb, row_subblock_idx * subblock_h, subblock_h, out_num_cols, out_col_offset, subblock_w);
    tile_regs_release();
}

// Row maxima combine the current QK block with the previous online maximum.
//
// Reference-max state (STANDARD, FAST): P = exp(scale * (s - m_ref)) with m_ref a row's
// reference max, kept until the true row max exceeds it by theta. O and l live in Float32 L1 (the
// packer accumulates PV and row sums onto them exactly); a chunk that raises m_ref rescales them once.
#define REF_MAX_PV_BEGIN(pv_cb, acc)                         \
    do {                                                \
        pack_reconfig_data_format(pv_cb);               \
        PACK((llk_pack_reconfig_l1_acc((acc) ? 1 : 0))); \
    } while (0)
#define REF_MAX_PV_END(restore_cb)                  \
    do {                                       \
        PACK((llk_pack_reconfig_l1_acc(0)));   \
        pack_reconfig_data_format(restore_cb); \
    } while (0)
// This chunk's BF16 row sums, folded into the Float32 l state per row group.
constexpr uint32_t kRefMaxChunkSumCb = 12;
// Exp headroom tau = 28 ln 2: the fast exp's additive constant is lowered by exactly 28 octaves
// (256 * 28 = 7168 INT16 steps), so P is 2^-28 times the unshifted P bit for bit and the exp's error
// pattern is unchanged. Its INT16 grid saturates about 0.72 above zero, so P stays in range while the
// row max exceeds m_ref by less than tau.
constexpr uint32_t kRefMaxExpOctaves = 28;
// Rescale threshold: keep m_ref until the row max exceeds it by theta (natural-log units of the
// scaled scores). theta + 0.72 must stay below tau, so P never saturates.
constexpr float kRefMaxTheta = 16.0f * 0.69314718055994531f;
// Fused chunks (recipe::fused_active: STANDARD and FAST without an attn_mask), see sdpa_fused_chunk.
// One BF16 tile: a row group's saturation check (max of its chunk row-sum tiles).
constexpr uint32_t kFusedCheckCb = 31;
// Per Q tile row, QK-subblock-width partial row-sum tiles: each subblock's P is L1-accumulated onto them
// with the same blocked row pack as P itself.
constexpr uint32_t kFusedSumCb = 30;
// QK-subblock-width copies of -e0 (column 0 = -1, transposed by the QK unpack into row 0) in K's format:
// one extra QK inner step [Q | M] x [K^T ; -e0] subtracts m_ref (M: the reference-max tile, m in column 0).
constexpr uint32_t kFusedNegUnitCb = 29;
// Redo a group whose 16-term partial row sums reach 0.5: P saturates near 1.66 at this scale, and an
// unsaturated partial that large holds a P within a few octaves of saturation.
constexpr uint16_t kFusedRedoSumBf16 = 0x3F00;
// SFPI code exists in the MATH and PACK builds only.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// dest tile 0: max(m_ref, rowmax) from the reduce; tile `group`: m_ref. Keep m_ref unless exceeded by theta.
template <uint32_t scale_fp32, int group>
inline void calculate_sdpa_ref_max_select() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float theta = kRefMaxTheta / __builtin_bit_cast(float, scale_fp32);
    // Called with VectorMode::C: only column 0 (the row maxima) is read by any consumer.
    for (int i = 0; i < 4; ++i) {
        sfpi::vFloat m = sfpi::dst_reg[0];
        sfpi::vFloat r = sfpi::dst_reg[32 * group];
        v_if(m <= r + theta) { sfpi::dst_reg[0] = r; }
        v_endif;
        sfpi::dst_reg += 2;
    }
}
// Fused chunks: round the reference max toward zero to 7 significant bits (clear the lowest BF16 mantissa bit):
// the fused QK reads m_ref through srcB, which keeps 7 at LoFi and HiFi2, so every chunk subtracts the same m_ref.
inline void calculate_sdpa_trunc7() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 4; ++i) {
        sfpi::vFloat m = sfpi::dst_reg[0];
        sfpi::vInt bits = sfpi::as<sfpi::vInt>(m) & sfpi::vInt(~0x10000);
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(bits);
        sfpi::dst_reg += 2;
    }
}
// dest tiles [0, w): state; [w, 2w): this chunk's term; tile 2w: column-broadcast correction c.
// state = state * c + term.
template <int w>
inline void calculate_sdpa_rescale_add() {
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

// The K-chunk step (recipe_step.inc): cold with fused chunks, hot otherwise.
#include "recipe_fused_chunk.hpp"
namespace recipe_cold_step {
#define SDPA_RECIPE_COLD __attribute__((noinline, optimize("Os")))
#include "recipe_step.inc"
#undef SDPA_RECIPE_COLD
}  // namespace recipe_cold_step
namespace recipe_hot_step {
#define SDPA_RECIPE_COLD
#include "recipe_step.inc"
#undef SDPA_RECIPE_COLD
}  // namespace recipe_hot_step

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
    if constexpr (recipe::fused_active) {
        recipe_cold_step::sdpa_segment_v2<
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
            independent_q_release>(state, k_num_chunks, final_segment, release_q);
    } else {
        recipe_hot_step::sdpa_segment_v2<
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
            independent_q_release>(state, k_num_chunks, final_segment, release_q);
    }
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
