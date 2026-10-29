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
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/ring_joint_ksplit.hpp"

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#endif
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// reduce_trigger uses a packer->unpacker semaphore handshake to start the reduce early and skip the
// input CB wait. Quasar has no such handshake, so it stays disabled there and the normal CB
// synchronization is kept (see can_reduce_trigger below).
#ifdef ARCH_QUASAR
constexpr bool reduce_trigger_supported = false;
#else
constexpr bool reduce_trigger_supported = true;
#endif

// SDPA_MATMUL_FIDELITY overrides the fidelity of the QK^T and softmax @ V matmuls. A LoFi replay image is
// specific to the matmul's shape, so each PV setup re-records it instead of reusing the last one whenever a LoFi
// image can be live: a LoFi override, or a LoFi compute config, whose normalize_row records a LoFi image between
// two PV setups. HiFi levels all record the same image, so HiFi over HiFi can reuse it.
#ifdef SDPA_MATMUL_FIDELITY
constexpr bool sdpa_matmul_fidelity_set = true;
constexpr MathFidelity sdpa_matmul_fidelity = static_cast<MathFidelity>(SDPA_MATMUL_FIDELITY);
#else
constexpr bool sdpa_matmul_fidelity_set = false;
constexpr MathFidelity sdpa_matmul_fidelity = MathFidelity::LoFi;
#endif
// The compute config's fidelity as seen by every TRISC (MATH_FIDELITY exists only on math and pack).
#ifdef SDPA_COMPUTE_LOFI
constexpr bool sdpa_compute_lofi = true;
#else
constexpr bool sdpa_compute_lofi = false;
#endif

ALWI void sdpa_mm_init(uint32_t in0_cb, uint32_t in1_cb, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    if constexpr (sdpa_matmul_fidelity_set) {
        mm_no_mop_init_short_fidelity<sdpa_matmul_fidelity>(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        mm_no_mop_init_short(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    }
}

ALWI void sdpa_mm_reinit(
    uint32_t in0_cb, uint32_t in1_cb, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    if constexpr (sdpa_matmul_fidelity_set) {
        mm_no_mop_reinit_short_fidelity<sdpa_matmul_fidelity>(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        mm_no_mop_reinit_short(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    }
}

ALWI void sdpa_pv_mm_setup(
    uint32_t in0_cb, uint32_t in1_cb, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    if constexpr ((sdpa_matmul_fidelity_set && sdpa_matmul_fidelity == MathFidelity::LoFi) || sdpa_compute_lofi) {
        sdpa_mm_init(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        sdpa_mm_reinit(in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim);
    }
}

ALWI void sdpa_matmul_block_no_mop(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t in0_index,
    uint32_t in1_index,
    uint32_t idst,
    bool transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    if constexpr (sdpa_matmul_fidelity_set) {
        matmul_block_no_mop_fidelity<sdpa_matmul_fidelity>(
            in0_cb, in1_cb, in0_index, in1_index, idst, transpose, ct_dim, rt_dim, kt_dim);
    } else {
        matmul_block_no_mop(in0_cb, in1_cb, in0_index, in1_index, idst, transpose, ct_dim, rt_dim, kt_dim);
    }
}

// Set by ring joint SDPA; other kernels that include this header keep the one-segment capacity.
#ifdef SLIDING_MAX_SOURCE_RANGES
constexpr uint32_t sliding_q_plan_max_source_ranges = SLIDING_MAX_SOURCE_RANGES;
#else
constexpr uint32_t sliding_q_plan_max_source_ranges =
    ttnn::operations::transformer::sdpa::ring_joint::sliding_q_work_plan_source_ranges(
        ttnn::operations::transformer::sdpa::ring_joint::sliding_max_halo_hops, 1);
#endif

// Set by ring joint SDPA: put the local slab first only when the halo spans several hops (see the program factory).
#ifdef SLIDING_LOCAL_FIRST
constexpr bool sliding_local_first = SLIDING_LOCAL_FIRST;
#else
constexpr bool sliding_local_first = false;
#endif

// Template-driven profiling: MaybeDeviceZoneScopedN(ENABLED, name)
// When ENABLED=true: RAII profileScope writes timestamps (same as DeviceZoneScopedN)
// When ENABLED=false: empty struct, zero overhead (compiler eliminates entirely)
#if defined(PROFILE_STREAMING)
// Not supported by the streaming profiler: these template-gated zones use the DRAM profiler's hash ids.
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

// Ring joint SDPA: the per-phase zones of the inner step are compiled out unless the factory defines
// SOL_STEP_PROFILE (TT_SOL_STEP_PROFILE=1), and they only emit under the device profiler (PROFILE_KERNEL).
#if defined(SOL_STEP_PROFILE) && defined(PROFILE_KERNEL)
constexpr bool kRingStepProfiling = true;
#else
constexpr bool kRingStepProfiling = false;
#endif
// Fine zones (per key subblock: Q@KT, SUB/EXP, QKT@V, rescale, normalize) add enough TRISC code to push the
// sparse build past the kernel config buffer; TT_SOL_STEP_PROFILE=2 (SOL_STEP_PROFILE_FINE) enables them. The
// default (=1) keeps only the two phase zones, Softmax(Q@KT) and Softmax(Q@KT)@V, one each per Q subblock / row
// group, which is enough to see step length against the gaps between steps.
#if defined(SOL_STEP_PROFILE_FINE)
constexpr bool kRingStepProfilingFine = true;
#else
constexpr bool kRingStepProfilingFine = false;
#endif
// Levels (TT_SOL_STEP_PROFILE): 1 = one zone per visit (RING_STEP around a computed chunk, RING_DRAIN around a
// dropped one), so the ~10-event per-core profiler buffer holds ~10 visits and the gaps between them are
// measurable; 2 = plus the two phase zones per Q subblock / row group; 3 = plus the per-key-subblock zones.
#if defined(SOL_STEP_PROFILE_PHASES) && defined(PROFILE_KERNEL)
constexpr bool kRingStepProfilingPhases = true;
#else
constexpr bool kRingStepProfilingPhases = false;
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
    // K chunks the last single-Q-chunk sdpa_ring_v2 call accumulated (0: it left no state in prev).
    uint32_t last_call_k_chunks = 0;
    // Sliding only: work items in the last Q chunk's plan, which a K split divides into bands.
    uint32_t sliding_plan_k_chunks = 0;
};

// Ring-streaming lightweight-mask context. Field NAMES match LightweightMaskContext so sdpa_ring_v2's
// generic `lw_mask.<field>` reads work for either type, BUT the compile-time-constant fields are
// template params (static constexpr → zero per-instance storage), leaving only the per-ring-iter
// runtime fields on the stack. Shrinks the live lw_mask object ~56 B → ~20 B on kernel_main's frame.
// The two partial columns are runtime fields because they may come from a device tensor per dispatch;
// their CB tile INDICES stay compile-time.
// Only the ring_joint_sdpa producer uses it; exp_ring keeps the plain LightweightMaskContext —
// sdpa_ring_v2's MaskCtx template param is deduced per caller.
template <
    uint32_t NeginfTileIdx,
    uint32_t CausalDiagTileIdx,
    uint32_t LocalNPaddedTiles,
    uint32_t GlobalNPartialTileIdx,
    uint32_t JointLPartialTileIdx,
    uint32_t StraddleMaskChunkId>
struct RingStreamingMaskCtx {
    // Per-ring-iter runtime fields (the only ones needing stack storage):
    bool is_causal = false;
    uint32_t global_n_padded_tiles = 0;
    uint32_t joint_n_padded_tiles = 0;
    uint32_t straddle_num_padded_tiles = 0;
    uint32_t global_n_partial_col = 0;
    uint32_t joint_l_partial_col = 0;
    // Compile-time-constant fields (no per-instance storage):
    static constexpr uint32_t neginf_tile_idx = NeginfTileIdx;
    static constexpr uint32_t causal_diag_tile_idx = CausalDiagTileIdx;
    static constexpr uint32_t primary_diag_tile_idx = CausalDiagTileIdx;
    static constexpr uint32_t local_n_padded_tiles = LocalNPaddedTiles;
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
// WH keeps 8: blocked pack reconfiguration costs more than it saves at smaller widths.
#ifdef ARCH_BLACKHOLE
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 2;
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

// kPackDoneMax: PACK_DONE's counter capacity. A SEMPOST at the maximum is lost (t6_semaphore_post does not stall
// on it), so it must cover the most tokens a path ever has outstanding: 2 for the streaming paths, 15 (the 4-bit
// field's limit) for the Sol resident step, whose interleaved exp / P@V stages post one token per S5 batch.
template <uint32_t kPackDoneMax = 2>
ALWI void init_sdpa_streaming_semaphores() {
    static_assert(kPackDoneMax >= 1 && kPackDoneMax <= 15, "SEMINIT's max value is a 4-bit field");
    // reduce_trigger runs the QK row-max reduce as a split MOP gated by a PACK->UNPACK handshake on
    // two T6 tokens (firmware inits neither). FPU_SFPU, posted after pack + mask + push, gates run()#2
    // (and run()#1 on the non-overlap path). UNPACK_MATH_DONE (the first-half token) is borrowed (unused elsewhere
    // in SDPA): PACK posts it early, once the first half is committed, to gate run()#1 on the overlap
    // path so it overlaps the second-half pack.
    PACK((t6_semaphore_init(semaphore::FPU_SFPU, 0, 1)));
    PACK((t6_semaphore_init(semaphore::UNPACK_MATH_DONE, 0, 1)));
    // PACK_DONE carries pack->unpack "in-place row committed" tokens. The pipelined in-place V drain
    // keeps up to two outstanding (column blocks kt and kt+1); every other streaming user posts and takes one.
    PACK((t6_semaphore_init(semaphore::PACK_DONE, 0, kPackDoneMax)));
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
        ckernel::sfpu::calculate_recip_first_column<DST_ACCUM_MODE>();
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
        sdpa_matmul_block_no_mop(
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
 * Output column vd is a matmul chain over K^T row vd (in1 base vd*KT_stride, inner stride 1), so
 * adjacent output columns are KT_stride tiles apart in the CB. The srcA per-tile address step is
 * widened to KT_stride tiles for the duration of the call, which lets one matmul call reuse the
 * score tile across a DST-sized batch of columns, like blocked_matmul_and_pack does for Q@K^T.
 *
 * Loops: outer walks columns in DST-sized batches; inner accumulates the batch over inner_dim score
 * tiles starting at inner_start. Each batch is packed out in one go (the caller enables L1
 * accumulation when it splits the inner dimension across calls).
 */
template <uint32_t vDHt, uint32_t dst_size, uint32_t subblock_h>
void inplace_v_matmul_pack_batched(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t inner_dim,
    uint32_t KT_stride,
    uint32_t inner_start = 0) {
    // Each output column is written column-major into DST (column c at c*subblock_h), but the
    // pack below reads DST row-major; the two orderings only coincide when subblock_h==1, which
    // kt_inplace_v guarantees via Sq_chunk_t==1. Enforce it so this can't silently corrupt if
    // reused with multi-tile Q.
    static_assert(subblock_h == 1, "inplace_v_matmul_pack_batched requires single-tile Q (subblock_h==1)");
    // One matmul call covers a DST-sized batch of output columns: the score tile is unpacked once
    // (srcB) and the MOP streams one V tile per column (srcA). V column vd lives in K^T row vd, so
    // consecutive columns are KT_stride tiles apart: stretch the srcA per-tile address step to a
    // K^T row for the duration of the V matmul, then restore it.
    constexpr uint32_t cols_per_batch = vDHt < dst_size / subblock_h ? vDHt : dst_size / subblock_h;
    sdpa_pv_mm_setup(in0_cb, in1_cb, false, cols_per_batch, subblock_h, KT_stride);
    matmul_set_in1_column_stride(in1_cb, KT_stride);
    configure_row_pack_width(out_cb, cols_per_batch);
    for (uint32_t vs0 = 0; vs0 < vDHt; vs0 += cols_per_batch) {
        const uint32_t cols = vDHt - vs0 < cols_per_batch ? vDHt - vs0 : cols_per_batch;
        if constexpr (vDHt % cols_per_batch != 0) {
            if (cols != cols_per_batch) {
                sdpa_pv_mm_setup(in0_cb, in1_cb, false, cols, subblock_h, KT_stride);
                matmul_set_in1_column_stride(in1_cb, KT_stride);
                configure_row_pack_width(out_cb, cols);
            }
        }
        tile_regs_acquire();
        uint32_t in0_index = in0_index_start + inner_start;
        uint32_t in1_index = vs0 * KT_stride + inner_start;
        for (uint32_t inner = 0; inner < inner_dim; ++inner) {
            sdpa_matmul_block_no_mop(in0_cb, in1_cb, in0_index, in1_index, 0, false, cols, subblock_h, KT_stride);
            in0_index++;
            in1_index++;
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_contiguous_rows_nocfg(out_cb, 0, subblock_h, vDHt, vs0, cols);
        tile_regs_release();
    }
    matmul_set_in1_column_stride(in1_cb, 1);
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

    reduce_block_max_row_init_runtime(out_cb, reduce_cols, in0_cb, scale_cb, respect_trigger);
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

    sub_init(in0_cb, in1_cb);

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

    mul_bcast_cols_init(out_in_cb, bcast_cb);

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
    uint32_t cb_attention_sink = INVALID_CB,
    bool sink_per_row = false>
static __attribute__((noinline, noclone)) void normalize_row_streaming(
    uint32_t cur_sum_cb,
    uint32_t cur_out_cb,
    uint32_t sbh,
    [[maybe_unused]] uint32_t cur_max_cb_rt = 0,
    [[maybe_unused]] uint32_t sink_row_offset = 0) {
    // Dense SDPA supplies one scalar tile; sparse SDPA supplies a first-column vector of head
    // scalars per tile row. Fold exp((sink - max)*scale) into the col-reduced denominator (DST[0]).
    if constexpr (use_attention_sink) {
        CircularBuffer(cb_attention_sink).wait_front(sink_per_row ? sink_row_offset + sbh : 1);
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
                // max - sink with a negated scale is equivalent and avoids expanding the sink
                // scalar to a first-column vector for every Q tile.
                // The sink can have its own format (e.g. bfloat8_b): unpack from max/sink, then restore
                // the matmul formats that the next row's folded reconfig assumes. scratch_cb stands in
                // for cur_max: every caller allocates both in the intermediate format.
                sdpa_maybe_reconfig_data_format<col_identity_cb, scratch_cb, scratch_cb, cb_attention_sink>();
                if constexpr (sink_per_row) {
                    // Sparse SDPA packs different heads in each row; both inputs hold a first-column vector.
                    sub_init(cur_max_cb_rt, cb_attention_sink);
                    sub_tiles(cur_max_cb_rt, cb_attention_sink, sink_row_offset + s, sink_row_offset + s, 1);
                } else {
                    sub_bcast_scalar_init(cur_max_cb_rt, cb_attention_sink);
                    sub_tiles_bcast_scalar(cur_max_cb_rt, cb_attention_sink, sink_row_offset + s, 0, 1);
                }
                sdpa_maybe_reconfig_data_format<scratch_cb, col_identity_cb, cb_attention_sink, scratch_cb>();
                // The custom first-column exp needs generic unary SFPU addrmod state, but not the
                // Blackhole approximate exp_init macro/replay setup used by exp_tile<true>.
                MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
                constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
                constexpr uint16_t negated_scale_bf16 = scale_bf16 ^ 0x8000;
                MATH((exp_tile_first_column<EXP_APPROX_MODE, negated_scale_bf16>(1)));
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
#ifdef ARCH_BLACKHOLE
            recip_tile_init();
            MATH((recip_tile(0 /*dst_index*/, VectorMode::C)));
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
            mul_bcast_cols_init(cur_out_cb, scratch_cb);
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
    uint32_t kv_pad_region_Nt = 0,
    uint32_t kv_pad_chunk_size_t = 0,
    uint32_t kv_pad_kv_local_padded_Nt = 0,
    uint32_t sliding_window_size = 0,
    bool circular_kv_cache = false,
    // Split-KV grouped sources: packed K columns follow packed_mask's runs instead of one local range.
    bool packed_kv_enabled = false>
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
    const KVPadRotationContext& kv_pad_rotation = {},
    [[maybe_unused]] const ring_joint::PackedKVChunkMask& packed_mask = {}) {
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
                const bool packed_columns = packed_kv_enabled && packed_mask.packed();
                const uint32_t mask_cols =
                    packed_columns ? active_Sk : (kv_pad_rotation_enabled ? num_cols : active_Sk);
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

                if (packed_columns) {
                    uint32_t begin = 0;
                    for (uint32_t run = 0; run < packed_mask.run_count; ++run) {
                        const auto interval = packed_mask.runs[run];
                        const uint32_t count = interval.column_end - begin;
                        // Each run is contiguous globally. Stamp at most one diagonal,
                        // then its masked suffix; no per-column division or modulo.
                        const uint32_t visible_end = q_pos_u32 < kv_pad_rotation.logical_tile_count
                                                         ? q_pos_u32
                                                         : kv_pad_rotation.logical_tile_count;
                        uint32_t visible =
                            visible_end > interval.global_start_tile ? visible_end - interval.global_start_tile : 0;
                        visible = visible < count ? visible : count;
                        uint32_t masked_begin = begin + visible;
                        if (q_pos_u32 >= interval.global_start_tile && q_pos_u32 < interval.global_start_tile + count &&
                            q_pos_u32 < kv_pad_rotation.logical_tile_count) {
                            l1_acc_single_tile(mask_cb, primary_diag_idx, out_cb, row_offset + masked_begin);
                            ++masked_begin;
                        }
                        l1_acc_neginf_cols(mask_cb, out_cb, row_offset, masked_begin, interval.column_end, neginf_idx);
                        begin = interval.column_end;
                    }
                } else if constexpr (kv_pad_rotation_enabled) {
                    for (uint32_t col = 0; col < mask_cols; col++) {
                        const uint32_t local_k_tile = kv_pad_rotation.k_local_start_tile + col;
                        if (local_k_tile >= kv_pad_kv_local_padded_Nt) {
                            l1_acc_single_tile(mask_cb, neginf_idx, out_cb, row_offset + col);
                            continue;
                        }
                        // Circular cache: this chunk's absolute K origin (sliding chunks never
                        // straddle a slab). Inverting local rows is ambiguous once several chunk groups
                        // alias one local slab; unbounded caches keep the inversion.
                        const uint32_t k_pos_u32 =
                            circular_kv_cache ? k_start_tile + col
                                              : chunked_kv_global_tile_for_local<kv_pad_chunk_size_t, kv_pad_region_Nt>(
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
                    // Each cache slab belongs to a different global chunk. A wide K chunk can
                    // cross several slabs, so apply the gap at every boundary, not just the first.
                    if (apply_causal) {
                        for (uint32_t col = 0; col < mask_cols; col++) {
                            int32_t k_pos = static_cast<int32_t>(k_start_tile) + static_cast<int32_t>(col);
                            if (col >= straddle_col) {
                                uint32_t crossed_slabs = 1;
                                if constexpr (kv_pad_region_Nt > 0 && num_cols > kv_pad_region_Nt) {
                                    crossed_slabs += (col - straddle_col) / kv_pad_region_Nt;
                                }
                                k_pos += static_cast<int32_t>(crossed_slabs * straddle_jump);
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
        reconfig_data_format_srca(cb_qkt_im, cb_mask_in);
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

// Packed split-KV counterpart of sdpa_first_half_unmasked. Columns [0, active_Sk/2) are unmasked
// for every row of the Q subblock iff each of their global K tiles is strictly below the first
// row's Q tile (no diagonal or suffix stamp) and below the logical length (no tail stamp).
// KV-pad rotation remaps Q rows, so it keeps the conservative answer like the contiguous proof.
template <bool kv_pad_rotation_enabled, uint32_t sliding_window_size, bool use_provided_mask>
inline bool sdpa_packed_first_half_unmasked(
    bool no_mask_this_iter,
    bool apply_mask,
    uint32_t lw_partial_tile_idx,
    const ring_joint::PackedKVChunkMask& packed_mask,
    uint32_t first_q_tile,
    uint32_t logical_tile_count,
    uint32_t active_Sk) {
    if (no_mask_this_iter) {
        return true;
    }
    if constexpr (kv_pad_rotation_enabled || sliding_window_size > 0 || use_provided_mask) {
        return false;
    }
    if (apply_mask && lw_partial_tile_idx > 0) {
        return false;
    }
    const uint32_t visible_end = first_q_tile < logical_tile_count ? first_q_tile : logical_tile_count;
    const uint32_t half = active_Sk / 2;
    const ring_joint::PackedKVMaskRun* runs = packed_mask.runs;
    uint32_t begin = 0;
    for (uint32_t run = 0; run < packed_mask.run_count && begin < half; ++run) {
        const uint32_t end = runs[run].column_end < half ? runs[run].column_end : half;
        // Runs are globally contiguous, so the run's last first-half column holds its largest tile.
        if (runs[run].global_start_tile + (end - begin) > visible_end) {
            return false;
        }
        begin = runs[run].column_end;
    }
    return true;
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
    uint32_t kv_pad_region_Nt = 0,
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
    bool has_q_base_tiles = false,
    // Circular sliding KV cache: mask K origins come from the work plan, not from
    // inverting local cache rows (ambiguous once chunk groups alias one local slab).
    bool circular_kv_cache = false,
    // Split-KV grouped sources; off for every other caller, so its branches compile out.
    bool packed_kv_enabled = false,
    // Sol-Attn pooled chunks: L1-accumulate one tile row of column bias (cb_col_bias, Sk_chunk_t tiles,
    // identical rows) onto every Q tile row of the QK^T tile before the max/exp.
    bool use_col_bias = false,
    uint32_t cb_col_bias = INVALID_CB>
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
    const uint32_t q_base_tiles = 0,
    // Packed split-KV K chunk: its column layout and mask runs. Contiguous for every other caller.
    [[maybe_unused]] const ring_joint::PackedKVChunkMask& packed_mask = {},
    // Sol-Attn: this chunk is a pooled chunk carrying a column bias.
    [[maybe_unused]] const bool apply_col_bias = false,
    // Resident mode: several Q chunks consume the same K/V entry in turn, so only the last of them pops it.
    const bool hold_kv = false,
    // Sol-Attn resident mode: block-run addressing. When set, K^T sub-block `s` (actual_sbw tiles wide) is
    // read from physical K^T column tile run_cols[s] of the entry instead of column s * actual_sbw, and the
    // matching V rows from row run_cols[s]; the scores stay packed at the compacted columns [0, active_Sk), so
    // the max / exp / sum / correction work unchanged on the kept blocks only. active_Sk = kept sub-blocks x
    // actual_sbw. Needs no masking inside the entry (the caller lists only full, kept blocks).
    const uint8_t* run_cols = nullptr) {
    // Callers guarantee active_Sk is evenly divisible by actual_sbw (via largest_factor_le).
    const uint32_t kt_num_full_subblocks = active_Sk / actual_sbw;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t in0_block_w = DHt;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qkt_subblock_h;
    constexpr uint32_t q_subblock_num_tiles = qkt_subblock_h * in0_block_w;
    constexpr uint32_t row_tiles = qkt_subblock_h * KT_stride;  // Use KT_stride for cb_qkt_im row width
    static_assert(!(use_padded_mask && ring_mode), "use_padded_mask and ring_mode are mutually exclusive");

    uint32_t pushed_rows = 0;
    // Q lives at [q_base_tiles, q_base_tiles + Sq_chunk_t*DHt) from the CB front. wait_front counts
    // from the front, so the wait target includes the chunks of earlier passes that stay resident.
    uint32_t q_wait_tiles = (has_q_base_tiles ? q_base_tiles : 0) + q_subblock_num_tiles;
    uint32_t q_index_offset = has_q_base_tiles ? q_base_tiles : 0;
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
        sdpa_mm_init(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);
        // Configure pack once before the kt loop for cb_qkt_im. Both sub_exp
        // and blocked_matmul_and_pack skip their internal configure (same cb+width).
        // sub_exp's configure_single_tile_pack(reduce_cb) clobbers the global to 1,
        // so blocked_matmul_and_pack must reconfigure when q_subblock > 0.
        // When q_subblock == 0, no sub_exp → global stays set → skip there too.
        configure_row_pack_width(cb_qkt_im, actual_sbw);

        // Mask plan for this q_subblock (single source of truth; reused by the mask stamp below).
        constexpr bool uses_lightweight_mask =
            sdpa_uses_lightweight_mask<ring_mode, is_causal_sdpa, use_padded_mask, sliding_window_size>();
        const bool packed_columns = packed_kv_enabled && packed_mask.packed();
        const bool packed_unmasked =
            packed_kv_enabled && packed_mask.mode == ring_joint::PackedKVMaskMode::PackedUnmasked;
        const bool should_apply_lightweight_mask = !packed_unmasked && sdpa_lightweight_mask_stamped(
                                                                           kv_pad_rotation_enabled,
                                                                           is_causal_sdpa && apply_causal,
                                                                           apply_sliding_window,
                                                                           apply_mask && lw_partial_tile_idx > 0);
        const bool no_mask_this_iter = !use_provided_mask && !(uses_lightweight_mask && should_apply_lightweight_mask);

        // run()#1 (the reduce's first half) may overlap the second-half pack only if no masked
        // column lands in [0, active_Sk/2) — see sdpa_first_half_unmasked.
        const bool overlap_first_half_contiguous = reduce_trigger && sdpa_first_half_unmasked<
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
        // Packed K columns are not one contiguous global range: prove the first half run by run.
        const bool overlap_first_half =
            packed_columns
                ? reduce_trigger &&
                      sdpa_packed_first_half_unmasked<kv_pad_rotation_enabled, sliding_window_size, use_provided_mask>(
                          no_mask_this_iter,
                          apply_mask,
                          lw_partial_tile_idx,
                          packed_mask,
                          mask_q_start_tile + q_subblock * qkt_subblock_h,
                          kv_pad_rotation.logical_tile_count,
                          active_Sk)
                : overlap_first_half_contiguous;
        // PACK posts the first-half token after the subblock covering the last first-half column
        // [active_Sk/2 - 1]; committing a superset of [0, active_Sk/2) is safe (run()#1's cols are a subset).
        const uint32_t first_half_last_sb = (active_Sk / 2 - 1) / actual_sbw;

        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_full_subblocks; ++kt_subblock) {
            if (q_subblock > 0) {
                uint32_t prev_q_subblock = q_subblock - 1;
                sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
                sub_exp_block_bcast_cols<profiling_enabled && kRingStepProfilingFine, scale_fp32>(
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
                sdpa_mm_reinit(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);
            }
            {
                MaybeDeviceZoneScopedN(profiling_enabled && kRingStepProfilingFine, "Q@KT MM+Pack");
                blocked_matmul_and_pack<true, KT_stride, KT_stride>(
                    cb_q_in,
                    cb_kt_in,
                    cb_qkt_im,
                    q_index_offset,
                    run_cols != nullptr ? static_cast<uint32_t>(run_cols[kt_subblock]) : kt_index_offset,
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
                    kv_pad_region_Nt,
                    kv_pad_chunk_size_t,
                    kv_pad_kv_local_padded_Nt,
                    sliding_window_size,
                    circular_kv_cache,
                    packed_kv_enabled>(
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
                    packed_columns ? active_Sk : (kv_pad_rotation_enabled ? KT_stride : active_Sk),
                    apply_sliding_window,
                    mask_straddle_col,
                    mask_straddle_jump,
                    kv_pad_rotation,
                    packed_mask);
                end_mask_l1_accumulate();
            }
        }
        if constexpr (use_col_bias) {
            if (apply_col_bias) {
                // Same tile row of bias for every Q tile row of this subblock (reader pushed Sk_chunk_t tiles).
                CircularBuffer(cb_col_bias).wait_front(Sk_chunk_t);
                begin_mask_l1_accumulate<false>(cb_qkt_im, cb_col_bias);
                for (uint32_t row = 0; row < qkt_subblock_h; ++row) {
                    l1_acc_tile_run(
                        cb_col_bias, 0, cb_qkt_im, (q_subblock * qkt_subblock_h + row) * KT_stride, active_Sk);
                }
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
            MaybeDeviceZoneScopedN(profiling_enabled && kRingStepProfilingFine, "Reduce max");
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
        if (!hold_kv) {
            CircularBuffer(cb_kt_in).pop_front(DHt * KT_stride);
        }
    }
    if constexpr (use_col_bias) {
        if (apply_col_bias) {
            CircularBuffer(cb_col_bias).pop_front(Sk_chunk_t);
        }
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
        // The in-place sub_exp below PACK-writes the last QK row group; the first V matmul then
        // UNPACK-reads rows [0, qktv_h). Overlapping ranges need a pack->unpack rendezvous or the
        // matmul reads stale L1. q_num_subblocks == 1 was too narrow: the granularities also diverge
        // when the QK solver falls back to subblock_h == 1 (Sk_chunk_t not divisible by 4) while
        // Phase 2 widens qktv_h to 2, i.e. exactly Sq_chunk_t == 2.
        constexpr bool qktv_first_group_reads_inplace_row = (q_num_subblocks - 1) * qkt_subblock_h < qktv_h;
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
            // the V-matmul unpack whenever its first row group overlaps the in-place writes.

            if constexpr (!kt_inplace_v) {
                // Split-drain (common, materialized-V path): interleave each column-subblock's
                // sub_exp with its partial V matmul; partial products accumulate across kt_sub via L1.
                for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {
                    sub_exp_block_bcast_cols<profiling_enabled && kRingStepProfilingFine, scale_fp32>(
                        cb_qkt_im,
                        cur.max,
                        cur.sum,
                        KT_stride,
                        q_num_subblocks - 1,
                        kt_sub * actual_sbw,
                        qkt_subblock_h,
                        actual_sbw);
                    if constexpr (qktv_first_group_reads_inplace_row) {
                        // PACK half only; SEMGET head-of-line-blocks the unpack thread, so the
                        // UNPACK half waits just before the matmul instead of stalling the setup
                        // below, none of which reads cb_qkt_im tile data.
                        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                    }
                    if (kt_sub == 0) {
                        CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
                        CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                    }
                    if (kt_sub > 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }

                    {
                        MaybeDeviceZoneScopedN(profiling_enabled && kRingStepProfilingFine, "QKT@V MM+Pack");
                        uint32_t v_index_offset = 0;
                        sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                            out_cb, out_cb);
                        // cb_qkt_im rows are laid out at KT_stride even when this kt_sub only consumes a
                        // narrower logical width. Keep unpack init on the physical stride; inner_dim below
                        // still limits how many V rows are multiplied.
                        sdpa_pv_mm_setup(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h, KT_stride);
                        configure_row_pack_width(out_cb, qktv_subblock_w);
                        if constexpr (qktv_first_group_reads_inplace_row) {
                            // UNPACK half of the rendezvous posted above.
                            UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                            UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                        }
                        // Block-run addressing (resident mode): this sub-block's V rows start at run_cols[kt_sub].
                        const uint32_t v_row0 =
                            run_cols != nullptr ? static_cast<uint32_t>(run_cols[kt_sub]) : kt_sub * matmul_inner;
                        for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                            const uint32_t qktv_in1_index = v_row0 * vDHt + v_index_offset;
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
                // In-place latent-V: software-pipelined split drain. The exp of column block kt+1
                // runs on the pack thread's SFPU while the FPU does the V matmul of block kt; partial
                // products accumulate in L1 across blocks. One PACK_DONE token per block orders the
                // in-place exp pack of block kt before the V matmul unpacks it: PACK posts it right
                // after block kt's pack, and UNPACK takes it immediately before block kt's V matmul.
                // Preparing block kt+1 before taking block kt's token allows up to two outstanding
                // tokens, matching PACK_DONE's capacity in init_sdpa_streaming_semaphores().
                static_assert(qktv_first_group_reads_inplace_row, "in-place V always reads the in-place row");
                static_assert(qktv_h == 1, "in-place V is single-tile Q");
                sub_exp_block_bcast_cols<profiling_enabled && kRingStepProfilingFine, scale_fp32>(
                    cb_qkt_im, cur.max, cur.sum, KT_stride, q_num_subblocks - 1, 0, qkt_subblock_h, actual_sbw);
                PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);
                for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {
                    const bool prepare_next_block = kt_sub + 1 < kt_num_full_subblocks;
                    const bool accumulate_in_l1 = kt_sub > 0;  // Block 0 overwrites the output row.
                    if (prepare_next_block) {
                        sub_exp_block_bcast_cols<profiling_enabled && kRingStepProfilingFine, scale_fp32>(
                            cb_qkt_im,
                            cur.max,
                            cur.sum,
                            KT_stride,
                            q_num_subblocks - 1,
                            (kt_sub + 1) * actual_sbw,
                            qkt_subblock_h,
                            actual_sbw);
                        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                    }
                    // Block kt's in-place exp must be committed before its V matmul unpacks it.
                    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                    if (accumulate_in_l1) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                    {
                        MaybeDeviceZoneScopedN(profiling_enabled && kRingStepProfilingFine, "QKT@V MM+Pack");
                        sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                            out_cb, out_cb);
                        inplace_v_matmul_pack_batched<vDHt, dst_size, qktv_h>(
                            cb_qkt_im,
                            cb_v_in,
                            out_cb,
                            qktv_in0_index_offset,
                            /*inner_dim=*/actual_sbw,
                            KT_stride,
                            /*inner_start=*/kt_sub * actual_sbw);
                        sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                    }
                    if (accumulate_in_l1) {
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

        // Per-row normalization lambda — fires on last K chunk (standard or deferred norm).
        // Takes sbh so it works for both full subblocks (qktv_h) and remainder (qktv_remainder_h).
        [[maybe_unused]] uint32_t sink_row_offset = 0;
        [[maybe_unused]] auto normalize_row = [&](uint32_t& pushed, uint32_t sbh) {
            MaybeDeviceZoneScopedN(profiling_enabled && kRingStepProfilingFine, "ROW_NORM");
            CircularBuffer(cur.sum).push_back(sbh);
            CircularBuffer(out_cb).push_back(sbh * vDHt);
            normalize_row_streaming<
                profiling_enabled && kRingStepProfilingFine,
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
                MaybeDeviceZoneScopedN(profiling_enabled && kRingStepProfilingFine, "S_CORR_FUSED");
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
                sub_exp_first_col_blocks<profiling_enabled && kRingStepProfilingFine, scale_fp32>(
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
                MaybeDeviceZoneScopedN(profiling_enabled && kRingStepProfilingFine, "QKT@V MM+Pack");
                uint32_t v_index_offset = 0;
                sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(
                    out_cb, out_cb);
                // See the q_subblock-0 V matmul above: active_Sk can be narrower than the physical
                // cb_qkt_im row stride, but the unpacker is configured for the physical layout.
                sdpa_pv_mm_setup(cb_qkt_im, cb_v_in, false, qktv_subblock_w, cur_h, KT_stride);
                // Configure once before v_subblock loop; skip inside.
                configure_row_pack_width(out_cb, qktv_subblock_w);
                if (run_cols == nullptr) {
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
                } else {
                    // Block-run addressing (resident mode): the inner dimension is split per kept sub-block, each
                    // read from its own V rows; partial products accumulate in L1 as the drain loop above does.
                    for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {
                        if (kt_sub > 0) {
                            PACK((llk_pack_reconfig_l1_acc(1)));
                        }
                        uint32_t v_col = 0;
                        for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                            blocked_matmul_and_pack<false, vDHt, vDHt>(
                                cb_qkt_im,
                                cb_v_in,
                                out_cb,
                                qktv_in0_index_offset + kt_sub * actual_sbw,
                                static_cast<uint32_t>(run_cols[kt_sub]) * vDHt + v_col,
                                w_q,
                                v_subblock * qktv_subblock_w,
                                qktv_subblock_w,
                                cur_h,
                                actual_sbw,
                                KT_stride,
                                /*skip_pack_configure=*/true);
                            v_col += qktv_subblock_w;
                        }
                        if (kt_sub > 0) {
                            PACK((llk_pack_reconfig_l1_acc(0)));
                        }
                    }
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
                    sub_exp_first_col_blocks<profiling_enabled && kRingStepProfilingFine, scale_fp32>(
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
                    sub_exp_first_col_blocks<profiling_enabled && kRingStepProfilingFine, scale_fp32>(
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

        if constexpr (use_attention_sink) {
            if (is_last_iter) {
                CircularBuffer(cb_attention_sink).pop_front(1);
            }
        }

        // For kt_inplace_v this is the deferred K^T pop: cb_v_in aliases cb_kt_in (v_shares_k_buffer),
        // and v_cb_physical_width_t == DHt, so this pops the same Sk_chunk_t*DHt entry that Phase 1
        // skipped. For the materialized path it pops the V entry as usual. Either way: one entry/chunk.
        if (!hold_kv) {
            CircularBuffer(cb_v_in).pop_front(KT_stride * v_cb_physical_width_t);
        }
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
    const bool use_zigzag_balancing = false,
    // K/V retention (one K chunk): Q heads per retention key, used to tell whether consecutive Q
    // chunks of this core share (batch, retention key) and so the same K and V. 0 = off.
    const uint32_t kv_reuse_group = 0) {
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
        // Keep K/V for the next Q chunk when it shares (batch, KV head) with this one (flat index b * NQH *
        // q_num_chunks + h * q_num_chunks + c, so its KV key is the flat index / (q_num_chunks * group)).
        const bool keep_kv =
            kv_reuse_group != 0 && q + 1 < q_chunks_per_core &&
            q_chunk_local / (q_num_chunks * kv_reuse_group) == (q_chunk_local + 1) / (q_num_chunks * kv_reuse_group);
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
                apply_sliding_window,
                0,      // mask_straddle_col
                0,      // mask_straddle_jump
                {},     // kv_pad_rotation
                0,      // q_base_tiles
                {},     // packed_mask
                false,  // apply_col_bias
                keep_kv && is_last);  // hold_kv: main's K/V retention (keep_kv) maps onto the Sol hold_kv (skip the pop)
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
 * @tparam q_local_padded_Nt Per-device Q padded sequence length in tiles.
 * @tparam kv_region_Nt      Per-source block-cyclic KV region; Q slab divided by the stripe split.
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
    // Head-serial passes: keep each pass's flash-attention state in an L1 FIFO
    // ({cb_sum_in, cb_max_in, cb_prev_out}, num_passes+1 entries deep) instead of the caller's
    // ping-pong halves. Compile-time so non-FIFO callers (normal ring joint) pay zero MATH-thread
    // instructions for the FIFO branches — the utilization-band perf gates resolve even
    // fraction-of-a-percent overhead on this path.
    bool use_l1_state_fifo = false,
    bool kv_pad_rotation_enabled = false,
    uint32_t v_cb_physical_width_t = vDHt,
    bool v_shares_k_buffer = false,
    bool kt_inplace_v = false,
    uint32_t sliding_window_size = 0,
    // Circular sliding KV cache slab count (0/1 = unbounded); wraps the sliding work
    // plan's local slab addressing. See sliding_window_work_plan.hpp.
    uint32_t circular_kv_slab_count = 0,
    uint32_t ring_size = 1,
    bool use_attention_sink = false,
    uint32_t cb_attention_sink = INVALID_CB,
    // Sharded joint: skip joint K chunks that lie entirely in the padded joint tail, mirroring the
    // reader's kv_chunk_is_beyond_logical_l skip so the K/V CB producer/consumer counts stay aligned.
    bool joint_n_skip_enabled = false,
    uint32_t joint_local_padded_Nt = 0,  // Lt_local: per-device joint tile count (sharded path)
    // Rotated Q split: map q-loop positions through the fixed base range and moving remainder.
    // Off by default (position is the flat id).
    bool rotated_q_split_enabled = false,
    // Dense chunked prefill: skip K chunks past this device's last Q row (the reader mirrors it).
    bool causal_k_skip = false,
    uint32_t kv_region_Nt = q_local_padded_Nt,
    // Split-KV: GROUPED_KV_SOURCE_COUNT > 1. Keeps the per-source path free of packed branches.
    bool packed_kv_enabled = false,
    typename MaskCtx = LightweightMaskContext,
    // Sol-Attn block gathering: the reader publishes per-(ring iter, Q chunk) plan words; chunks are
    // dense gathered chunks (last one narrowed) followed, on the first active iteration, by the pooled
    // chunks that carry a column bias.
    bool sol_enabled = false,
    uint32_t cb_sol_plan = INVALID_CB,
    uint32_t cb_sol_bias = INVALID_CB,
    uint32_t sol_pooled_chunks = 0,
    uint32_t sol_pooled_last_tiles = 0,
    // Chunk mode: the reader streams every chunk (chains on); plan words carry a kept-chunk bitmask and
    // the pooled flag; dropped chunks are popped unprocessed, pooled chunks follow the local ones.
    bool sol_chunk_mode = false,
    // Debug: drain every streamed chunk (own kept bits read as 0); measures the streaming floor. Wrong numerics.
    bool sol_drain_all = false,
    // Compact mode: streamed chunks hold this core's kept blocks left-justified; plan words 8..23 carry the per-chunk
    // kept-block counts (one byte each) and the step runs on the first 2 * count tile columns, no mask.
    bool sol_compact = false,
    // Cross-chunk packing. Compact mode compacts within a chunk, so a visited page carries only that chunk's
    // kept blocks (a fraction of the page at low density) while the fixed per-step cost is paid in full. Under
    // sol_pack the reader appends kept blocks across K-chunk boundaries and pushes a page once it holds
    // sol_blocks_per_page of them, flushing a partial page at the end of each ring iteration (a page cannot span
    // iterations -- shard r's K/V is resident only while iteration r streams).
    //
    // Packed pages arrive on cb_k_in / cb_v_in, the same CBs that carry chunks, so the inner-step call below is
    // the same and compute never sees -- or drains -- a raw chunk: the reader stages those privately. No plan
    // words are added either; reader and compute walk the same kept bitmask and derive identical page
    // boundaries. The loop's unit becomes the page, so k_chunk is a page ordinal under sol_pack.
    bool sol_pack = false,
    uint32_t sol_blocks_per_page = 1,
    // K/V CB depth in chunks; the end-of-iteration phase alignment pads to a whole number of cycles of it.
    uint32_t kv_cb_pages = 2>
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
    // True (unpadded) joint length in tiles; joint K chunks starting at/after it are pure padding.
    const uint32_t logical_lt = 0,
    // Tile offset of this call's Q chunk within cb_q_in (head-serial passes; 0 otherwise).
    const uint32_t q_base_tiles = 0,
    // Rotated Q split only: fixed base range plus this iteration's remainder unit.
    [[maybe_unused]] const RotatedQSlots& rotated_slots = {},
    // K split only: the local K chunks this core attends to (the reader skips the same).
    const uint32_t k_split_begin = 0,
    const uint32_t k_split_end = 0xFFFFFFFFu,
    // Sliding K split only: this core's band of the work plan (ring_joint_ksplit.hpp).
    const uint32_t sliding_split_idx = 0,
    const uint32_t sliding_split_count = 1,
    const uint32_t* packed_source_ids = nullptr,
    const uint32_t packed_source_count = 0,
    // Rows of newest slabs this pass appends (packed_kv_pass_rows).
    const uint32_t packed_newest_rows = 0,
    // Padded multicast slots the reader runs after this core's Q chunks; their K/V entries are drained.
    const uint32_t padded_q_slots = 0) {
    init_sdpa_streaming_semaphores();
    const bool packed_sources = packed_kv_enabled && packed_source_count > 1;
    const ring_joint::PackedKVGroupPlan packed_kv = ring_joint::packed_kv_pass_plan(
        {ring_joint::packed_kv_source_tiles(local_padded_Nt, logical_nt, kv_region_Nt, ring_size),
         packed_sources ? packed_source_count : 1,
         Sk_chunk_t,
         kv_region_Nt},
        packed_sources ? packed_newest_rows : 0);
    // Global end of the head slabs: every head tile lies below it.
    const uint32_t packed_head_global_end = packed_sources ? packed_kv.head_tiles() / kv_region_Nt * chunk_size_t : 0;

    constexpr bool has_sliding_window = sliding_window_size > 0;
    static_assert(!has_sliding_window || chunked_enabled, "Sliding windows require chunked prefill");
    // is_causal: diagonal stamp only on iter 0 (K is local-frame). Chunked: every iter (absolute coords).
    const bool is_causal_iter = (is_causal_sdpa && (ring_iter == 0)) || chunked_enabled;

    // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
    // All conditions are compile-time except the active_Sk == Sk_chunk_t guard (padded chunks).
    constexpr bool can_reduce_trigger = reduce_trigger_supported && (Sk_chunk_t % qkt_subblock_w == 0) &&
                                        (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

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

    auto normalize_only = [&]() {
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
            cb_normalized_out,
            scale_fp32,
            use_attention_sink,
            cb_attention_sink>(q_prev_norm.sum, q_prev_norm.out, Sq_chunk_t, q_prev_norm.max);
        if constexpr (use_attention_sink) {
            CircularBuffer(cb_attention_sink).pop_front(1);
        }
        sdpa_cb_pop_front_out_of_line(q_prev_norm.max, Sq_chunk_t);
        if (q_per_core > 1) {
            CircularBuffer(cb_signal).reserve_back(1);
            sdpa_cb_push_back_out_of_line(cb_signal, 1);
        }
    };

    // Last ring iter: normalize accumulated state and signal writer, then skip K-loop.
    auto try_normalize_only = [&](uint32_t q_chunk) -> bool {
        if constexpr (is_balanced_sdpa) {
            if (skip_first_half_q && q_chunk < num_q_chunks / 2 && is_last_ring_iter) {
                normalize_only();
                return true;
            }
        }
        return false;
    };

    // ---- K-loop helpers ---------------------------------------------------

    [[maybe_unused]] const uint32_t causal_end_nt =
        !causal_k_skip ? 0
                       : chunked_q_global_end_tile<kv_pad_rotation_enabled, q_local_padded_Nt>(
                             logical_nt,
                             chunked.ring_index,
                             ring_size,
                             chunked.kv_pad_rotation.q_pre_wrap_start_tile,
                             chunked.kv_pad_rotation.q_pre_wrap_tile_count,
                             chunked.kv_pad_rotation.q_post_wrap_start_tile,
                             chunked.kv_pad_rotation.q_valid_tile_count);

    // Skip KV chunks beyond the logical sequence length (padding tiles).
    auto try_skip_oob_kv = [&](uint32_t source_ring_id, uint32_t k_chunk, bool kv_chunk_is_joint) -> bool {
        if (kv_chunk_is_joint) {
            // Skip joint chunk at/after logical_lt - pure padding
            if constexpr (joint_n_skip_enabled) {
                const uint32_t joint_global_start_tile =
                    ring_id * joint_local_padded_Nt + (k_chunk - num_local_k_chunks) * Sk_chunk_t;
                return joint_global_start_tile >= logical_lt;
            }
            return false;
        }
        if (k_chunk < k_split_begin || k_chunk >= k_split_end) {
            return true;
        }
        if constexpr (chunked_enabled && !has_sliding_window && causal_k_skip) {
            return !chunked_kv_chunk_is_live<
                kv_pad_rotation_enabled,
                chunked_enabled,
                local_padded_Nt,
                chunk_size_t,
                q_local_padded_Nt>(source_ring_id, k_chunk, Sk_chunk_t, logical_nt, causal_end_nt);
        }
        return !kv_chunk_starts_before_logical_end<
            kv_pad_rotation_enabled,
            chunked_enabled,
            local_padded_Nt,
            chunk_size_t,
            kv_region_Nt>(source_ring_id, k_chunk * Sk_chunk_t, logical_nt);
    };

    // Causal skip: K chunks fully above the diagonal — drain K/V from CBs and skip.
    auto try_skip_causal_above_diag = [&](uint32_t k_chunk, uint32_t causal_k_limit) -> bool {
        if constexpr (is_causal_sdpa) {
            if constexpr (!has_sliding_window) {
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
        }
        return false;
    };

    // -----------------------------------------------------------------------

    for (uint32_t q = global_q_start; q < global_q_end; q++) {
        // Compute Q chunk index (with optional zigzag remapping for causal balancing).
        // num_q_chunks is total per-head chunks (local + joint), matching the divisor the
        // writer/reader use to flatten (batch, head, q_chunk) — see ring_joint_sdpa.cpp.
        uint32_t q_flat = q;
        if constexpr (rotated_q_split_enabled) {
            q_flat = rotated_slots.at(q);
        }
        uint32_t q_chunk = remap_q_index(q_flat, num_q_chunks, use_zigzag_balancing) % num_q_chunks;

        // Causal K-chunk limit and Q start tile for this Q chunk
        uint32_t causal_k_limit = num_kv_chunks;
        uint32_t q_start_tile = 0;
        if constexpr (has_sliding_window) {
            q_start_tile = logical_nt - ring_size * q_local_padded_Nt + chunked.ring_index * q_local_padded_Nt +
                           q_chunk * Sq_chunk_t;
        } else if constexpr (is_causal_sdpa) {
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

        ttnn::operations::transformer::sdpa::ring_joint::SlidingQWorkPlan<sliding_q_plan_max_source_ranges>
            sliding_q_plan;
        constexpr bool circular_kv_cache = circular_kv_slab_count > 1;
        if constexpr (has_sliding_window) {
            const ttnn::operations::transformer::sdpa::ring_joint::ChunkedQMapping q_mapping{
                chunked.kv_pad_rotation.q_pre_wrap_start_tile,
                chunked.kv_pad_rotation.q_pre_wrap_tile_count,
                chunked.kv_pad_rotation.q_post_wrap_start_tile,
                chunked.kv_pad_rotation.q_valid_tile_count};
            sliding_q_plan = ttnn::operations::transformer::sdpa::ring_joint::build_sliding_q_work_plan<
                sliding_q_plan_max_source_ranges>(
                q_chunk * Sq_chunk_t,
                Sq_chunk_t,
                chunked.ring_index,
                q_local_padded_Nt,
                ring_size,
                sliding_window_size,
                TILE_HEIGHT,
                local_padded_Nt,
                Sk_chunk_t,
                logical_nt,
                circular_kv_slab_count,
                kv_pad_rotation_enabled ? &q_mapping : nullptr,
                sliding_local_first && sliding_split_count == 1);
            ASSERT(sliding_q_plan.is_valid);
            acc_state.sliding_plan_k_chunks = sliding_q_plan.total_k_chunk_count;
            ASSERT(sliding_q_plan.total_k_chunk_count > 0);
        }
        // Per-Q pre-scan: count K chunks that will actually be processed.
        // Placed after balanced-skip guards so skipped Q chunks don't pay for the scan.
        const auto sliding_band = ttnn::operations::transformer::sdpa::ring_joint::sliding_ksplit_range(
            sliding_q_plan.total_k_chunk_count, sliding_split_idx, sliding_split_count);
        // Sol chunk mode: plan words [kept bitmask lo, hi, do_pooled] for this (ring iter, Q chunk).
        // Packing page accounting, both from plan word 5 (see the read below).
        [[maybe_unused]] uint32_t sol_total_kept_blocks = 0, sol_n_pages = 0;
        [[maybe_unused]] uint32_t sol_kept_lo = 0, sol_kept_hi = 0, sol_do_pooled = 0;
        [[maybe_unused]] uint32_t sol_union_lo = 0xffffffffu, sol_union_hi = 0xffffffffu;
        [[maybe_unused]] uint32_t sol_cnt_words[16] = {};
        if constexpr (sol_chunk_mode) {
            CircularBuffer(cb_sol_plan).wait_front(1);
            sol_kept_lo = ckernel::read_tile_value(cb_sol_plan, 0, 0);
            sol_kept_hi = ckernel::read_tile_value(cb_sol_plan, 0, 1);
            sol_do_pooled = ckernel::read_tile_value(cb_sol_plan, 0, 2);
            sol_union_lo = ckernel::read_tile_value(cb_sol_plan, 0, 3);
            sol_union_hi = ckernel::read_tile_value(cb_sol_plan, 0, 4);
            if constexpr (sol_pack) {
                // The reader publishes the kept-block TOTAL (word 5), computed from the same per-chunk counts it
                // wrote into words 8..23. Both the page count and the trailing page's fill come from it, so they
                // agree with each other and with the reader; re-deriving the total here can disagree with the
                // reader's page boundaries and deadlock (see the note in ring_joint_reader.cpp).
                sol_total_kept_blocks = ckernel::read_tile_value(cb_sol_plan, 0, 5);
                sol_n_pages = (sol_total_kept_blocks + sol_blocks_per_page - 1) / sol_blocks_per_page;
            }
            if constexpr (sol_compact) {
                for (uint32_t w = 0; w < (num_local_k_chunks + 3) / 4; ++w) {
                    sol_cnt_words[w] = ckernel::read_tile_value(cb_sol_plan, 0, 8 + w);
                }
            }
            CircularBuffer(cb_sol_plan).pop_front(1);
            if constexpr (sol_drain_all) {
                // Drain every streamed chunk except chunk 0 (the producer forces it kept, so every ring iteration still
                // computes one chunk; an iteration with zero computed chunks is not a supported state, see the
                // compact-mode fallback in the reader). The floor measurement therefore includes 1/nkc of the compute.
                sol_kept_lo &= 1u;
                sol_kept_hi = 0;
            }
        }
        const auto sol_chunk_kept = [&](uint32_t k) -> bool {
            return k < 32 ? ((sol_kept_lo >> k) & 1u) != 0 : ((sol_kept_hi >> (k - 32)) & 1u) != 0;
        };
        // Compact mode: kept blocks of local chunk k (the reader set the kept bit iff this is non-zero).
        [[maybe_unused]] const auto sol_chunk_blocks = [&](uint32_t k) -> uint32_t {
            return (sol_cnt_words[k / 4] >> (8 * (k % 4))) & 0xffu;
        };
        // Streamed by the reader at all (chain-wide union); own-dropped streamed chunks are drained.
        const auto sol_chunk_streamed = [&](uint32_t k) -> bool {
            return k < 32 ? ((sol_union_lo >> k) & 1u) != 0 : ((sol_union_hi >> (k - 32)) & 1u) != 0;
        };
        uint32_t per_q_valid_kv =
            packed_sources ? packed_kv.chunk_count() : (has_sliding_window ? sliding_band.end - sliding_band.begin : 0);
        for (uint32_t k = 0;
             !(sol_enabled && !sol_chunk_mode) && !packed_sources && !has_sliding_window && k < num_kv_chunks;
             ++k) {
            const uint32_t source_ring_id = ring_id;
            const uint32_t source_k_chunk = k;
            const bool is_joint = k >= num_local_k_chunks;
            if (try_skip_oob_kv(source_ring_id, source_k_chunk, is_joint)) {
                continue;
            }
            if constexpr (is_causal_sdpa) {
                if (is_causal_iter && k >= causal_k_limit) {
                    continue;
                }
            }
            if constexpr (sol_chunk_mode) {
                if (!sol_chunk_kept(k)) {
                    continue;
                }
            }
            if constexpr (sol_pack) {
                continue;  // a kept chunk is not a step under packing; the steps are the pages, counted above
            }
            per_q_valid_kv++;
        }
        if constexpr (sol_pack) {
            per_q_valid_kv += sol_n_pages;  // from the plan, not re-derived
        }
        if constexpr (sol_chunk_mode) {
            if (sol_do_pooled) {
                per_q_valid_kv += sol_pooled_chunks;
            }
        }
        // Use persistent accumulator state from caller (single Q-chunk),
        // restore from DRAM staging (multi Q-chunk), or pop from the L1 FIFO (head-serial passes).
        AccumulatorHalf q_prev = acc_state.prev, q_cur = acc_state.cur;

        const bool is_first_kv_for_this_q = is_first_active_iter;

        // Multi Q-chunk restore: K0 reads prev accumulators directly from staging buffers
        // (cb_prev_out, cb_max_in, cb_sum_in) — no copy_block needed.
        // After K0's swap, reset q_cur to original accumulator CBs for normal ping-pong.
        const AccumulatorHalf original_prev = q_prev;
        const bool restore_from_staging = (q_per_core > 1 && !is_first_active_iter);
        ASSERT(!has_sliding_window || !restore_from_staging);
        // FIFO entry: this pass's own state sits at the front (the fixed cyclic pass order
        // guarantees it), so the first K chunk merges straight out of the FIFO and pops it.
        // The staging path builds the identical triple, so both share one redirect.
        const bool fifo_entry = use_l1_state_fifo && !is_first_active_iter;
        if (restore_from_staging || fifo_entry) {
            q_prev = {cb_sum_in, cb_max_in, cb_prev_out};
        }

        uint32_t KV_chunks_processed = 0;

        // Global K bounds of this Q chunk's rows, shared by all its packed K chunks.
        static_assert(
            !packed_kv_enabled || Sk_chunk_t <= ring_joint::kMaxPackedKVChunkTiles,
            "packed mask runs live on the TRISC stack; the host groups only chunks up to the cap");
        static_assert(ring_joint::kPackedKVInvalidRowTile == KV_PAD_ROTATION_INVALID_TILE);
        ring_joint::PackedKVRowBounds packed_row_bounds{};
        if (packed_sources) {
            const uint32_t mask_q_start = kv_pad_rotation_enabled ? q_chunk * Sq_chunk_t : q_start_tile;
            packed_row_bounds =
                ring_joint::packed_kv_row_bounds(Sq_chunk_t, logical_nt, packed_head_global_end, [&](uint32_t row) {
                    return q_global_tile_for_mask_row<kv_pad_rotation_enabled>(
                        row, mask_q_start, chunked.kv_pad_rotation);
                });
        }

        uint32_t q_k_loop_count = has_sliding_window ? sliding_band.end : num_kv_chunks;
        if constexpr (sol_chunk_mode) {
            q_k_loop_count = num_kv_chunks + (sol_do_pooled ? sol_pooled_chunks : 0);
        }
        // Under packing the loop's unit is a PAGE, not a K chunk. The reader pushed one page per
        // sol_blocks_per_page kept blocks plus a partial at the end of the iteration, and nothing else -- raw
        // chunks never reach compute, so there is nothing to skip and nothing to drain. Pooled chunks follow the
        // pages in the same way they follow the chunks in chunk mode.
        if constexpr (sol_pack) {
            q_k_loop_count = sol_n_pages + (sol_do_pooled ? sol_pooled_chunks : 0);
        }
        if constexpr (sol_enabled && !sol_chunk_mode) {
            // ---- Sol-Attn gathered-chunk loop. Plan words from the reader: [n_gather, last_tiles, do_pooled].
            CircularBuffer(cb_sol_plan).wait_front(1);
            const uint32_t sol_n_gather = ckernel::read_tile_value(cb_sol_plan, 0, 0);
            const uint32_t sol_last_tiles = ckernel::read_tile_value(cb_sol_plan, 0, 1);
            const uint32_t sol_do_pooled = ckernel::read_tile_value(cb_sol_plan, 0, 2);
            CircularBuffer(cb_sol_plan).pop_front(1);
            const uint32_t sol_total = sol_n_gather + (sol_do_pooled ? sol_pooled_chunks : 0);
            for (uint32_t c = 0; c < sol_total; ++c) {
                const bool is_pooled = c >= sol_n_gather;
                uint32_t active_Sk_param = Sk_chunk_t;
                if (is_pooled) {
                    if (c - sol_n_gather + 1 == sol_pooled_chunks) {
                        active_Sk_param = sol_pooled_last_tiles;
                    }
                } else if (c + 1 == sol_n_gather) {
                    active_Sk_param = sol_last_tiles;
                }
                const uint32_t chunk_sbw =
                    active_Sk_param == Sk_chunk_t ? full_sbw : largest_factor_le(active_Sk_param, qkt_subblock_w);
                KV_chunks_processed++;
                KV_chunks_processed_in_iter++;
                const bool is_first = is_first_kv_for_this_q && (KV_chunks_processed == 1);
                const bool is_last_k = (KV_chunks_processed == sol_total);
                const bool is_last_k_of_last_ring_iter = is_last_ring_iter && is_last_k;
                if (is_last_k && q_per_core > 1) {
                    CircularBuffer(cb_signal).reserve_back(1);
                    sdpa_cb_push_back_out_of_line(cb_signal, 1);
                }
                const bool save_to_staging = is_last_k && !is_last_ring_iter && q_per_core > 1;
                const uint32_t step_save_out_cb = save_to_staging ? cb_out : INVALID_CB;
                const uint32_t step_save_max_cb = save_to_staging ? cb_max_out : INVALID_CB;
                if (save_to_staging) {
                    q_cur.sum = cb_sum_out;
                }
                {
                    MaybeDeviceZoneScopedN(kRingStepProfiling, "RING_STEP");
                    sdpa_inner_loop_step<
                        kRingStepProfilingPhases,  // profiling_enabled: phase zones (level 2), fine zones (level 3)
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
                        false,  // use_padded_mask
                        true,   // ring_mode
                        false,  // causal
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
                        false,  // kv_pad_rotation
                        kv_region_Nt,
                        chunk_size_t,
                        local_padded_Nt,
                        v_cb_physical_width_t,
                        kt_inplace_v,
                        0,  // sliding window
                        use_attention_sink,
                        cb_attention_sink,
                        false,  // use_provided_mask
                        use_l1_state_fifo,
                        false,  // circular_kv_cache
                        false,  // packed_kv_enabled
                        true,   // use_col_bias
                        cb_sol_bias>(
                        q_prev,
                        q_cur,
                        is_last_k_of_last_ring_iter,
                        is_first,
                        /*apply_mask=*/false,
                        /*lw_partial_tile_idx=*/0,
                        active_Sk_param,
                        can_reduce_trigger && (active_Sk_param == Sk_chunk_t) && !is_pooled,
                        chunk_sbw,
                        step_save_out_cb,
                        step_save_max_cb,
                        /*apply_causal=*/false,
                        0,
                        0,
                        lw_mask.neginf_tile_idx,
                        0,
                        0,
                        0,
                        0,
                        false,
                        0,
                        0,
                        chunked.kv_pad_rotation,
                        q_base_tiles,
                        {},
                        /*apply_col_bias=*/is_pooled);
                }
                if (!is_first) {
                    sdpa_cb_pop_front_out_of_line(q_prev.max, Sq_chunk_t);
                    sdpa_cb_pop_front_out_of_line(q_prev.sum, Sq_chunk_t);
                }
                if (is_last_k_of_last_ring_iter) {
                    sdpa_cb_pop_front_out_of_line(q_cur.max, Sq_chunk_t);
                } else {
                    std::swap(q_prev, q_cur);
                    if ((restore_from_staging || fifo_entry) && KV_chunks_processed == 1) {
                        q_cur = {original_prev.sum, original_prev.max, original_prev.out};
                    }
                }
            }
        } else {
            for (uint32_t k_chunk = has_sliding_window ? sliding_band.begin : 0; k_chunk < q_k_loop_count; ++k_chunk) {
                const auto sliding_k_chunk = sliding_q_plan.k_chunk_at(k_chunk);
                // Circular cache: the mask's K origin is the plan's absolute K-chunk index (inverting local
                // cache rows is ambiguous once several chunk groups alias one local slab); unbounded caches
                // keep the inversion. Resolved once here, outside the [&] lambdas below: anything new they
                // capture changes the unbounded kernels' register allocation.
                uint32_t circular_k_origin_tile = 0;
                if constexpr (circular_kv_cache) {
                    circular_k_origin_tile = sliding_q_plan.global_k_chunk_at(k_chunk) * Sk_chunk_t;
                }
                const uint32_t source_ring_id = has_sliding_window ? sliding_k_chunk.source_ring_id : ring_id;
                const uint32_t source_k_chunk = has_sliding_window ? sliding_k_chunk.source_k_chunk : k_chunk;
                // Sol chunk mode: indices past the local chunks are the pooled chunks (always processed, with bias).
                [[maybe_unused]] bool sol_is_pooled = false;
                if constexpr (sol_chunk_mode) {
                    sol_is_pooled = k_chunk >= num_kv_chunks;
                }
                // Packing: k_chunk indexes pages, so the pooled chunks start after the last page, not after the last
                // K chunk. Its blocks are all kept by construction -- the reader only packed kept ones -- so the
                // streamed/kept/causal/OOB gating below is skipped entirely for a page.
                [[maybe_unused]] uint32_t sol_page_blocks = 0;
                if constexpr (sol_pack) {
                    sol_is_pooled = k_chunk >= sol_n_pages;
                    if (!sol_is_pooled) {
                        const uint32_t remaining = sol_total_kept_blocks - k_chunk * sol_blocks_per_page;
                        sol_page_blocks =
                            remaining < sol_blocks_per_page ? remaining : sol_blocks_per_page;  // tail page
                    }
                }
                // Packing: a page is not a K chunk, so none of the chunk-indexed gating applies to it. Every block in
                // a page was selected by this core, the reader already resolved out-of-range and joint chunks
                // before packing, and k_chunk is a page ordinal that these predicates would misread.
                const bool kv_chunk_is_joint = !sol_pack && !sol_is_pooled && !packed_sources && !has_sliding_window &&
                                               k_chunk >= num_local_k_chunks;
                if (!sol_pack && !sol_is_pooled && !packed_sources && !has_sliding_window &&
                    try_skip_oob_kv(source_ring_id, source_k_chunk, kv_chunk_is_joint)) {
                    continue;
                }
                if (!sol_pack && !sol_is_pooled && try_skip_causal_above_diag(source_k_chunk, causal_k_limit)) {
                    continue;
                }
                if constexpr (sol_chunk_mode && !sol_pack) {
                    if (!sol_is_pooled && !sol_chunk_streamed(k_chunk)) {
                        continue;  // skipped by the whole chain: nothing was pushed
                    }
                    // Dropped chunk: the reader streamed it (chain lockstep); drain K^T and V without computing.
                    if (!sol_is_pooled && !sol_chunk_kept(k_chunk)) {
                        MaybeDeviceZoneScopedN(kRingStepProfiling, "RING_DRAIN");
                        CircularBuffer(cb_kt_in).wait_front(DHt * Sk_chunk_t);
                        sdpa_cb_pop_front_out_of_line(cb_kt_in, DHt * Sk_chunk_t);
                        if constexpr (!kt_inplace_v) {
                            CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                            sdpa_cb_pop_front_out_of_line(cb_v_in, Sk_chunk_t * v_cb_physical_width_t);
                        }
                        KV_chunks_processed_in_iter++;
                        continue;
                    }
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
                    is_global_n_mask_chunk = ring_iter_needs_global_n_mask && source_k_chunk == global_n_mask_chunk_id;
                }
                if constexpr (local_n_mask_enabled) {
                    is_local_n_mask_chunk = local_n_needs_masking && source_k_chunk == local_n_mask_chunk_id;
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
                        lw_mask.straddle_num_padded_tiles > 0 && source_k_chunk == lw_mask.straddle_mask_chunk_id;
                }

                bool apply_mask = kv_pad_rotation_enabled;
                if constexpr (global_n_mask_enabled) {
                    apply_mask = apply_mask || is_global_n_mask_chunk;
                }
                if constexpr (joint_n_mask_enabled) {
                    // Mirror the spatial local_n strategy for the joint chunk. The exp variant's LightweightMaskContext
                    // keeps it runtime.
                    if (lw_mask.joint_l_partial_col > 0) {
                        apply_mask = apply_mask || is_joint_n_mask_chunk;
                    }
                }

                // Resolve lightweight mask params for partial tile masking
                uint32_t lw_partial_tile_idx = 0;
                if constexpr (lightweight_mask_enabled) {
                    if (apply_mask) {
                        bool partial_tile_selected = false;
                        // Live column 0 = tile-aligned boundary this dispatch: no partial stamp.
                        if constexpr (global_n_mask_enabled) {
                            if (!packed_sources && is_global_n_mask_chunk) {
                                lw_partial_tile_idx =
                                    lw_mask.global_n_partial_col > 0 ? lw_mask.global_n_partial_tile_idx : 0u;
                                partial_tile_selected = true;
                            }
                        }
                        if constexpr (joint_n_mask_enabled) {
                            if (!partial_tile_selected && is_joint_n_mask_chunk) {
                                lw_partial_tile_idx =
                                    lw_mask.joint_l_partial_col > 0 ? lw_mask.joint_l_partial_tile_idx : 0u;
                            }
                        }
                    }
                }

                // Tile-level matmul skip for global_n, local_n, or joint_n padding.
                // Runtime reduce narrows to active_Sk; matmul/sub_exp/V also need narrowing.
                // Also select pre-computed subblock width for this chunk's mask type.
                uint32_t active_Sk_param = packed_sources ? packed_kv.valid_tiles(k_chunk) : Sk_chunk_t;
                uint32_t chunk_sbw = packed_sources && active_Sk_param != Sk_chunk_t
                                         ? largest_factor_le(active_Sk_param, qkt_subblock_w)
                                         : full_sbw;
                bool narrowed_by_mask = packed_sources;
                if constexpr (global_n_mask_enabled) {
                    if (!packed_sources && is_global_n_mask_chunk) {
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
                    // A short balanced shard can end inside the same K chunk as its
                    // causal half boundary. Padding does not supersede that tighter
                    // boundary: intersect both masks rather than keeping the first one.
                    const uint32_t straddle_active = Sk_chunk_t - lw_mask.straddle_num_padded_tiles;
                    if (is_straddle_mask_chunk && straddle_active < active_Sk_param) {
                        active_Sk_param = straddle_active;
                        chunk_sbw = straddle_sbw;
                    }
                }

                // Causal narrowing on the diagonal-crossing K-chunk: cols past the last Q-row's
                // diag tile are -inf for every row in the Q-chunk, so skip matmul/sub_exp/V there.
                // Composes with any prior padding narrowing via min().
                if constexpr (is_causal_sdpa && !kv_pad_rotation_enabled) {
                    if constexpr (has_sliding_window) {
                        const uint32_t k_global_start =
                            circular_kv_cache
                                ? circular_k_origin_tile
                                : kv_global_tile_for_local<true, local_padded_Nt, chunk_size_t, kv_region_Nt>(
                                      source_ring_id, source_k_chunk * Sk_chunk_t);
                        const uint32_t q_global_end = q_start_tile + Sq_chunk_t;
                        if (k_global_start < q_global_end && q_global_end - k_global_start < active_Sk_param) {
                            active_Sk_param = q_global_end - k_global_start;
                            chunk_sbw = largest_factor_le(active_Sk_param, qkt_subblock_w);
                        }
                    } else if (is_causal_iter && k_chunk == causal_k_limit - 1) {
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
                ASSERT(!has_sliding_window || !save_to_staging);
                // FIFO exit: sum/out redirect into the FIFO (pack-only, write-pointer relative). max
                // cannot — the step reads cur.max front-relative — so mirror it via step_save_max_cb.
                const bool fifo_exit = use_l1_state_fifo && is_last_k && !is_last_ring_iter;
                const uint32_t step_save_out_cb = save_to_staging ? cb_out : (fifo_exit ? cb_prev_out : INVALID_CB);
                const uint32_t step_save_max_cb = save_to_staging ? cb_max_out : (fifo_exit ? cb_max_in : INVALID_CB);
                if (save_to_staging) {
                    q_cur.sum = cb_sum_out;
                } else if (fifo_exit) {
                    q_cur.sum = cb_sum_in;
                }

                // K start tile fed to diag stamp must share Q's coord frame (local for is_causal, global for chunked).
                const uint32_t step_k_start_tile_local = [&]() {
                    if constexpr (has_sliding_window) {
                        return kv_global_tile_for_local<true, local_padded_Nt, chunk_size_t, kv_region_Nt>(
                            source_ring_id, source_k_chunk * Sk_chunk_t);
                    } else if constexpr (chunked_enabled) {
                        return kv_global_tile_for_local<true, local_padded_Nt, chunk_size_t, kv_region_Nt>(
                            ring_id, k_chunk * Sk_chunk_t);
                    } else {
                        return k_chunk * Sk_chunk_t;
                    }
                }();
                // Circular cache: the plan's absolute origin replaces the inversion above.
                const uint32_t step_k_start_tile = circular_kv_cache ? circular_k_origin_tile : step_k_start_tile_local;
                const bool step_apply_causal = [&]() {
                    if constexpr (has_sliding_window) {
                        return q_start_tile < step_k_start_tile + Sk_chunk_t &&
                               q_start_tile + Sq_chunk_t > step_k_start_tile;
                    }
                    return is_causal_iter;
                }();

                // Chunked-prefill straddle. Background: each device's K cache holds the per-chunk
                // K region for every chunk back-to-back, kv_region_Nt tiles per region. When
                // k_chunk_size does not divide kv_region_Nt, a single K-chunk can begin in
                // one region (chunk j) and end one or more regions later. Because adjacent regions
                // map to *non-adjacent* global K positions (jumping by chunk_size_t between them),
                // the global K coord is no longer contiguous across the K-chunk's columns. We
                // signal this to the diag stamp via:
                //   - straddle_col: column index at which the jump happens (= tiles remaining in
                //     region j from this K-chunk's start)
                //   - straddle_jump: the global-K increment at that boundary
                //     (= chunk_size_t - kv_region_Nt, i.e. the gap between region j's end and
                //     region j+1's start in global K).
                // When straddle_col > 0 the stamp evaluates the diagonal per column instead of
                // per row, applying the jump at straddle_col and every kv_region_Nt tiles thereafter.
                uint32_t step_straddle_col = 0;
                uint32_t step_straddle_jump = 0;
                if constexpr (chunked_enabled) {
                    if (kv_region_Nt > 0) {
                        const uint32_t local_start = source_k_chunk * Sk_chunk_t;
                        const uint32_t slab_end_local = (local_start / kv_region_Nt + 1) * kv_region_Nt;
                        if (local_start + Sk_chunk_t > slab_end_local) {
                            step_straddle_col = slab_end_local - local_start;
                            step_straddle_jump = chunk_size_t - kv_region_Nt;
                        }
                    }
                }
                KVPadRotationContext step_kv_pad_rotation = chunked.kv_pad_rotation;
                step_kv_pad_rotation.k_local_start_tile = source_k_chunk * Sk_chunk_t;
                step_kv_pad_rotation.ring_id = source_ring_id;
                step_kv_pad_rotation.logical_tile_count = logical_nt;

                // Packed chunks mostly hold prefix slabs that every row of this Q chunk sees in full; the
                // plan narrows the live width and says whether the chunk needs stamping.
                if constexpr (sol_chunk_mode) {
                    if (sol_is_pooled) {
                        apply_mask = false;
                        active_Sk_param =
                            (k_chunk - num_kv_chunks + 1 == sol_pooled_chunks) ? sol_pooled_last_tiles : Sk_chunk_t;
                        chunk_sbw = active_Sk_param == Sk_chunk_t ? full_sbw
                                                                  : largest_factor_le(active_Sk_param, qkt_subblock_w);
                    }
                }
                if constexpr (sol_compact) {
                    if (!sol_is_pooled) {
                        // Compacted chunk: the first 2 * kept tile columns (K^T) / rows (V) are full valid blocks; the
                        // ragged tail never reaches here (partial blocks are pooled), so no mask is applied.
                        //
                        // Packing: same shape, but the count is the PAGE's fill rather than one chunk's. Every page
                        // but the last is full, so active_Sk_param == Sk_chunk_t and the subblock width is the optimal
                        // one -- the awkward factorisations of a partial width occur at most once per ring iteration
                        // instead of on every visit. The tail page's slack is never computed and is NOT
                        // zero-filled: a zero K tile would give QK = 0, hence exp(0) = 1, a full-weight softmax
                        // term against a zero V, inflating the denominator.
                        apply_mask = false;
                        // 2 tiles per 64-token block (kSolBlockTiles in the reader and the factory).
                        active_Sk_param = (sol_pack ? sol_page_blocks : sol_chunk_blocks(k_chunk)) * 2;
                        chunk_sbw = active_Sk_param == Sk_chunk_t ? full_sbw
                                                                  : largest_factor_le(active_Sk_param, qkt_subblock_w);
                    }
                }
                ring_joint::PackedKVMaskRun packed_runs[packed_kv_enabled ? Sk_chunk_t : 1];
                ring_joint::PackedKVChunkMask packed_mask;
                if (packed_sources) {
                    const ring_joint::PackedKVChunkPlan plan = ring_joint::packed_kv_chunk_plan(
                        packed_kv,
                        k_chunk,
                        packed_source_ids,
                        chunk_size_t,
                        packed_row_bounds,
                        qkt_subblock_w,
                        active_Sk_param,
                        packed_runs);
                    if (plan.active_tiles < active_Sk_param) {
                        active_Sk_param = plan.active_tiles;
                        chunk_sbw = largest_factor_le(active_Sk_param, qkt_subblock_w);
                    }
                    packed_mask = plan.mask;
                }

                {
                    MaybeDeviceZoneScopedN(kRingStepProfiling, "RING_STEP");
                    sdpa_inner_loop_step<
                        kRingStepProfilingPhases,  // profiling_enabled: phase zones (level 2), fine zones (level 3)
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
                        kv_region_Nt,
                        chunk_size_t,
                        local_padded_Nt,
                        v_cb_physical_width_t,
                        kt_inplace_v,
                        sliding_window_size,
                        use_attention_sink,
                        cb_attention_sink,
                        false,              // use_provided_mask
                        use_l1_state_fifo,  // has_q_base_tiles: head-serial passes read Q at q_base_tiles
                        circular_kv_cache,
                        packed_kv_enabled,
                        sol_chunk_mode,  // use_col_bias (pooled chunks only, runtime-gated)
                        cb_sol_bias>(
                        q_prev,
                        q_cur,
                        is_last_k_of_last_ring_iter,
                        is_first,
                        apply_mask,
                        lw_partial_tile_idx,
                        active_Sk_param,
                        can_reduce_trigger && (active_Sk_param == Sk_chunk_t) && !sol_is_pooled,
                        chunk_sbw,
                        step_save_out_cb,
                        step_save_max_cb,
                        step_apply_causal,
                        kv_pad_rotation_enabled ? q_chunk * Sq_chunk_t : q_start_tile,
                        step_k_start_tile,
                        lw_mask.neginf_tile_idx,
                        has_sliding_window ? 1 : lw_mask.causal_diag_tile_idx,
                        has_sliding_window ? 2 : 0,
                        has_sliding_window ? 3 : 0,
                        has_sliding_window ? 4 : 0,
                        has_sliding_window,
                        step_straddle_col,
                        step_straddle_jump,
                        step_kv_pad_rotation,
                        q_base_tiles,
                        packed_mask,
                        /*apply_col_bias=*/sol_is_pooled);
                }

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
                    // After K0's swap, q_cur holds the staging/FIFO buffers. Reset to the scratch
                    // accumulator CBs so the rest of the pass ping-pongs between scratch halves.
                    if ((restore_from_staging || fifo_entry) && KV_chunks_processed == 1) {
                        q_cur = {original_prev.sum, original_prev.max, original_prev.out};
                    }
                }
            }
        }  // sol_enabled / dense K loop
        ASSERT(sol_enabled || !has_sliding_window || KV_chunks_processed == per_q_valid_kv);

        // Pop Q — not popped inside step since ring_mode gates the early Q pop.
        // When q_per_core == 1, Q is identical across ring iterations so we keep it
        // fronted in the CB and only pop on the last iteration to avoid redundant DRAM re-reads.
        // Head-serial passes own the pop themselves: all resident chunks are popped together after
        // the last pass of the last ring iteration, so nothing is popped here.
        if constexpr (!use_l1_state_fifo) {
            if (q_per_core > 1 || is_last_ring_iter) {
                sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
            }
        }

        // Persist or save accumulators for next ring iteration.
        //
        // FIFO passes persist nothing. Only the scratch max (mirrored, not redirected) is left;
        // after the last swap it is q_prev.max — pop it.
        if constexpr (use_l1_state_fifo) {
            if (!is_last_ring_iter) {
                sdpa_cb_pop_front_out_of_line(q_prev.max, Sq_chunk_t);
            }
        } else if (q_per_core == 1) {
            // Single Q-chunk: persist in L1 (no DRAM round-trip)
            acc_state.prev = q_prev;
            acc_state.cur = q_cur;
            acc_state.last_call_k_chunks = KV_chunks_processed;
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

    // Padded multicast slots: the reader pushes every K (and V) entry it receives so this core's CB
    // advances with the injector's. Drain them in reader order with the same skips as a real Q chunk;
    // padded_slot_flat_q decodes the chunk each slot stands for, exactly as the reader does.
    if constexpr (!has_sliding_window) {
        for (uint32_t slot = 0; slot < padded_q_slots; ++slot) {
            const uint32_t owned = global_q_end - global_q_start;
            const uint32_t q_flat = padded_slot_flat_q<rotated_q_split_enabled>(
                rotated_slots, global_q_start, owned, owned + slot, use_zigzag_balancing ? 2 : 1);
            const uint32_t q_chunk = remap_q_index(q_flat, num_q_chunks, use_zigzag_balancing) % num_q_chunks;
            if constexpr (is_balanced_sdpa) {
                if (skip_first_half_q && q_chunk < num_q_chunks / 2) {
                    continue;
                }
            }
            const uint32_t padded_k_chunks = packed_sources ? packed_kv.chunk_count() : num_kv_chunks;
            for (uint32_t k_chunk = 0; k_chunk < padded_k_chunks; ++k_chunk) {
                if (!packed_sources && try_skip_oob_kv(ring_id, k_chunk, k_chunk >= num_local_k_chunks)) {
                    continue;
                }
                CircularBuffer(cb_kt_in).wait_front(DHt * Sk_chunk_t);
                sdpa_cb_pop_front_out_of_line(cb_kt_in, DHt * Sk_chunk_t);
                if constexpr (!kt_inplace_v) {
                    CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                    sdpa_cb_pop_front_out_of_line(cb_v_in, Sk_chunk_t * v_cb_physical_width_t);
                }
                KV_chunks_processed_in_iter++;
            }
        }
    }

    // Dummy KV traffic exists only to align chained readers. Sliding mode uses independent
    // per-core reads so each core can prune a different K range and needs no phase padding.
    if constexpr (!has_sliding_window) {
        for (uint32_t dummy_chunk = 0;
             dummy_chunk < dummy_kv_chunks_for_phase_alignment<v_shares_k_buffer, kt_inplace_v, kv_cb_pages>(
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
}

// ============ Sol-Attn resident mode: in-place slot state, lazy running max, fused per-piece steps ============

// ---- Outlined helpers: each is reached from several sites on every TRISC, and the kernel config buffer has
// little margin left for the resident build. ----

// Packs a DEST batch of units into the compact score scratch: unit u = member unit_g[u], score column unit_c0[u],
// width unit_cw[u]; row r of member g is at off[g] + r * W[g]. Blocked (one call per row) when the unit is at least
// MIN_BLOCKED_PACK_TILES wide, per tile otherwise; the pack width is reconfigured only when it changes (pk_w).
template <uint32_t Sq_chunk_t, uint32_t cb>
static __attribute__((noinline, noclone)) void sol_res_pack_units(
    const uint8_t* unit_g,
    const uint8_t* unit_c0,
    const uint8_t* unit_cw,
    uint32_t uu,
    const uint16_t* off,
    const uint8_t* W,
    uint32_t& pk_w) {
    uint32_t dst = 0;
    for (uint32_t u = 0; u < uu; ++u) {
        const uint32_t cw = unit_cw[u];
        const bool blocked = should_use_blocked_pack_width(cw);
        const uint32_t want = blocked ? cw : 1u;
        if (want != pk_w) {
            configure_pack_width(cb, want);
            pk_w = want;
        }
        const uint32_t g = unit_g[u];
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            const uint32_t idx = off[g] + r * W[g] + unit_c0[u];
            if (blocked) {
                sdpa_pack_tile_ooo(dst, cb, idx);
                dst += cw;
            } else {
                for (uint32_t c = 0; c < cw; ++c) {
                    sdpa_pack_tile_ooo(dst++, cb, idx + c);
                }
            }
        }
    }
}

// Members sorted by width (stable insertion sort of an index permutation), then stepped in groups of at most
// kQkCap scratch tiles (Sq_chunk_t x width per member).
template <uint32_t kSlots, uint32_t kBlocksPerPiece, uint32_t Sq_chunk_t, uint32_t kQkCap, typename StepFn>
static __attribute__((noinline, noclone)) void sol_res_run_groups(
    StepFn& run_step,
    const uint8_t* members,
    const uint8_t* widths,
    const uint8_t* mruns,
    const uint8_t* mrunws,
    const uint8_t* mnruns,
    const uint32_t n,
    const bool apply_col_bias) {
    uint8_t order[kSlots];
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t j = i;
        while (j > 0 && widths[order[j - 1]] > widths[i]) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = static_cast<uint8_t>(i);
    }
    uint8_t sg[kSlots];
    uint8_t sruns[kSlots * kBlocksPerPiece];
    uint8_t srunws[kSlots * kBlocksPerPiece];
    uint8_t snr[kSlots];
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t m = order[i];
        sg[i] = members[m];
        snr[i] = mnruns[m];
        for (uint32_t s = 0; s < mnruns[m]; ++s) {
            sruns[i * kBlocksPerPiece + s] = mruns[m * kBlocksPerPiece + s];
            srunws[i * kBlocksPerPiece + s] = mrunws[m * kBlocksPerPiece + s];
        }
    }
    uint32_t g0 = 0;
    while (g0 < n) {
        uint32_t g1 = g0;
        uint32_t tiles = 0;
        while (g1 < n && tiles + Sq_chunk_t * widths[order[g1]] <= kQkCap) {
            tiles += Sq_chunk_t * widths[order[g1]];
            ++g1;
        }
        ASSERT(g1 > g0);  // one member always fits: Sq_chunk_t x Sk_chunk_t <= kQkCap
        run_step(
            sg + g0, g1 - g0, sruns + g0 * kBlocksPerPiece, srunws + g0 * kBlocksPerPiece, snr + g0, apply_col_bias);
        g0 = g1;
    }
}

// Per-step state of the compact step's DEST-batch issue, shared with its outlined flushes (sol_res_s1_flush,
// sol_res_s5_flush): one copy of each flush body per TRISC instead of one per call site. The 5 s build sits within
// a few KB of the kernel config buffer (M9's and M10's first builds both overflowed it on inlined flushes).
template <uint32_t kDst, uint32_t kMaxMembers>
struct SolResStepState {
    // The step's member tables (set once per step).
    const uint8_t* slots = nullptr;
    const uint8_t* run_cols = nullptr;
    const uint8_t* run_ws = nullptr;
    const uint8_t* n_runs = nullptr;
    const uint8_t* W = nullptr;
    const uint16_t* off = nullptr;
    uint32_t n_slots = 0;
    uint32_t initialized_mask = 0;
    // The open DEST batch: units (member, first score column, width) and the tiles they occupy.
    uint8_t unit_g[kDst];
    uint8_t unit_c0[kDst];
    uint8_t unit_cw[kDst];
    uint32_t uu = 0;
    uint32_t uu_tiles = 0;
    uint32_t pk_w = 0;  // pack MOP width in effect (0: unknown)
    // S5/S6 interleave (see the step).
    uint8_t cbatch[kMaxMembers];   // S5 batch holding member g's last unit, valid for g < n_done
    uint32_t n_done = 0;           // members whose S5 units have all been issued (members finish in order)
    uint32_t batch_idx = 0;        // S5 batches flushed so far (the open batch has this index)
    bool batch_completes = false;  // the open S5 batch holds some member's last unit
    uint32_t init_cw = 0;          // broadcast-subtract MOP width in effect (0: a matmul MOP is)
    uint32_t pv_stride = 0;        // P@V MOP row stride in effect (0: the subtract MOP is)
    bool mm_fmt = false;           // unpacker srcA format: V (true) or the intermediate (false)
    bool v_waited = false;
    uint32_t s6_next = 0;  // next member whose P@V is pending
};

// P@V of one member of the compact step (S6): pv_w-wide sub-blocks of its Sq_chunk_t x vDHt output, one DEST batch
// each, K dimension = the member's kept width (its run table; row stride w in the compact scratch), packed to the
// slot's rows of cb_prev_out and L1-accumulated for an initialized slot. The matmul MOP is (re)issued when the
// stride in effect differs, the pack MOP when the width in effect differs (pk_w is shared with the S5 packs).
// Outlined: it is reached from the S5 flush sites of the interleaved S5/S6 loop.
template <
    uint32_t Sq_chunk_t,
    uint32_t vDHt,
    uint32_t pv_w,
    uint32_t kMaxRuns,
    uint32_t cb_qkt_im,
    uint32_t cb_v_in,
    uint32_t cb_prev_out>
static __attribute__((noinline, noclone)) void sol_res_pv_member(
    const uint32_t slot,
    const bool acc,
    const uint32_t w,
    const uint32_t off_g,
    const uint8_t* run_cols,
    const uint8_t* run_ws,
    const uint32_t n_runs,
    uint32_t& pv_stride,
    uint32_t& pk_w) {
    if (pv_stride != w) {
        sdpa_pv_mm_setup(cb_qkt_im, cb_v_in, false, pv_w, Sq_chunk_t, w);
        pv_stride = w;
    }
    const bool blocked = should_use_blocked_pack_width(pv_w);
    const uint32_t want = blocked ? pv_w : 1u;
    const uint32_t out_row0 = slot * Sq_chunk_t;
    for (uint32_t v0 = 0; v0 < vDHt; v0 += pv_w) {
        tile_regs_acquire();
        uint32_t col = 0;
        for (uint32_t s = 0; s < n_runs; ++s) {
            const uint32_t rw = run_ws[s];
            for (uint32_t t = 0; t < rw; ++t) {
                sdpa_matmul_block_no_mop(
                    cb_qkt_im,
                    cb_v_in,
                    off_g + col + t,
                    (static_cast<uint32_t>(run_cols[s]) + t) * vDHt + v0,
                    0,
                    false,
                    pv_w,
                    Sq_chunk_t,
                    w);
            }
            col += rw;
        }
        tile_regs_commit();
        tile_regs_wait();
        if (want != pk_w) {
            configure_pack_width(cb_prev_out, want);
            pk_w = want;
        }
        PACK((llk_pack_reconfig_l1_acc(acc ? 1 : 0)));
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            const uint32_t idx = (out_row0 + r) * vDHt + v0;
            if (blocked) {
                sdpa_pack_tile_ooo(r * pv_w, cb_prev_out, idx);
            } else {
                for (uint32_t c = 0; c < pv_w; ++c) {
                    sdpa_pack_tile_ooo(r * pv_w + c, cb_prev_out, idx + c);
                }
            }
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        tile_regs_release();
    }
}

// S1 flush of the compact step: commit the open Q@K^T DEST batch and pack its units into the compact scratch.
template <uint32_t Sq_chunk_t, uint32_t cb_qkt_im, uint32_t kDst, uint32_t kMaxMembers>
static __attribute__((noinline, noclone)) void sol_res_s1_flush(SolResStepState<kDst, kMaxMembers>& st) {
    if (st.uu == 0) {
        return;
    }
    tile_regs_commit();
    tile_regs_wait();
    sol_res_pack_units<Sq_chunk_t, cb_qkt_im>(st.unit_g, st.unit_c0, st.unit_cw, st.uu, st.off, st.W, st.pk_w);
    tile_regs_release();
    st.uu = 0;
    st.uu_tiles = 0;
}

// S5 flush of the compact step: commit the open subtract batch, exp it in DEST on the pack thread (packer ReLU
// zero, so the subtracted-max underflow packs as 0), pack the units back in place, post the batch's PACK_DONE
// token if it holds some member's last unit, then issue the P@V (S6) of every member whose completion batch is at
// least one S5 batch old (all remaining members when `final`). See the step's S5 + S6 comment for the ordering.
template <
    uint32_t Sq_chunk_t,
    uint32_t vDHt,
    uint32_t pv_w,
    uint32_t kMaxRuns,
    uint32_t kDst,
    uint32_t kMaxMembers,
    uint32_t v_entry_tiles,
    uint32_t cb_qkt_im,
    uint32_t cb_v_in,
    uint32_t cb_prev_out>
static __attribute__((noinline, noclone)) void sol_res_s5_flush(
    SolResStepState<kDst, kMaxMembers>& st, const bool final) {
    if (st.uu > 0) {
        tile_regs_commit();
        tile_regs_wait();
        PACK((llk_pack_relu_config(ReluConfig::zero())));
        constexpr int iterations = 32;
        constexpr VectorMode vector_mode_exp = VectorMode::None;
        for (uint32_t i = 0; i < st.uu_tiles; ++i) {
            exp_packthread_tile<true, false, InputClamping::None, iterations>(i, vector_mode_exp);
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        sol_res_pack_units<Sq_chunk_t, cb_qkt_im>(st.unit_g, st.unit_c0, st.unit_cw, st.uu, st.off, st.W, st.pk_w);
        PACK((llk_pack_relu_config(ReluConfig::none())));
        if (st.batch_completes) {
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
        }
        tile_regs_release();
        st.uu = 0;
        st.uu_tiles = 0;
        st.batch_completes = false;
        ++st.batch_idx;
    }
    while (st.s6_next < st.n_done && (final || static_cast<uint32_t>(st.cbatch[st.s6_next]) + 1u < st.batch_idx)) {
        const uint32_t g = st.s6_next;
        if (!st.v_waited) {
            CircularBuffer(cb_v_in).wait_front(v_entry_tiles);
            st.v_waited = true;
        }
        if (g == 0 || st.cbatch[g] != st.cbatch[g - 1]) {
            UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
            UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
        }
        if (!st.mm_fmt) {
            sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_v_in, cb_qkt_im, cb_qkt_im>();
            st.mm_fmt = true;
        }
        st.init_cw = 0;
        const uint32_t slot = st.slots[g];
        sol_res_pv_member<Sq_chunk_t, vDHt, pv_w, kMaxRuns, cb_qkt_im, cb_v_in, cb_prev_out>(
            slot,
            ((st.initialized_mask >> slot) & 1u) != 0,
            st.W[g],
            st.off[g],
            st.run_cols + g * kMaxRuns,
            st.run_ws + g * kMaxRuns,
            st.n_runs[g],
            st.pv_stride,
            st.pk_w);
        ++st.s6_next;
    }
}

// ---- M9 (Tier 2) compact per-piece step ----
// ---- Compact per-piece step ----
/**
 * One step over a group of resident chunks for one piece (or pooled entry). The per-tile overheads (matmul calls,
 * address-programmed packs, per-stage round trips) dominate a sparse step, so the work is issued as follows:
 *  1. Compact score layout. Member g's scores occupy Sq_chunk_t rows of exactly W_g tiles (its kept width) at
 *     offset off[g] = sum of the earlier members' Sq_chunk_t x W; the row stride is W_g, not the piece width. A
 *     step can then hold every chunk of the head that keeps a block of the piece (the capacity is the scratch's
 *     tile count, qk_cap), so one step covers the piece, and each stage runs several DEST batches back to back
 *     (half-sync DEST overlaps math and pack across them).
 *  2. Runs of adjacent kept blocks are one run of 2, 4, ... tiles (run_ws[g * kMaxRuns + s]); a unit of up to
 *     qk_cw_cap columns is one matmul call and, when it is >= MIN_BLOCKED_PACK_TILES wide, one pack call per row
 *     instead of one per tile. The same for the exp stage.
 *  3. P@V uses the widest sub-block DEST allows (pv_w = min(vDHt, kDst / Sq_chunk_t)): fewer matmul calls per
 *     member and blocked output packs.
 * Members with equal widths should be adjacent in the group (the caller sorts by width): the row-max reduce, the
 * subtract and the two P@V setups are configured per width and re-issued only when it changes.
 *  4. The exp stage and the P@V are interleaved per member (see S5 + S6 below): the P@V of the members finished
 *     in one S5 batch runs under the exp and packs of the next, with one PACK_DONE token per S5 batch.
 * S2-S4 hold the lazy running max; the row sum is P x e0; the pooled bias, the in-place slot state contract and
 * the PACK_DONE tokens are described at their stages. run_cols / run_ws / n_runs are per member; a pooled entry
 * is one run of its active width from column 0 with apply_col_bias.
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_col_identity,
    uint32_t cb_max_scratch,
    uint32_t cb_exp_max_diff,
    uint32_t cb_max_in,
    uint32_t cb_sum_in,
    uint32_t cb_prev_out,
    uint32_t cb_out,
    uint32_t cb_sol_bias,
    uint32_t v_cb_physical_width_t,
    uint32_t qk_cap,
    uint32_t stats_cap>
static __attribute__((noinline, noclone)) void sdpa_resident_piece_step_compact(
    const uint8_t* slots,
    const uint32_t n_slots,
    const uint8_t* run_cols,
    const uint8_t* run_ws,
    const uint8_t* n_runs,
    const uint32_t initialized_mask,
    const bool apply_col_bias) {
    constexpr uint32_t kDst = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t kBlockTiles = 2;
    constexpr uint32_t kMaxRuns = Sk_chunk_t / kBlockTiles;
    constexpr uint32_t kMaxMembers = 32;
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t kt_entry_tiles = DHt * Sk_chunk_t;
    constexpr uint32_t v_entry_tiles = Sk_chunk_t * v_cb_physical_width_t;
    static_assert(v_cb_physical_width_t == vDHt, "resident mode addresses V rows at a vDHt tile stride");
    static_assert(Sk_chunk_t % kBlockTiles == 0, "a piece holds whole 64-token blocks");
    constexpr uint32_t max_cols_per_acq = kDst / Sq_chunk_t;
    static_assert(max_cols_per_acq >= 1, "one Q chunk row of one tile column must fit DEST");
    constexpr uint32_t qk_cw_cap = qkt_subblock_w < max_cols_per_acq ? qkt_subblock_w : max_cols_per_acq;
    constexpr uint32_t pv_w = vDHt < max_cols_per_acq ? vDHt : max_cols_per_acq;
    static_assert(vDHt % pv_w == 0, "vDHt must be a multiple of the P@V sub-block width");
    constexpr uint32_t pv_unit_tiles = Sq_chunk_t * pv_w;
    static_assert(pv_unit_tiles <= kDst, "a P@V sub-block must fit DEST");
    constexpr uint32_t stat_slots_per_acq = kDst / Sq_chunk_t;
    constexpr uint32_t alpha_slots_per_acq = kDst / (2 * Sq_chunk_t);
    static_assert(alpha_slots_per_acq >= 1, "the alpha and the new max of one slot must fit DEST together");
    static_assert(Sq_chunk_t <= kDst, "salad_correct_fused needs one Q chunk column in DEST");
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    ASSERT(n_slots >= 1 && n_slots <= kMaxMembers);
    const auto slot_init = [&](uint32_t g) -> bool { return ((initialized_mask >> slots[g]) & 1u) != 0; };
    const auto run_col = [&](uint32_t g, uint32_t s) -> uint32_t {
        return static_cast<uint32_t>(run_cols[g * kMaxRuns + s]);
    };
    const auto run_wd = [&](uint32_t g, uint32_t s) -> uint32_t {
        return static_cast<uint32_t>(run_ws[g * kMaxRuns + s]);
    };
    // Member widths and compact offsets.
    uint8_t W[kMaxMembers];
    uint16_t off[kMaxMembers];
    uint32_t total = 0;
    uint32_t q_wait_tiles = 0;
    for (uint32_t g = 0; g < n_slots; ++g) {
        uint32_t w = 0;
        for (uint32_t s = 0; s < n_runs[g]; ++s) {
            w += run_wd(g, s);
        }
        ASSERT(n_runs[g] >= 1 && w >= 1 && w <= Sk_chunk_t);
        W[g] = static_cast<uint8_t>(w);
        off[g] = static_cast<uint16_t>(total);
        total += Sq_chunk_t * w;
        const uint32_t t = (static_cast<uint32_t>(slots[g]) + 1) * q_chunk_tiles;
        q_wait_tiles = t > q_wait_tiles ? t : q_wait_tiles;
    }
    ASSERT(total <= qk_cap);
    const auto sidx = [&](uint32_t g, uint32_t r, uint32_t c) -> uint32_t { return off[g] + r * W[g] + c; };

    // ---- S1: Q@K^T into the compact scratch. Units: (member, run, column batch of cw <= qk_cw_cap). ----
    CircularBuffer(cb_qkt_im).reserve_back(qk_cap);
    CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles);
    CircularBuffer(cb_q_in).wait_front(q_wait_tiles);
    sdpa_maybe_pack_reconfig_data_format<cb_out, cb_qkt_im>();
    sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in>();
    SolResStepState<kDst, kMaxMembers> st{};
    st.slots = slots;
    st.run_cols = run_cols;
    st.run_ws = run_ws;
    st.n_runs = n_runs;
    st.W = W;
    st.off = off;
    st.n_slots = n_slots;
    st.initialized_mask = initialized_mask;
    {
        uint32_t mm_cw = 0;
        for (uint32_t g = 0; g < n_slots; ++g) {
            uint32_t col = 0;
            for (uint32_t s = 0; s < n_runs[g]; ++s) {
                const uint32_t rw = run_wd(g, s);
                for (uint32_t q0 = 0; q0 < rw;) {
                    const uint32_t cw = rw - q0 < qk_cw_cap ? rw - q0 : qk_cw_cap;
                    if (st.uu_tiles + Sq_chunk_t * cw > kDst) {
                        sol_res_s1_flush<Sq_chunk_t, cb_qkt_im, kDst, kMaxMembers>(st);
                    }
                    if (st.uu == 0) {
                        tile_regs_acquire();
                    }
                    if (cw != mm_cw) {
                        sdpa_mm_init(cb_q_in, cb_kt_in, true, cw, Sq_chunk_t, DHt);
                        mm_cw = cw;
                    }
                    uint32_t in0 = slots[g] * q_chunk_tiles;
                    uint32_t in1 = run_col(g, s) + q0;
                    for (uint32_t d = 0; d < DHt; ++d) {
                        sdpa_matmul_block_no_mop(cb_q_in, cb_kt_in, in0, in1, st.uu_tiles, true, cw, Sq_chunk_t, DHt);
                        in0++;
                        in1 += Sk_chunk_t;
                    }
                    st.unit_g[st.uu] = static_cast<uint8_t>(g);
                    st.unit_c0[st.uu] = static_cast<uint8_t>(col + q0);
                    st.unit_cw[st.uu] = static_cast<uint8_t>(cw);
                    ++st.uu;
                    st.uu_tiles += Sq_chunk_t * cw;
                    q0 += cw;
                }
                col += rw;
            }
        }
        sol_res_s1_flush<Sq_chunk_t, cb_qkt_im, kDst, kMaxMembers>(st);
    }
    sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
    if (apply_col_bias) {
        // Pooled entry: every member one contiguous run from column 0 with its own bias tile row (the same row for
        // each of its Q tile rows), consumed one at a time (the reader pushes them in member order; the CB holds
        // one row).
        for (uint32_t g = 0; g < n_slots; ++g) {
            ASSERT(n_runs[g] == 1 && run_col(g, 0) == 0);
            CircularBuffer(cb_sol_bias).wait_front(Sk_chunk_t);
            begin_mask_l1_accumulate<false>(cb_qkt_im, cb_sol_bias);
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                l1_acc_tile_run(cb_sol_bias, 0, cb_qkt_im, sidx(g, r, 0), W[g]);
            }
            end_mask_l1_accumulate();
            CircularBuffer(cb_sol_bias).pop_front(Sk_chunk_t);
        }
    }
    cb_push_back_hold_wr_ptr(cb_qkt_im, qk_cap);

    // Lazy rescale. S2 computes every member's new row max (this piece merged with the stored max) into the
    // scratch. The step rescales (S3 alpha + S4) only when some initialized row's max grew by more than
    // kLazyThrRaw raw logit units; otherwise the stored max stays and S5 subtracts it, so the exp's input
    // (score - stored max) * scale is <= 0.5: inside the approximate exp's safe range (see the function comment)
    // and with bf16 resolution <= 1/256 of the argument. A slot's first step (uninitialized) only copies its new
    // max into the slot. Either way S5 subtracts cb_max_in, after the S3 packs are drained.
    constexpr float kScale = __builtin_bit_cast(float, scale_fp32);
    constexpr float kLazyThrRaw = 0.5f / kScale;
    constexpr uint16_t kLazyThrBits = static_cast<uint16_t>(__builtin_bit_cast(uint32_t, kLazyThrRaw) >> 16);
    uint32_t n_init = 0;
    for (uint32_t g = 0; g < n_slots; ++g) {
        n_init += slot_init(g) ? 1u : 0u;
    }
    bool rescale = false;
    CircularBuffer(cb_qkt_im).wait_front(qk_cap);
    {
        // ---- S2: row max over each member's W_g columns, eltwise-max with the slot's stored max, into the scratch.
        CircularBuffer(cb_max_scratch).reserve_back(stats_cap);
        configure_single_tile_pack(cb_max_scratch);
        uint32_t last_w = 0;
        for (uint32_t g0 = 0; g0 < n_slots; g0 += stat_slots_per_acq) {
            const uint32_t n_g = n_slots - g0 < stat_slots_per_acq ? n_slots - g0 : stat_slots_per_acq;
            tile_regs_acquire();
            bool any_init = false;
            for (uint32_t gg = 0; gg < n_g; ++gg) {
                any_init |= slot_init(g0 + gg);
            }
            if (any_init) {
                if (last_w != 0) {
                    reduce_block_max_row_uninit_runtime(cb_qkt_im, false, false);
                    last_w = 0;
                }
                sdpa_reduce_copy_tile_to_dst_init_short(cb_max_in);
                for (uint32_t gg = 0; gg < n_g; ++gg) {
                    if (!slot_init(g0 + gg)) {
                        continue;
                    }
                    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                        copy_tile(cb_max_in, slots[g0 + gg] * Sq_chunk_t + r, gg * Sq_chunk_t + r);
                    }
                }
            }
            for (uint32_t gg = 0; gg < n_g; ++gg) {
                const uint32_t w = W[g0 + gg];
                if (w != last_w) {
                    if (last_w != 0) {
                        reduce_block_max_row_uninit_runtime(cb_qkt_im, false, false);
                    }
                    reduce_block_max_row_init_runtime(cb_max_scratch, w, cb_qkt_im, cb_identity_scale_in, false);
                    last_w = w;
                }
                for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                    reduce_block_max_row_runtime(
                        cb_qkt_im, cb_identity_scale_in, sidx(g0 + gg, r, 0), gg * Sq_chunk_t + r, false, false);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t gg = 0; gg < n_g; ++gg) {
                for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                    sdpa_pack_tile_ooo(gg * Sq_chunk_t + r, cb_max_scratch, slots[g0 + gg] * Sq_chunk_t + r);
                }
            }
            tile_regs_release();
        }
        if (last_w != 0) {
            reduce_block_max_row_uninit_runtime(cb_qkt_im, false, false);
        }
        cb_push_back_hold_wr_ptr(cb_max_scratch, stats_cap);
        CircularBuffer(cb_max_scratch).wait_front(stats_cap);
        CircularBuffer(cb_exp_max_diff).reserve_back(stats_cap);
        cb_push_back_hold_wr_ptr(cb_exp_max_diff, stats_cap);
        CircularBuffer(cb_exp_max_diff).wait_front(stats_cap);

        // ---- S2b: the rescale test. d = new max - stored max of every initialized member (>= 0, column 0),
        //      column-max-reduced into one tile whose [0][0] is the largest growth. The unpack thread's RISC reads
        //      that word once the pack has landed (it polls the PACK_DONE token itself: a Tensix-side wait would not
        //      order its load) and read_tile_value hands the same word to the math and pack threads, so the three
        //      threads take the same branch. ----
        if (n_init > 0) {
            // d = new - stored max of every initialized member, DEST batches of up to kDst tiles, packed to the
            // alpha scratch pages 0..nd-1 (nd <= n_slots * Sq_chunk_t <= stats_cap).
            uint32_t nd = 0;
            configure_single_tile_pack(cb_exp_max_diff);
            for (uint32_t g = 0; g < n_slots;) {
                uint32_t nb = 0;
                tile_regs_acquire();
                sub_init(cb_max_scratch, cb_max_in);
                for (; g < n_slots && nb + Sq_chunk_t <= kDst; ++g) {
                    if (!slot_init(g)) {
                        continue;
                    }
                    const uint32_t row0 = slots[g] * Sq_chunk_t;
                    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                        sub_tiles(cb_max_scratch, cb_max_in, row0 + r, row0 + r, nb++);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < nb; ++i) {
                    sdpa_pack_tile_ooo(i, cb_exp_max_diff, nd + i);
                }
                tile_regs_release();
                nd += nb;
            }
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
            UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
            UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
            tile_regs_acquire();
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_exp_max_diff, cb_identity_scale_in, cb_exp_max_diff);
            for (uint32_t i = 0; i < nd; ++i) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_exp_max_diff, cb_identity_scale_in, i, 0, 0);
            }
            reduce_uninit();
            tile_regs_commit();
            tile_regs_wait();
            configure_single_tile_pack(cb_exp_max_diff);
            sdpa_pack_tile_ooo(0, cb_exp_max_diff, 0);
            tile_regs_release();
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
            UNPACK({
                ckernel::tensix_sync();
                while (ckernel::semaphore_read(semaphore::PACK_DONE) == 0) {
                }
                t6_semaphore_get<>(semaphore::PACK_DONE);
                invalidate_l1_cache();
            });
            const uint32_t growth = ckernel::read_tile_value(cb_exp_max_diff, 0, 0);
            rescale = (growth & 0xFFFFu) > kLazyThrBits;
        }

        // ---- S3: rescale: alpha = exp((old max - new max) * scale) into cb_exp_max_diff and the new max into every
        //      member's slot; otherwise only the first-step (uninitialized) members get their max copied. ----
        if (rescale) {
            exp_packthread_tile_init<EXP_APPROX_MODE>();
        }
        for (uint32_t g0 = 0; g0 < n_slots; g0 += alpha_slots_per_acq) {
            const uint32_t n_g = n_slots - g0 < alpha_slots_per_acq ? n_slots - g0 : alpha_slots_per_acq;
            uint32_t n_alpha = 0;
            uint32_t n_copy = 0;
            for (uint32_t gg = 0; gg < n_g; ++gg) {
                const bool init = slot_init(g0 + gg);
                n_alpha += (rescale && init) ? 1u : 0u;
                n_copy += (rescale || !init) ? 1u : 0u;
            }
            if (n_copy == 0) {
                continue;
            }
            tile_regs_acquire();
            if (n_alpha > 0) {
                sub_init(cb_max_in, cb_max_scratch);
                uint32_t a = 0;
                for (uint32_t gg = 0; gg < n_g; ++gg) {
                    if (!slot_init(g0 + gg)) {
                        continue;
                    }
                    const uint32_t row0 = slots[g0 + gg] * Sq_chunk_t;
                    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                        sub_tiles(cb_max_in, cb_max_scratch, row0 + r, row0 + r, a * Sq_chunk_t + r);
                    }
                    ++a;
                }
            }
            const uint32_t copy_base = n_alpha * Sq_chunk_t;
            copy_init(cb_max_scratch);
            {
                uint32_t c = 0;
                for (uint32_t gg = 0; gg < n_g; ++gg) {
                    if (!rescale && slot_init(g0 + gg)) {
                        continue;
                    }
                    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                        copy_tile(cb_max_scratch, slots[g0 + gg] * Sq_chunk_t + r, copy_base + c * Sq_chunk_t + r);
                    }
                    ++c;
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            if (n_alpha > 0) {
                for (uint32_t i = 0; i < copy_base; ++i) {
                    PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(i)));
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                configure_single_tile_pack(cb_exp_max_diff);
                uint32_t a = 0;
                for (uint32_t gg = 0; gg < n_g; ++gg) {
                    if (!slot_init(g0 + gg)) {
                        continue;
                    }
                    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                        sdpa_pack_tile_ooo(a * Sq_chunk_t + r, cb_exp_max_diff, slots[g0 + gg] * Sq_chunk_t + r);
                    }
                    ++a;
                }
            }
            configure_single_tile_pack(cb_max_in);
            {
                uint32_t c = 0;
                for (uint32_t gg = 0; gg < n_g; ++gg) {
                    if (!rescale && slot_init(g0 + gg)) {
                        continue;
                    }
                    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                        sdpa_pack_tile_ooo(copy_base + c * Sq_chunk_t + r, cb_max_in, slots[g0 + gg] * Sq_chunk_t + r);
                    }
                    ++c;
                }
            }
            tile_regs_release();
        }
        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));

        // ---- S4 (rescale only). ----
        if (rescale) {
            for (uint32_t g = 0; g < n_slots; ++g) {
                if (!slot_init(g)) {
                    continue;
                }
                salad_correct_fused<Sq_chunk_t, vDHt, kDst>(
                    cb_prev_out, cb_sum_in, cb_exp_max_diff, cb_prev_out, cb_sum_in, slots[g], slots[g], slots[g]);
            }
        }
    }  // S2-S4

    // ---- S5 + S6, interleaved per member (M10). S5: in-place exp of the scores, per member over its W_g columns
    //      in units of <= max_cols_per_acq (the broadcast subtract on MATH, the exp and the packs on PACK). S6: P@V
    //      per member (sol_res_pv_member). The two stages load complementary threads (S5 is pack-thread bound, S6
    //      math/unpack bound), so instead of draining S5 for every member before the first P@V, the P@V of the
    //      members whose last S5 unit was packed in batch k is issued right after S5 batch k+1: with the half-sync
    //      DEST, MATH runs those matmuls while PACK is still on the exp and packs of batch k+1, and PACK's output
    //      packs of the P@V overlap the subtracts of the next S5 batch. Ordering: a P@V reads its member's exp'd
    //      scores from L1, so PACK posts one PACK_DONE token after every S5 batch that holds some member's last
    //      unit, and UNPACK takes one before the first P@V of that batch's members (posts and takes are both in
    //      batch order, one per completion batch). A member's P@V follows its completion batch by one S5 batch, so
    //      only a few tokens are ever outstanding; PACK_DONE's capacity in resident mode is 15
    //      (init_sdpa_streaming_semaphores<15>), so a post never saturates the counter. The MOPs are re-issued on
    //      every stage switch (the broadcast subtract and the matmul share the unpacker/math configuration), as is
    //      the unpacker's srcA format when V is block float (a compile-time no-op otherwise). ----
    exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
    CircularBuffer(cb_col_identity).wait_front(1);
    {
        constexpr auto s5_flush = sol_res_s5_flush<
            Sq_chunk_t,
            vDHt,
            pv_w,
            kMaxRuns,
            kDst,
            kMaxMembers,
            v_entry_tiles,
            cb_qkt_im,
            cb_v_in,
            cb_prev_out>;
        st.pk_w = 0;  // S2/S3 packed at width 1 through configure_single_tile_pack: unknown to the shared tracker
        for (uint32_t g = 0; g < n_slots; ++g) {
            const uint32_t w = W[g];
            for (uint32_t c0 = 0; c0 < w;) {
                const uint32_t cw = w - c0 < max_cols_per_acq ? w - c0 : max_cols_per_acq;
                if (st.uu_tiles + Sq_chunk_t * cw > kDst) {
                    s5_flush(st, false);
                }
                if (cw != st.init_cw) {
                    // The bcast-subtract MOP is configured for a column width: flush the open batch before changing it
                    // (a flush may also have issued P@V matmuls, which leave init_cw at 0).
                    s5_flush(st, false);
                    if (st.mm_fmt) {
                        sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
                        st.mm_fmt = false;
                    }
                    sub_bcast_cols_init_short_custom(cb_qkt_im, cb_max_in, cw);
                    st.init_cw = cw;
                    st.pv_stride = 0;
                }
                if (st.uu == 0) {
                    tile_regs_acquire();
                }
                for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                    sub_tiles_bcast_cols_custom(
                        cb_qkt_im, cb_max_in, sidx(g, r, c0), slots[g] * Sq_chunk_t + r, st.uu_tiles + r * cw, cw);
                }
                st.unit_g[st.uu] = static_cast<uint8_t>(g);
                st.unit_c0[st.uu] = static_cast<uint8_t>(c0);
                st.unit_cw[st.uu] = static_cast<uint8_t>(cw);
                ++st.uu;
                st.uu_tiles += Sq_chunk_t * cw;
                c0 += cw;
            }
            st.cbatch[g] = static_cast<uint8_t>(st.batch_idx);
            st.batch_completes = true;
            st.n_done = g + 1;
        }
        s5_flush(st, true);
    }

    // ---- S6b: the row sums P@e0 in batches of stat_slots_per_acq members (the matmul takes the member's row
    //      stride W_g, re-issued when it changes). Every S5 token has been taken by now, so all exp'd scores are in
    //      L1. ----
    sdpa_maybe_reconfig_data_format<cb_v_in, cb_col_identity, cb_qkt_im, cb_qkt_im>();
    configure_single_tile_pack(cb_sum_in);
    {
        uint32_t rs_stride = 0;
        for (uint32_t g0 = 0; g0 < n_slots; g0 += stat_slots_per_acq) {
            const uint32_t n_g = n_slots - g0 < stat_slots_per_acq ? n_slots - g0 : stat_slots_per_acq;
            tile_regs_acquire();
            for (uint32_t gg = 0; gg < n_g; ++gg) {
                const uint32_t g = g0 + gg;
                if (W[g] != rs_stride) {
                    sdpa_pv_mm_setup(cb_qkt_im, cb_col_identity, false, 1, Sq_chunk_t, W[g]);
                    rs_stride = W[g];
                }
                for (uint32_t c = 0; c < W[g]; ++c) {
                    sdpa_matmul_block_no_mop(
                        cb_qkt_im, cb_col_identity, sidx(g, 0, c), 0, gg * Sq_chunk_t, false, 1, Sq_chunk_t, W[g]);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            bool acc_on = false;
            PACK((llk_pack_reconfig_l1_acc(0)));
            for (uint32_t gg = 0; gg < n_g; ++gg) {
                const bool acc = slot_init(g0 + gg);
                if (acc != acc_on) {
                    PACK((llk_pack_reconfig_l1_acc(acc ? 1 : 0)));
                    acc_on = acc;
                }
                const uint32_t out_row0 = slots[g0 + gg] * Sq_chunk_t;
                for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                    sdpa_pack_tile_ooo(gg * Sq_chunk_t + r, cb_sum_in, out_row0 + r);
                }
            }
            if (acc_on) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
            tile_regs_release();
        }
    }
    sdpa_maybe_reconfig_data_format<cb_col_identity, cb_qkt_im, cb_qkt_im, cb_qkt_im>();

    CircularBuffer(cb_qkt_im).pop_front(qk_cap);
    CircularBuffer(cb_max_scratch).pop_front(stats_cap);
    CircularBuffer(cb_exp_max_diff).pop_front(stats_cap);
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

/**
 * Sol-Attn resident mode: one ring iteration.
 *
 * Every K/V entry the reader pushes is consumed by the steps that need it and popped once, after the last of them.
 * The per-chunk flash state (running sum, running max, output accumulator) lives in the restore CBs {cb_sum_in,
 * cb_max_in, cb_prev_out}, one SLOT per resident chunk in Q-chunk order, updated in place (see
 * sdpa_resident_piece_step_compact for the contract). The slot CBs are filled once, on the first active ring iteration
 * (exactly full, write pointer held at the base), and are neither pushed nor popped until the last ring iteration,
 * where normalize_row_streaming consumes the n_res slots from the front in order and the unused slots are popped.
 *
 * Steps are per piece: per streamed piece, one step per group of resident chunks of the head that keep a block of
 * it (grouped up to the score scratch capacity), each chunk with its own run table of kept 64-token blocks (its Q
 * rows against those blocks' K^T columns / V rows, addressed in place in the staged entry). A piece streamed only
 * for another chain member is drained. The reader forces block 0 of every shard kept for every chunk, so the first
 * step of a head on piece 0 creates every slot's state before any other piece.
 * Dropped blocks are pooled by the column bias of the pooled chunks that follow the local pieces on the first
 * active ring iteration: one full-width step per resident chunk of the head, each with its own bias row (pushed by
 * the reader in Q-chunk order after the pooled K/V entry).
 *
 * The Q chunks a core owns can span two heads (a contiguous flat split); the reader streams the heads' pieces
 * interleaved per k, in head order, and this function mirrors that order. On the last ring iteration the normalized
 * output of every resident chunk is left in cb_out, in Q-chunk order, for the writer, and the resident Q chunks are
 * popped.
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t NH,
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
    uint32_t cb_max_scratch,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_mask_in,
    uint32_t cb_max_in,
    uint32_t cb_prev_out,
    uint32_t cb_out,
    uint32_t cb_sum_in,
    uint32_t local_padded_Nt,
    uint32_t kv_region_Nt,
    uint32_t chunk_size_t,
    uint32_t v_cb_physical_width_t,
    bool global_n_mask_enabled,
    bool local_n_mask_enabled,
    uint32_t cb_sol_plan,
    uint32_t cb_sol_bias,
    uint32_t sol_pooled_chunks,
    uint32_t sol_pooled_last_tiles,
    uint32_t sol_resident_slots,
    uint32_t sol_res_block_words,
    uint32_t kv_cb_pages,
    uint32_t sol_res_plan_rec_base,
    bool sol_res_no_dummies,
    typename MaskCtx>
void sdpa_ring_resident(
    const uint32_t global_q_start,
    const uint32_t global_q_end,
    const uint32_t num_q_chunks,
    const uint32_t ring_id,
    const uint32_t num_local_k_chunks,
    const uint32_t logical_nt,
    const bool ring_iter_needs_global_n_mask,
    const bool local_n_needs_masking,
    const uint32_t global_n_mask_chunk_id,
    const uint32_t local_n_mask_chunk_id,
    RingAccumulatorState& acc_state,
    const bool is_last_ring_iter,
    const bool is_first_active_iter,
    const MaskCtx& lw_mask) {
    // PACK_DONE capacity 15: the compact step's interleaved S5/S6 posts one token per S5 batch (see the step).
    init_sdpa_streaming_semaphores<15>();
    static_assert(sol_resident_slots >= 1 && sol_resident_slots <= 32, "resident state is tracked in a 32-bit mask");
    // The step reconfigures the unpacker/packer only around the Q@K^T and P@V matmuls (whose inputs may be block
    // float) and otherwise assumes every state, scratch and scale CB carries the intermediate format.
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    static_assert(
        !sdpa_unpack_format_changed<cb_qkt_im, cb_prev_out>() && !sdpa_unpack_format_changed<cb_qkt_im, cb_max_in>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_sum_in>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_max_scratch>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_exp_max_diff>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_identity_scale_in>(),
        "resident mode: the state, scratch and scale CBs must use the step's intermediate format");
#endif
#ifdef TRISC_PACK
    static_assert(
        pack_dst_format[cb_prev_out] == pack_dst_format[cb_qkt_im] &&
            pack_dst_format[cb_max_in] == pack_dst_format[cb_qkt_im] &&
            pack_dst_format[cb_sum_in] == pack_dst_format[cb_qkt_im] &&
            pack_dst_format[cb_max_scratch] == pack_dst_format[cb_qkt_im] &&
            pack_dst_format[cb_exp_max_diff] == pack_dst_format[cb_qkt_im],
        "resident mode: the state and scratch CBs must use the step's intermediate format");
#endif
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t kt_entry_tiles = DHt * Sk_chunk_t;
    constexpr uint32_t v_entry_tiles = Sk_chunk_t * v_cb_physical_width_t;
    constexpr uint32_t norm_dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;

    // 64-token blocks of 2 tiles; a piece holds kBlocksPerPiece of them. The plan page carries this shard's
    // kept-block bits per resident chunk (block b: word b/32, bit b%32; only full blocks, block 0 forced).
    constexpr uint32_t kBlockTiles = 2;
    constexpr uint32_t kBlocksPerPiece = Sk_chunk_t / kBlockTiles;
    constexpr uint32_t kBW = sol_res_block_words;
    static_assert(Sk_chunk_t % kBlockTiles == 0, "a piece holds whole 64-token blocks");
    static_assert(kBW >= 1 && kBW + 3 <= 16, "the plan header holds the block words, the union bits and the flag");
    static_assert(kBlocksPerPiece <= 255 && Sk_chunk_t <= 255, "run tables are byte-indexed");

    // CB capacities, as the program factory allocates them in resident mode: the score
    // scratch holds max(qk_tiles, slots x Sq_chunk_t x kBlockTiles); the max/alpha scratches and the max/sum slot
    // CBs hold Sq_chunk_t per slot; the out slot CB holds Sq_chunk_t x vDHt per slot. The step pushes and pops the
    // scratches at full capacity and the slot CBs are filled at full capacity, so the counts must match exactly.
    // kGroupCap sizes the score scratch: one piece for up to kGroupCap resident chunks at the full piece width
    // (the factory's kSolResidentGroup).
    constexpr uint32_t kGroupCap = sol_resident_slots < 4 ? sol_resident_slots : 4;
    constexpr uint32_t kQkTiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t kQkCap =
        kQkTiles > kGroupCap * Sq_chunk_t * Sk_chunk_t ? kQkTiles : kGroupCap * Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t kStatsCap = sol_resident_slots * Sq_chunk_t;
    constexpr uint32_t kOutCap = sol_resident_slots * out_tiles;
#ifdef TRISC_PACK
    // ASSERT is a statement, not an expression: keep it out of the PACK(...) expression macro.
    ASSERT(get_local_cb_interface(cb_qkt_im).fifo_num_pages == kQkCap);
    ASSERT(get_local_cb_interface(cb_max_scratch).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_exp_max_diff).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_max_in).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_sum_in).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_prev_out).fifo_num_pages == kOutCap);
#endif

    const uint32_t n_res = global_q_end - global_q_start;
    ASSERT(n_res >= 1 && n_res <= sol_resident_slots);

    // Resident chunk i: head, kept-block words and the head chain's kept-piece union (plan words from the reader).
    uint32_t res_head[sol_resident_slots];
    uint32_t res_blk[sol_resident_slots][kBW];
    uint32_t union_lo[sol_resident_slots];
    uint32_t union_hi[sol_resident_slots];
    uint32_t do_pooled = 0;
    if constexpr (sol_res_plan_rec_base != 0) {
        // One plan page per ring iteration: chunk i's record [kBW block words][union lo, hi][do_pooled] at word
        // rec_base + i * (kBW + 3). Popped once, early, so the reader can push the next iteration's plan while this
        // iteration's pieces are still being computed.
        constexpr uint32_t kRec = kBW + 3;
        CircularBuffer(cb_sol_plan).wait_front(1);
        for (uint32_t i = 0; i < n_res; ++i) {
            const uint32_t flat = global_q_start + i;
            res_head[i] = (flat / num_q_chunks) % NH;
            const uint32_t r = sol_res_plan_rec_base + i * kRec;
            for (uint32_t w = 0; w < kBW; ++w) {
                res_blk[i][w] = ckernel::read_tile_value(cb_sol_plan, 0, r + w);
            }
            union_lo[i] = ckernel::read_tile_value(cb_sol_plan, 0, r + kBW);
            union_hi[i] = ckernel::read_tile_value(cb_sol_plan, 0, r + kBW + 1);
            do_pooled = ckernel::read_tile_value(cb_sol_plan, 0, r + kBW + 2);
        }
        CircularBuffer(cb_sol_plan).pop_front(1);
    } else {
        for (uint32_t i = 0; i < n_res; ++i) {
            const uint32_t flat = global_q_start + i;
            res_head[i] = (flat / num_q_chunks) % NH;
            CircularBuffer(cb_sol_plan).wait_front(1);
            for (uint32_t w = 0; w < kBW; ++w) {
                res_blk[i][w] = ckernel::read_tile_value(cb_sol_plan, 0, w);
            }
            union_lo[i] = ckernel::read_tile_value(cb_sol_plan, 0, kBW);
            union_hi[i] = ckernel::read_tile_value(cb_sol_plan, 0, kBW + 1);
            do_pooled = ckernel::read_tile_value(cb_sol_plan, 0, kBW + 2);
            CircularBuffer(cb_sol_plan).pop_front(1);
        }
    }
    const auto bit = [](uint32_t lo, uint32_t hi, uint32_t k) -> bool {
        return k < 32 ? ((lo >> k) & 1u) != 0 : ((hi >> (k - 32)) & 1u) != 0;
    };
    // Does resident chunk i keep local block b (of this ring iteration's shard)?
    const auto block_kept = [&](uint32_t i, uint32_t b) -> bool {
        return b < kBW * 32 && ((res_blk[i][b >> 5] >> (b & 31)) & 1u) != 0;
    };
    // Which resident chunks have state in their slot. Carried across ring iterations in acc_state.last_call_k_chunks
    // (unused by this mode otherwise): the slots persist in L1 between calls.
    uint32_t initialized = is_first_active_iter ? 0u : acc_state.last_call_k_chunks;

    if (is_first_active_iter) {
        // Fill the slot CBs once, exactly full, write pointers held at the base (the step's state contract). Their
        // contents are written by each slot's first step and updated in place by every later one.
        CircularBuffer(cb_max_in).reserve_back(kStatsCap);
        cb_push_back_hold_wr_ptr(cb_max_in, kStatsCap);
        CircularBuffer(cb_sum_in).reserve_back(kStatsCap);
        cb_push_back_hold_wr_ptr(cb_sum_in, kStatsCap);
        CircularBuffer(cb_prev_out).reserve_back(kOutCap);
        cb_push_back_hold_wr_ptr(cb_prev_out, kOutCap);
        CircularBuffer(cb_max_in).wait_front(kStatsCap);
        CircularBuffer(cb_sum_in).wait_front(kStatsCap);
        CircularBuffer(cb_prev_out).wait_front(kOutCap);
    }

    uint32_t KV_chunks_processed_in_iter = 0;

    // One piece (or pooled entry) for a group of resident chunks (every chunk of the head that keeps a block,
    // up to the scratch capacity): run_cols[g * kBlocksPerPiece + s] is the K^T column tile where run s of member
    // g starts, run_ws[...] its width in tiles (adjacent kept blocks merged), n_runs[g] the run count.
    const auto run_step = [&](const uint8_t* slots,
                              const uint32_t n_slots,
                              const uint8_t* run_cols,
                              const uint8_t* run_ws,
                              const uint8_t* n_runs,
                              const bool apply_col_bias) {
        // The step's first wait, taken before the zone opens (it returns at once inside): RING_STEP then measures
        // compute only, and a late K/V entry shows as a gap between steps instead of as step time.
        CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles);
        MaybeDeviceZoneScopedN(kRingStepProfiling, "RING_STEP");
        sdpa_resident_piece_step_compact<
            Sq_chunk_t,
            Sk_chunk_t,
            DHt,
            vDHt,
            scale_fp32,
            qkt_subblock_w,
            cb_q_in,
            cb_kt_in,
            cb_v_in,
            cb_qkt_im,
            cb_identity_scale_in,
            cb_col_identity,
            cb_max_scratch,
            cb_exp_max_diff,
            cb_max_in,
            cb_sum_in,
            cb_prev_out,
            cb_out,
            cb_sol_bias,
            v_cb_physical_width_t,
            kQkCap,
            kStatsCap>(slots, n_slots, run_cols, run_ws, n_runs, initialized, apply_col_bias);
        for (uint32_t g = 0; g < n_slots; ++g) {
            initialized |= 1u << slots[g];
        }
    };
    // Members sorted by width and stepped in groups of at most kQkCap scratch tiles (sol_res_run_groups, outlined:
    // reached from the piece loop and the pooled loop).
    const auto run_groups = [&](const uint8_t* members,
                                const uint8_t* widths,
                                const uint8_t* mruns,
                                const uint8_t* mrunws,
                                const uint8_t* mnruns,
                                const uint32_t n,
                                const bool apply_col_bias) {
        sol_res_run_groups<sol_resident_slots, kBlocksPerPiece, Sq_chunk_t, kQkCap>(
            run_step, members, widths, mruns, mrunws, mnruns, n, apply_col_bias);
    };
    // One K/V entry, after its last step (or untouched: streamed for another chain member only).
    const auto pop_kv = [&]() {
        CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles);
        sdpa_cb_pop_front_out_of_line(cb_kt_in, kt_entry_tiles);
        CircularBuffer(cb_v_in).wait_front(v_entry_tiles);
        sdpa_cb_pop_front_out_of_line(cb_v_in, v_entry_tiles);
    };

    // Local pieces of shard ring_id. Heads are interleaved per k in head order, exactly as the reader streams them.
    for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
        if (!kv_chunk_starts_before_logical_end<false, false, local_padded_Nt, chunk_size_t, kv_region_Nt>(
                ring_id, k * Sk_chunk_t, logical_nt)) {
            continue;
        }
        // Tail of this piece (logical_n boundary or padded local shard tail): only full blocks are ever kept, so no
        // kept block reaches into it and the steps need no in-entry masking. Checked below.
        [[maybe_unused]] uint32_t active_Sk_param = Sk_chunk_t;
        bool narrowed = false;
        if constexpr (global_n_mask_enabled) {
            if (ring_iter_needs_global_n_mask && k == global_n_mask_chunk_id) {
                active_Sk_param = Sk_chunk_t - lw_mask.global_n_padded_tiles;
                narrowed = true;
            }
        }
        if constexpr (local_n_mask_enabled) {
            if (!narrowed && local_n_needs_masking && k == local_n_mask_chunk_id) {
                active_Sk_param = Sk_chunk_t - lw_mask.local_n_padded_tiles;
            }
        }
        (void)narrowed;
        // One pass per head whose chain streamed this piece; a head's streamed set is in its chunks' plan words.
        uint32_t heads_done = 0;  // bitmask over resident chunk indices already covered by an earlier head
        for (uint32_t i0 = 0; i0 < n_res; ++i0) {
            if ((heads_done >> i0) & 1u) {
                continue;
            }
            uint32_t head_mask = 0;
            for (uint32_t i = i0; i < n_res; ++i) {
                if (res_head[i] == res_head[i0]) {
                    head_mask |= 1u << i;
                }
            }
            heads_done |= head_mask;
            if (!bit(union_lo[i0], union_hi[i0], k)) {
                continue;  // nobody on this head's chain needs the piece: nothing was pushed
            }
            // The piece's group: every chunk of the head that keeps a block of it, with its run table (adjacent
            // kept blocks merged into one run; the whole group is stepped up to the scratch capacity).
            uint8_t group[sol_resident_slots];
            uint8_t gwidth[sol_resident_slots];
            uint8_t runs[sol_resident_slots * kBlocksPerPiece];
            uint8_t runws[sol_resident_slots * kBlocksPerPiece];
            uint8_t n_runs[sol_resident_slots];
            uint32_t n_group = 0;
            uint32_t group_mask = 0;
            for (uint32_t i = i0; i < n_res; ++i) {
                if (((head_mask >> i) & 1u) == 0) {
                    continue;
                }
                uint32_t cnt = 0;
                uint32_t width = 0;
                bool prev_kept = false;
                for (uint32_t j = 0; j < kBlocksPerPiece; ++j) {
                    const bool kept = block_kept(i, k * kBlocksPerPiece + j);
                    if (kept) {
                        // Kept blocks are full blocks: never in the padded tail of the piece.
                        ASSERT((j + 1) * kBlockTiles <= active_Sk_param);
                        if (prev_kept) {
                            runws[n_group * kBlocksPerPiece + cnt - 1] += kBlockTiles;
                        } else {
                            runs[n_group * kBlocksPerPiece + cnt] = static_cast<uint8_t>(j * kBlockTiles);
                            runws[n_group * kBlocksPerPiece + cnt] = kBlockTiles;
                            ++cnt;
                        }
                        width += kBlockTiles;
                    }
                    prev_kept = kept;
                }
                if (cnt == 0) {
                    continue;
                }
                group[n_group] = static_cast<uint8_t>(i);
                gwidth[n_group] = static_cast<uint8_t>(width);
                n_runs[n_group] = static_cast<uint8_t>(cnt);
                ++n_group;
                group_mask |= 1u << i;
            }
            if (n_group > 0) {
                // Block 0 of piece 0 is forced kept for every chunk, so piece 0 creates every slot of the head
                // before any other piece touches one, and no later step meets an uninitialized slot.
                ASSERT(k != 0 || group_mask == head_mask);
                ASSERT(k == 0 || (initialized & group_mask) == group_mask);
                run_groups(group, gwidth, runs, runws, n_runs, n_group, false);
            }
            pop_kv();
            KV_chunks_processed_in_iter++;
        }
    }

    // Pooled chunks (first active ring iteration): full-width steps over the head's resident chunks in groups of
    // the scratch capacity, each member with its own bias row (the reader pushes them in Q-chunk order after the
    // pooled K/V entry; the step consumes them one at a time).
    if (do_pooled) {
        uint32_t heads_done = 0;
        for (uint32_t i0 = 0; i0 < n_res; ++i0) {
            if ((heads_done >> i0) & 1u) {
                continue;
            }
            uint32_t head_mask = 0;
            for (uint32_t i = i0; i < n_res; ++i) {
                if (res_head[i] == res_head[i0]) {
                    head_mask |= 1u << i;
                }
            }
            heads_done |= head_mask;
            uint8_t pgroup[sol_resident_slots];
            uint8_t pwidth[sol_resident_slots];
            uint8_t pcols[sol_resident_slots * kBlocksPerPiece];
            uint8_t pws[sol_resident_slots * kBlocksPerPiece];
            uint8_t pruns[sol_resident_slots];
            uint32_t n_p = 0;
            for (uint32_t i = i0; i < n_res; ++i) {
                if (((head_mask >> i) & 1u) == 0) {
                    continue;
                }
                pgroup[n_p] = static_cast<uint8_t>(i);
                pcols[n_p * kBlocksPerPiece] = 0;  // one run from column 0
                pruns[n_p] = 1;
                ++n_p;
            }
            for (uint32_t pc = 0; pc < sol_pooled_chunks; ++pc) {
                const uint32_t active_width = (pc + 1 == sol_pooled_chunks) ? sol_pooled_last_tiles : Sk_chunk_t;
                for (uint32_t g = 0; g < n_p; ++g) {
                    pwidth[g] = static_cast<uint8_t>(active_width);
                    pws[g * kBlocksPerPiece] = static_cast<uint8_t>(active_width);
                }
                run_groups(pgroup, pwidth, pcols, pws, pruns, n_p, true);
                pop_kv();
                KV_chunks_processed_in_iter++;
            }
        }
    }
    acc_state.last_call_k_chunks = initialized;

    if (is_last_ring_iter) {
        // Normalize every resident chunk, in slot order, into cb_out for the writer: the sum and out slots are
        // consumed from the front. Then release the unused slots, the max slots and Q, so every slot CB is empty with
        // its pointers back at the base.
        MaybeDeviceZoneScopedN(kRingStepProfiling, "RES_NORM");
        for (uint32_t i = 0; i < n_res; ++i) {
            ASSERT((initialized >> i) & 1u);
            normalize_row_streaming<
                false,
                vDHt,
                norm_dst_size,
                cb_col_identity,
                cb_recip_scratch,
                cb_out,
                scale_fp32,
                false,
                INVALID_CB>(cb_sum_in, cb_prev_out, Sq_chunk_t, cb_max_in);
        }
        if (n_res < sol_resident_slots) {
            sdpa_cb_pop_front_out_of_line(cb_sum_in, (sol_resident_slots - n_res) * Sq_chunk_t);
            sdpa_cb_pop_front_out_of_line(cb_prev_out, (sol_resident_slots - n_res) * out_tiles);
        }
        sdpa_cb_pop_front_out_of_line(cb_max_in, kStatsCap);
        sdpa_cb_pop_front_out_of_line(cb_q_in, n_res * q_chunk_tiles);
    }

    // Phase alignment of the K/V CBs with the chained reader (same rule as sdpa_ring_v2), unless the reader drops
    // the dummies (sol_res_no_dummies).
    if constexpr (!sol_res_no_dummies) {
        MaybeDeviceZoneScopedN(kRingStepProfiling, "RES_DUMMY");
        for (uint32_t dummy_chunk = 0;
             dummy_chunk < dummy_kv_chunks_for_phase_alignment<false, false, kv_cb_pages>(KV_chunks_processed_in_iter);
             ++dummy_chunk) {
            CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles);
            sdpa_cb_pop_front_out_of_line(cb_kt_in, kt_entry_tiles);
            CircularBuffer(cb_v_in).wait_front(v_entry_tiles);
            sdpa_cb_pop_front_out_of_line(cb_v_in, v_entry_tiles);
        }
    }
}
