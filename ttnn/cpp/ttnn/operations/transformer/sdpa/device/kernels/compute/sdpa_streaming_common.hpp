// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streaming SDPA compute helpers shared by compute_streaming.hpp and sparse_ring_streaming.hpp.
// Depends on primitives from compute_common.hpp (must be included first). The includer must also define
// should_use_blocked_pack_width(uint32_t) before including this header: each op sets its own blocked-pack threshold.

#pragma once

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#endif
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

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

ALWI void sdpa_mm_init(
    uint32_t in0_cb, uint32_t in1_cb, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
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

// Sentinel for "no CB" — beyond the valid 0-31 range.
constexpr uint32_t INVALID_CB = 32;

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
// field's limit) for the resident step, whose interleaved exp / P@V stages post one token per S5 batch.
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
