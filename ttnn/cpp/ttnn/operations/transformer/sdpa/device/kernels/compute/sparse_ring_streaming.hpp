// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute helpers of sparse ring attention (sparse_ring_sdpa.cpp): the per-piece step over the resident Q
// chunks and the ring-iteration driver. Depends on primitives from compute_common.hpp (must be included first).
// The generic streaming helpers it shares with compute_streaming.hpp live in sdpa_streaming_common.hpp.

#pragma once

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#include "internal/tt-1xx/cache.h"  // invalidate_l1_cache on every compute thread (plan records read from L1)
#endif
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

// BH benefits from blocked pack at width 4; WH keeps the threshold at 8 because
// width-4 blocked-pack reconfiguration costs more than it saves there.
#ifdef ARCH_BLACKHOLE
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 4;
#else
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 8;
#endif
ALWI bool should_use_blocked_pack_width(uint32_t pack_width) { return pack_width >= MIN_BLOCKED_PACK_TILES; }

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa_streaming_common.hpp"

// Ring joint SDPA: the per-phase zones of the inner step are compiled out unless the factory defines
// SPARSE_STEP_PROFILE (not set by the factory; a profiling build adds it), and they only emit under the device profiler
// (PROFILE_KERNEL).
#if defined(SPARSE_STEP_PROFILE) && defined(PROFILE_KERNEL)
constexpr bool kRingStepProfiling = true;
#else
constexpr bool kRingStepProfiling = false;
#endif

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
                if constexpr (sink_per_row) {
                    // Sparse SDPA packs different heads in each row; both inputs hold a first-column vector.
                    sub_init(cur_max_cb_rt, cb_attention_sink);
                    sub_tiles(cur_max_cb_rt, cb_attention_sink, sink_row_offset + s, sink_row_offset + s, 1);
                } else {
                    sub_bcast_scalar_init(cur_max_cb_rt, cb_attention_sink);
                    sub_tiles_bcast_scalar(cur_max_cb_rt, cb_attention_sink, sink_row_offset + s, 0, 1);
                }
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

// ============ Resident Q: in-place slot state, lazy running max, fused per-piece steps ============

// Helpers of the per-piece step, outlined: each is reached from several sites on every TRISC, and the kernel
// config buffer at 5 s has only a few KB of margin.

// Packs a DEST batch of units into the compact score scratch: unit u = member unit_g[u], score column unit_c0[u],
// width unit_cw[u]; row r of member g is at off[g] + r * W[g]. Blocked (one call per row) when the unit is at least
// MIN_BLOCKED_PACK_TILES wide, per tile otherwise; the pack width is reconfigured only when it changes (pk_w).
template <uint32_t Sq_chunk_t, uint32_t cb>
static __attribute__((noinline, noclone)) void res_pack_units(
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

// Members sorted by width (stable insertion sort of the member rows, in place: the tables are the caller's one
// shared set, so no copy lands on the compute threads' stack), then stepped in groups of at most kQkCap scratch tiles
// (Sq_chunk_t x width per member).
template <uint32_t kRunStride, uint32_t Sq_chunk_t, uint32_t kQkCap, typename StepFn>
static __attribute__((noinline, noclone)) void res_run_groups(
    StepFn& run_step,
    uint8_t* members,
    uint8_t* widths,
    uint8_t* runs,
    uint8_t* runws,
    uint8_t* nruns,
    const uint32_t n,
    const bool apply_col_bias) {
    const auto swap8 = [](uint8_t* t, uint32_t a, uint32_t b) {
        const uint8_t x = t[a];
        t[a] = t[b];
        t[b] = x;
    };
    for (uint32_t i = 1; i < n; ++i) {
        for (uint32_t j = i; j > 0 && widths[j - 1] > widths[j]; --j) {
            swap8(members, j - 1, j);
            swap8(widths, j - 1, j);
            swap8(nruns, j - 1, j);
            for (uint32_t s = 0; s < kRunStride; ++s) {
                swap8(runs, (j - 1) * kRunStride + s, j * kRunStride + s);
                swap8(runws, (j - 1) * kRunStride + s, j * kRunStride + s);
            }
        }
    }
    uint32_t g0 = 0;
    while (g0 < n) {
        uint32_t g1 = g0;
        uint32_t tiles = 0;
        while (g1 < n && tiles + Sq_chunk_t * widths[g1] <= kQkCap) {
            tiles += Sq_chunk_t * widths[g1];
            ++g1;
        }
        ASSERT(g1 > g0);  // one member always fits: Sq_chunk_t x Sk_chunk_t <= kQkCap
        run_step(members + g0, g1 - g0, runs + g0 * kRunStride, runws + g0 * kRunStride, nruns + g0, apply_col_bias);
        g0 = g1;
    }
}

// Per-step state of the compact step's DEST-batch issue, shared with its outlined flushes (res_s1_flush,
// res_s5_flush): one copy of each flush body per TRISC instead of one per call site. The 5 s build sits within
// a few KB of the kernel config buffer (inlined flushes overflowed it).
template <uint32_t kDst, uint32_t kMaxMembers>
struct ResidentStepState {
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
    uint32_t v_wait_tiles = 0;  // the step's V entries (one piece, or a pair with pair_steps)
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
static __attribute__((noinline, noclone)) void res_pv_member(
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
static __attribute__((noinline, noclone)) void res_s1_flush(ResidentStepState<kDst, kMaxMembers>& st) {
    if (st.uu == 0) {
        return;
    }
    tile_regs_commit();
    tile_regs_wait();
    res_pack_units<Sq_chunk_t, cb_qkt_im>(st.unit_g, st.unit_c0, st.unit_cw, st.uu, st.off, st.W, st.pk_w);
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
static __attribute__((noinline, noclone)) void res_s5_flush(
    ResidentStepState<kDst, kMaxMembers>& st, const bool final) {
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
        res_pack_units<Sq_chunk_t, cb_qkt_im>(st.unit_g, st.unit_c0, st.unit_cw, st.uu, st.off, st.W, st.pk_w);
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
            CircularBuffer(cb_v_in).wait_front(st.v_wait_tiles);
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
        res_pv_member<Sq_chunk_t, vDHt, pv_w, kMaxRuns, cb_qkt_im, cb_v_in, cb_prev_out>(
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

// ---- Per-piece step ----
/**
 * One K/V piece (or pooled entry) applied to a group of resident Q chunks ("members"), in place on their slots.
 * How the work is issued (per-tile matmul calls, per-tile address-programmed packs and per-stage round trips cost
 * ~1 us per score tile against the dense kernel's 0.28 us, so the step batches all three):
 *  1. Compact score layout. Member g's scores occupy Sq_chunk_t rows of exactly W_g tiles (its kept width) at
 *     offset off[g] = sum of the earlier members' Sq_chunk_t x W; the row stride is W_g, not the piece width. A
 *     step can then hold every chunk of the head that keeps a block of the piece (the capacity is the scratch's
 *     tile count, qk_cap), so there is one step per piece instead of one per sub-group of four, and each stage
 *     runs several DEST batches back to back (half-sync DEST overlaps math and pack across them).
 *  2. Runs of adjacent kept blocks are one run of 2, 4, ... tiles (run_ws[g * kMaxRuns + s]); a unit of up to
 *     qk_cw_cap columns is one matmul call and, when it is >= MIN_BLOCKED_PACK_TILES wide, one pack call per row
 *     instead of one per tile. The same for the exp stage.
 *  3. P@V uses the widest sub-block DEST allows (pv_w = min(vDHt, kDst / Sq_chunk_t)): fewer matmul calls per
 *     member and blocked output packs.
 * Members with equal widths should be adjacent in the group (the caller sorts by width): the row-max reduce, the
 * subtract and the two P@V setups are configured per width and re-issued only when it changes.
 *  4. The exp stage and the P@V are interleaved per member (see S5 + S6 below): the P@V of the members
 *     finished in one S5 batch runs under the exp and packs of the next, with one PACK_DONE token per S5 batch.
 * The running max is lazy (S2-S4: the state is rescaled only when some row's max grows past a threshold), the row
 * sum is P x e0, and the pooled entries add their column bias before the max. run_cols / run_ws / n_runs are per
 * member; a pooled entry is one run of its active width from column 0 with apply_col_bias.
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
    uint32_t cb_pooled_bias,
    uint32_t v_cb_physical_width_t,
    uint32_t qk_cap,
    uint32_t stats_cap,
    uint32_t kStepPieces>
static __attribute__((noinline, noclone)) void sdpa_resident_piece_step_compact(
    const uint8_t* slots,
    const uint32_t n_slots,
    const uint8_t* run_cols,
    const uint8_t* run_ws,
    const uint8_t* n_runs,
    const uint32_t initialized_mask,
    const bool apply_col_bias,
    const uint32_t n_entries) {
    constexpr uint32_t kDst = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t kBlockTiles = get_named_compile_time_arg_val("block_tiles");
    // A step covers n_entries (1, or 2 with kStepPieces == 2: pair steps) consecutive K/V entries of the CB. Run
    // columns are step columns: column c is tile c % Sk_chunk_t of entry c / Sk_chunk_t (runs never cross an entry),
    // so K^T tiles are addressed per entry and V rows, whose entries are contiguous rows of vDHt tiles, linearly.
    constexpr uint32_t kMaxRuns = (Sk_chunk_t / kBlockTiles) * kStepPieces;
    constexpr uint32_t kMaxMembers = 32;
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t kt_entry_tiles = DHt * Sk_chunk_t;
    constexpr uint32_t v_entry_tiles = Sk_chunk_t * v_cb_physical_width_t;
    static_assert(v_cb_physical_width_t == vDHt, "resident mode addresses V rows at a vDHt tile stride");
    static_assert(Sk_chunk_t % kBlockTiles == 0, "a piece holds whole blocks");
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
        ASSERT(n_runs[g] >= 1 && w >= 1 && w <= Sk_chunk_t * n_entries);
        W[g] = static_cast<uint8_t>(w);
        off[g] = static_cast<uint16_t>(total);
        total += Sq_chunk_t * w;
        const uint32_t t = (static_cast<uint32_t>(slots[g]) + 1) * q_chunk_tiles;
        q_wait_tiles = t > q_wait_tiles ? t : q_wait_tiles;
    }
    ASSERT(total <= qk_cap);
    const auto sidx = [&](uint32_t g, uint32_t r, uint32_t c) -> uint32_t { return off[g] + r * W[g] + c; };

    // ---- S1: Q@K^T into the compact scratch. Units: (member, run, column batch of cw <= qk_cw_cap). ----
    ASSERT(n_entries >= 1 && n_entries <= kStepPieces && (!apply_col_bias || n_entries == 1));
    CircularBuffer(cb_qkt_im).reserve_back(qk_cap);
    CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles * n_entries);
    CircularBuffer(cb_q_in).wait_front(q_wait_tiles);
    sdpa_maybe_pack_reconfig_data_format<cb_out, cb_qkt_im>();
    sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in>();
    ResidentStepState<kDst, kMaxMembers> st{};
    st.v_wait_tiles = v_entry_tiles * n_entries;
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
                        res_s1_flush<Sq_chunk_t, cb_qkt_im, kDst, kMaxMembers>(st);
                    }
                    if (st.uu == 0) {
                        tile_regs_acquire();
                    }
                    if (cw != mm_cw) {
                        sdpa_mm_init(cb_q_in, cb_kt_in, true, cw, Sq_chunk_t, DHt);
                        mm_cw = cw;
                    }
                    uint32_t in0 = slots[g] * q_chunk_tiles;
                    const uint32_t c1 = run_col(g, s) + q0;
                    uint32_t in1 = (c1 / Sk_chunk_t) * kt_entry_tiles + c1 % Sk_chunk_t;
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
        res_s1_flush<Sq_chunk_t, cb_qkt_im, kDst, kMaxMembers>(st);
    }
    sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();
    if (apply_col_bias) {
        // Pooled entry: every member one contiguous run from column 0 with its own bias tile row (the same row for
        // each of its Q tile rows), consumed one at a time (the reader pushes them in member order; the CB holds
        // one row).
        for (uint32_t g = 0; g < n_slots; ++g) {
            ASSERT(n_runs[g] == 1 && run_col(g, 0) == 0);
            CircularBuffer(cb_pooled_bias).wait_front(Sk_chunk_t);
            begin_mask_l1_accumulate<false>(cb_qkt_im, cb_pooled_bias);
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                l1_acc_tile_run(cb_pooled_bias, 0, cb_qkt_im, sidx(g, r, 0), W[g]);
            }
            end_mask_l1_accumulate();
            CircularBuffer(cb_pooled_bias).pop_front(Sk_chunk_t);
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

    // ---- S5 + S6, interleaved per member. S5: in-place exp of the scores, per member over its W_g columns
    //      in units of <= max_cols_per_acq (the broadcast subtract on MATH, the exp and the packs on PACK). S6: P@V
    //      per member (res_pv_member). The two stages load complementary threads (S5 is pack-thread bound, S6
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
        constexpr auto s5_flush = res_s5_flush<
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
 * One ring iteration of the resident step loop.
 *
 * Every K/V entry the reader pushes is consumed by the steps that need it and popped once, after the last of them.
 * The per-chunk flash state (running sum, running max, output accumulator) lives in the restore CBs {cb_sum_in,
 * cb_max_in, cb_prev_out}, one SLOT per resident chunk in Q-chunk order, updated in place (see
 * sdpa_resident_piece_step_compact for the contract). The slot CBs are filled once per pass, on its first active ring
 * iteration (exactly full, write pointer held at the base), and are neither pushed nor popped until its last ring
 * iteration, where normalize_row_streaming consumes the n_res slots from the front in order and the unused slots are
 * popped, so every slot CB is empty with its pointers at the base for the next pass.
 *
 * Steps are per piece: per streamed piece, one step per group of the head's resident chunks that keep a block of it,
 * grouped until their compact scores fill kQkCap tiles; each chunk has its own run table of kept blocks (its Q
 * rows against those blocks' K^T columns / V rows, addressed in place in the staged entry). A piece streamed only for
 * another chain member is drained. The reader forces block 0 of every shard kept for every chunk, so the first step of
 * a head on piece 0 creates every slot's state before any other piece. Dropped blocks are pooled by the column bias of
 * the pooled chunks that follow the local pieces on the pass's first active ring iteration: one full-width step per
 * resident chunk of the head, each with its own bias row (pushed by the reader in Q-chunk order after the pooled K/V
 * entry).
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
    uint32_t qkt_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_max_scratch,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
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
    uint32_t local_n_padded_tiles,
    uint32_t cb_sparse_plan,
    uint32_t cb_pooled_bias,
    uint32_t pooled_chunks,
    uint32_t pooled_last_tiles,
    bool pooled_route,
    uint32_t resident_slots,
    uint32_t plan_block_words,
    uint32_t plan_rec_base,
    uint32_t num_q_passes,
    bool pair_steps,
    uint32_t kv_cb_pages>
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
    const uint32_t global_n_padded_tiles,
    uint32_t& initialized_slots,
    uint32_t& kv_pos,
    const bool is_last_ring_iter,
    const bool is_first_active_iter) {
    // PACK_DONE capacity 15: the compact step's interleaved S5/S6 posts one token per S5 batch (see the step).
    init_sdpa_streaming_semaphores<15>();
    static_assert(resident_slots >= 1 && resident_slots <= 32, "resident state is tracked in a 32-bit mask");
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

    // Blocks of kBlockTiles tiles (block_tiles); a piece holds kBlocksPerPiece of them. The plan page carries this
    // shard's kept-block bits per resident chunk (block b: word b/32, bit b%32; only full blocks, block 0 forced).
    constexpr uint32_t kBlockTiles = get_named_compile_time_arg_val("block_tiles");
    constexpr uint32_t kBlocksPerPiece = Sk_chunk_t / kBlockTiles;
    constexpr uint32_t kBW = plan_block_words;
    static_assert(Sk_chunk_t % kBlockTiles == 0, "a piece holds whole blocks");
    static_assert(kBW >= 1 && kBW + 3 <= 16, "the plan header holds the block words, the union bits and the flag");
    static_assert(kBlocksPerPiece <= 255 && Sk_chunk_t <= 255, "run tables are byte-indexed");

    // CB capacities, as the program factory allocates them for resident Q: the score
    // scratch holds max(qk_tiles, slots x Sq_chunk_t x kBlockTiles); the max/alpha scratches and the max/sum slot
    // CBs hold Sq_chunk_t per slot; the out slot CB holds Sq_chunk_t x vDHt per slot. The step pushes and pops the
    // scratches at full capacity and the slot CBs are filled at full capacity, so the counts must match exactly.
    // The score scratch holds kGroupCap full-width chunks (the factory's kResidentGroup); a step packs as many
    // compact members as fit in kQkCap.
    constexpr uint32_t kGroupCap = resident_slots < 4 ? resident_slots : 4;
    constexpr uint32_t kQkTiles = Sq_chunk_t * Sk_chunk_t;
    // Pair steps: a step covers two consecutive pieces of a head, so it holds two pieces' widths per member (the
    // factory doubles the scratch the same way) and the run tables have room for both pieces' runs.
    constexpr uint32_t kStepPieces = pair_steps ? 2 : 1;
    constexpr uint32_t kRunStride = kBlocksPerPiece * kStepPieces;  // run-table entries per member
    static_assert(kRunStride <= 255, "run tables are byte-indexed");
    static_assert(!pair_steps || kv_cb_pages >= 2, "a pair needs two K/V entries in the CB");
    constexpr uint32_t kQkCap = kQkTiles > kGroupCap * Sq_chunk_t * Sk_chunk_t * kStepPieces
                                    ? kQkTiles
                                    : kGroupCap * Sq_chunk_t * Sk_chunk_t * kStepPieces;
    constexpr uint32_t kStatsCap = resident_slots * Sq_chunk_t;
    constexpr uint32_t kOutCap = resident_slots * out_tiles;
#ifdef TRISC_PACK
    // ASSERT is a statement, not an expression: keep it out of the PACK(...) expression macro.
    ASSERT(get_local_cb_interface(cb_qkt_im).fifo_num_pages == kQkCap);
    // The pair-step wrap check (kv_pos) assumes the K/V CBs hold exactly kv_cb_pages entries.
    ASSERT(get_local_cb_interface(cb_kt_in).fifo_num_pages == kv_cb_pages * kt_entry_tiles);
    ASSERT(get_local_cb_interface(cb_v_in).fifo_num_pages == kv_cb_pages * v_entry_tiles);
    ASSERT(get_local_cb_interface(cb_max_scratch).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_exp_max_diff).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_max_in).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_sum_in).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_prev_out).fifo_num_pages == kOutCap);
#endif

    // The pass's chunks (global_q_start / global_q_end bound the current pass).
    const uint32_t n_res = global_q_end - global_q_start;
    ASSERT(n_res <= resident_slots);

    // Chunk i's plan record: [kBW kept-block words][union lo, hi][do_pooled]. With one plan page per ring iteration
    // (plan_rec_base != 0) the records are read in place, from L1, for the whole iteration: the page is popped at the
    // end of this call, and the plan CB holds two pages so the reader fills the next iteration's meanwhile. Copying
    // them to the stack (kBW + 3 words per slot) overflowed the unpack and pack threads' stack, which shares 4 KB with
    // their globals, at 14 slots. With a page per chunk (plan_rec_base == 0: a one-shard ring) each record is copied
    // and its page popped at once.
    constexpr uint32_t kRec = kBW + 3;
    constexpr bool kPlanInL1 = plan_rec_base != 0;
    [[maybe_unused]] uint32_t plan_copy[kPlanInL1 ? 1 : resident_slots * kRec];
    [[maybe_unused]] const volatile uint32_t* plan_l1 = nullptr;
    if constexpr (kPlanInL1) {
        CircularBuffer(cb_sparse_plan).wait_front(1);
        // UNPACK waited for the page and hands its address to MATH and PACK; every thread then reads it directly. The
        // fence drops L1 cache lines a thread kept from the page's previous use (Blackhole's RISC L1 cache is not
        // coherent with NoC writes).
        plan_l1 =
            reinterpret_cast<const volatile uint32_t*>(ckernel::get_tile_address(cb_sparse_plan, 0)) + plan_rec_base;
        invalidate_l1_cache();
    } else {
        for (uint32_t i = 0; i < n_res; ++i) {
            CircularBuffer(cb_sparse_plan).wait_front(1);
            for (uint32_t w = 0; w < kRec; ++w) {
                plan_copy[i * kRec + w] = ckernel::read_tile_value(cb_sparse_plan, 0, w);
            }
            CircularBuffer(cb_sparse_plan).pop_front(1);
        }
    }
    const auto plan_word = [&](uint32_t i, uint32_t w) -> uint32_t {
        if constexpr (kPlanInL1) {
            return plan_l1[i * kRec + w];
        } else {
            return plan_copy[i * kRec + w];
        }
    };
    const auto res_head = [&](uint32_t i) -> uint32_t { return ((global_q_start + i) / num_q_chunks) % NH; };
    const auto union_lo = [&](uint32_t i) -> uint32_t { return plan_word(i, kBW); };
    const auto union_hi = [&](uint32_t i) -> uint32_t { return plan_word(i, kBW + 1); };
    // The same in every record of the iteration.
    [[maybe_unused]] const uint32_t do_pooled = n_res > 0 ? plan_word(0, kBW + 2) : 0u;
    const auto bit = [](uint32_t lo, uint32_t hi, uint32_t k) -> bool {
        return k < 32 ? ((lo >> k) & 1u) != 0 : ((hi >> (k - 32)) & 1u) != 0;
    };
    // Does resident chunk i keep local block b (of this ring iteration's shard)?
    const auto block_kept = [&](uint32_t i, uint32_t b) -> bool {
        return b < kBW * 32 && ((plan_word(i, b >> 5) >> (b & 31)) & 1u) != 0;
    };
    // Which resident chunks have state in their slot. Carried across ring iterations in initialized_slots: the slots
    // persist in L1 between calls.
    uint32_t initialized = is_first_active_iter ? 0u : initialized_slots;

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

    // One piece, a pair of pieces, or a pooled entry for a group of resident chunks (every chunk of the head that
    // keeps a block, up to the scratch capacity): run_cols[g * kRunStride + s] is the step column where run s of
    // member g starts, run_ws[...] its width in tiles (adjacent kept blocks merged), n_runs[g] the run count.
    uint32_t step_entries = 1;  // K/V entries of the steps run_groups issues next (2 for a pair)
    const auto run_step = [&](const uint8_t* slots,
                              const uint32_t n_slots,
                              const uint8_t* run_cols,
                              const uint8_t* run_ws,
                              const uint8_t* n_runs,
                              const bool apply_col_bias) {
        // The step's first wait, taken before the zone opens (it returns at once inside): RING_STEP then measures
        // compute only, and a late K/V entry shows as a gap between steps instead of as step time.
        CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles * step_entries);
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
            cb_pooled_bias,
            v_cb_physical_width_t,
            kQkCap,
            kStatsCap,
            kStepPieces>(slots, n_slots, run_cols, run_ws, n_runs, initialized, apply_col_bias, step_entries);
        for (uint32_t g = 0; g < n_slots; ++g) {
            initialized |= 1u << slots[g];
        }
    };
    // Member tables of the next step group: resident chunk index, kept width, run columns / widths (kRunStride per
    // member) and run count. One set, shared by the piece loop and the pooled loop and sorted in place by
    // res_run_groups: the stack holds no per-step or per-group copies (see the plan records above).
    uint8_t m_chunk[resident_slots];
    uint8_t m_width[resident_slots];
    uint8_t m_cols[resident_slots * kRunStride];
    uint8_t m_ws[resident_slots * kRunStride];
    uint8_t m_nruns[resident_slots];
    // The first n members sorted by width (in place) and stepped in groups of at most kQkCap scratch tiles
    // (res_run_groups, outlined: reached from the piece loop and the pooled loop).
    const auto run_groups = [&](const uint32_t n, const bool apply_col_bias) {
        res_run_groups<kRunStride, Sq_chunk_t, kQkCap>(
            run_step, m_chunk, m_width, m_cols, m_ws, m_nruns, n, apply_col_bias);
    };
    // One K/V entry, after its last step (or untouched: streamed for another chain member only). kv_pos is the K/V
    // CBs' read position in entries, carried across ring iterations and passes: a pair must not wrap the CB end,
    // since the step addresses its second entry right after the first.
    const auto pop_kv = [&]() {
        CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles);
        sdpa_cb_pop_front_out_of_line(cb_kt_in, kt_entry_tiles);
        CircularBuffer(cb_v_in).wait_front(v_entry_tiles);
        sdpa_cb_pop_front_out_of_line(cb_v_in, v_entry_tiles);
        kv_pos = kv_pos + 1 == kv_cb_pages ? 0 : kv_pos + 1;
    };

    // Tail of piece k (logical_n boundary or padded local shard tail): only full blocks are ever kept, so no kept
    // block reaches into it and the steps need no in-entry masking (checked in step_pieces).
    const auto active_sk = [&](uint32_t k) -> uint32_t {
        if constexpr (global_n_mask_enabled) {
            if (ring_iter_needs_global_n_mask && k == global_n_mask_chunk_id) {
                return Sk_chunk_t - global_n_padded_tiles;
            }
        }
        if constexpr (local_n_mask_enabled) {
            if (local_n_needs_masking && k == local_n_mask_chunk_id) {
                return Sk_chunk_t - local_n_padded_tiles;
            }
        }
        return Sk_chunk_t;
    };
    // One step group over np (1, or 2 for a pair) consecutive K/V entries holding pieces ks[0..np) of one head
    // (head_mask: its resident chunks): every chunk of the head that keeps a block of them, with its run table over
    // the step's columns (piece p's tile t is column p * Sk_chunk_t + t; adjacent kept blocks of one piece merged
    // into one run, never across pieces), stepped up to the scratch capacity; then the entries popped.
    const auto step_pieces = [&](uint32_t head_mask, const uint8_t* ks, uint32_t np) {
        uint32_t n_group = 0;
        uint32_t group_mask = 0;
        for (uint32_t i = 0; i < n_res; ++i) {
            if (((head_mask >> i) & 1u) == 0) {
                continue;
            }
            uint32_t cnt = 0;
            uint32_t width = 0;
            for (uint32_t p = 0; p < np; ++p) {
                [[maybe_unused]] const uint32_t sk = active_sk(ks[p]);
                bool prev_kept = false;
                for (uint32_t j = 0; j < kBlocksPerPiece; ++j) {
                    const bool kept = block_kept(i, ks[p] * kBlocksPerPiece + j);
                    if (kept) {
                        // Kept blocks are full blocks: never in the padded tail of the piece.
                        ASSERT((j + 1) * kBlockTiles <= sk);
                        if (prev_kept) {
                            m_ws[n_group * kRunStride + cnt - 1] += kBlockTiles;
                        } else {
                            m_cols[n_group * kRunStride + cnt] = static_cast<uint8_t>(p * Sk_chunk_t + j * kBlockTiles);
                            m_ws[n_group * kRunStride + cnt] = kBlockTiles;
                            ++cnt;
                        }
                        width += kBlockTiles;
                    }
                    prev_kept = kept;
                }
            }
            if (cnt == 0) {
                continue;
            }
            m_chunk[n_group] = static_cast<uint8_t>(i);
            m_width[n_group] = static_cast<uint8_t>(width);
            m_nruns[n_group] = static_cast<uint8_t>(cnt);
            ++n_group;
            group_mask |= 1u << i;
        }
        if (n_group > 0) {
            // Block 0 of piece 0 is forced kept for every chunk, so piece 0 creates every slot of the head before any
            // other piece touches one, and no later step meets an uninitialized slot.
            ASSERT(ks[0] != 0 || group_mask == head_mask);
            ASSERT(ks[0] == 0 || (initialized & group_mask) == group_mask);
            step_entries = np;
            run_groups(n_group, false);
            step_entries = 1;
        }
        for (uint32_t p = 0; p < np; ++p) {
            pop_kv();
        }
    };
    const auto piece_live = [&](uint32_t k) -> bool {
        return kv_chunk_starts_before_logical_end<false, false, local_padded_Nt, chunk_size_t, kv_region_Nt>(
            ring_id, k * Sk_chunk_t, logical_nt);
    };

    // Local pieces of shard ring_id. Heads are interleaved per k in head order, exactly as the reader streams them.
    bool one_head = true;
    for (uint32_t i = 1; i < n_res; ++i) {
        one_head = one_head && res_head(i) == res_head(0);
    }
    if (pair_steps && one_head && n_res > 0) {
        // One head, so the reader's entries are this head's streamed pieces in k order. Step them two at a time when
        // neither is a masked tail piece and the pair does not wrap the K/V CB end; otherwise one at a time.
        constexpr uint32_t kMaxPieces = 64;  // the union words' range
        uint8_t ks[kMaxPieces];              // bytes: the compute threads' stack shares 4 KB with their globals
        uint32_t n_ks = 0;
        for (uint32_t k = 0; k < num_local_k_chunks && n_ks < kMaxPieces; ++k) {
            if (piece_live(k) && bit(union_lo(0), union_hi(0), k)) {
                ks[n_ks++] = static_cast<uint8_t>(k);
            }
        }
        const uint32_t all = n_res >= 32 ? 0xFFFFFFFFu : (1u << n_res) - 1u;
        for (uint32_t idx = 0; idx < n_ks;) {
            const bool pair = idx + 1 < n_ks && active_sk(ks[idx]) == Sk_chunk_t &&
                              active_sk(ks[idx + 1]) == Sk_chunk_t && kv_pos + 2 <= kv_cb_pages;
            const uint32_t np = pair ? 2u : 1u;
            step_pieces(all, ks + idx, np);
            idx += np;
        }
    } else {
        for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
            if (!piece_live(k)) {
                continue;
            }
            // One pass per head whose chain streamed this piece; a head's streamed set is in its chunks' plan words.
            uint32_t heads_done = 0;  // bitmask over resident chunk indices already covered by an earlier head
            for (uint32_t i0 = 0; i0 < n_res; ++i0) {
                if ((heads_done >> i0) & 1u) {
                    continue;
                }
                uint32_t head_mask = 0;
                for (uint32_t i = i0; i < n_res; ++i) {
                    if (res_head(i) == res_head(i0)) {
                        head_mask |= 1u << i;
                    }
                }
                heads_done |= head_mask;
                if (!bit(union_lo(i0), union_hi(i0), k)) {
                    continue;  // nobody on this head's chain needs the piece: nothing was pushed
                }
                const uint8_t k8 = static_cast<uint8_t>(k);
                step_pieces(head_mask, &k8, 1);
            }
        }
    }

    // Pooled chunks (pooled route only, on the pass's first active ring iteration): full-width steps over the head's
    // resident chunks in groups of the scratch capacity, each member with its own bias row (the reader pushes them in
    // Q-chunk order after the pooled K/V entry; the step consumes them one at a time). Without the pooled route the
    // plan's do_pooled is always 0 and no step ever adds a column bias.
    if constexpr (pooled_route) {
        if (do_pooled) {
            uint32_t heads_done = 0;
            for (uint32_t i0 = 0; i0 < n_res; ++i0) {
                if ((heads_done >> i0) & 1u) {
                    continue;
                }
                uint32_t head_mask = 0;
                for (uint32_t i = i0; i < n_res; ++i) {
                    if (res_head(i) == res_head(i0)) {
                        head_mask |= 1u << i;
                    }
                }
                heads_done |= head_mask;
                uint32_t n_p = 0;
                for (uint32_t i = i0; i < n_res; ++i) {
                    if (((head_mask >> i) & 1u) == 0) {
                        continue;
                    }
                    m_chunk[n_p] = static_cast<uint8_t>(i);
                    m_cols[n_p * kRunStride] = 0;  // one run from column 0
                    m_nruns[n_p] = 1;
                    ++n_p;
                }
                for (uint32_t pc = 0; pc < pooled_chunks; ++pc) {
                    const uint32_t active_width = (pc + 1 == pooled_chunks) ? pooled_last_tiles : Sk_chunk_t;
                    for (uint32_t g = 0; g < n_p; ++g) {
                        m_width[g] = static_cast<uint8_t>(active_width);
                        m_ws[g * kRunStride] = static_cast<uint8_t>(active_width);
                    }
                    // Equal widths: the in-place sort keeps the members in Q-chunk order, the reader's bias-row order.
                    run_groups(n_p, true);
                    pop_kv();
                }
            }
        }
    }
    initialized_slots = initialized;

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
        if (n_res < resident_slots) {
            sdpa_cb_pop_front_out_of_line(cb_sum_in, (resident_slots - n_res) * Sq_chunk_t);
            sdpa_cb_pop_front_out_of_line(cb_prev_out, (resident_slots - n_res) * out_tiles);
        }
        sdpa_cb_pop_front_out_of_line(cb_max_in, kStatsCap);
        if constexpr (num_q_passes > 1) {
            // The reader fills cb_q_in to capacity every pass: pop it whole, so the next pass's Q starts at the base.
            CircularBuffer(cb_q_in).wait_front(resident_slots * q_chunk_tiles);
            sdpa_cb_pop_front_out_of_line(cb_q_in, resident_slots * q_chunk_tiles);
        } else {
            sdpa_cb_pop_front_out_of_line(cb_q_in, n_res * q_chunk_tiles);
        }
    }
    if constexpr (kPlanInL1) {
        // The plan records were read in place for the whole iteration.
        CircularBuffer(cb_sparse_plan).pop_front(1);
    }
}
