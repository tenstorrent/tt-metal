// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#include "internal/tt-1xx/cache.h"
#endif
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

#ifdef ARCH_BLACKHOLE
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 4;
#else
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 8;
#endif
ALWI bool should_use_blocked_pack_width(uint32_t pack_width) { return pack_width >= MIN_BLOCKED_PACK_TILES; }

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa_streaming_common.hpp"

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
        ASSERT(g1 > g0);
        run_step(members + g0, g1 - g0, runs + g0 * kRunStride, runws + g0 * kRunStride, nruns + g0, apply_col_bias);
        g0 = g1;
    }
}

template <uint32_t kDst, uint32_t kMaxMembers>
struct ResidentStepState {
    const uint8_t* slots = nullptr;
    const uint8_t* run_cols = nullptr;
    const uint8_t* run_ws = nullptr;
    const uint8_t* n_runs = nullptr;
    const uint8_t* W = nullptr;
    const uint16_t* off = nullptr;
    uint32_t n_slots = 0;
    uint32_t initialized_mask = 0;
    uint8_t unit_g[kDst];
    uint8_t unit_c0[kDst];
    uint8_t unit_cw[kDst];
    uint32_t uu = 0;
    uint32_t uu_tiles = 0;
    uint32_t pk_w = 0;
    uint8_t cbatch[kMaxMembers];
    uint32_t n_done = 0;
    uint32_t batch_idx = 0;
    bool batch_completes = false;
    uint32_t init_cw = 0;
    uint32_t pv_stride = 0;
    bool mm_fmt = false;
    bool v_waited = false;
    uint32_t v_wait_tiles = 0;
    uint32_t s6_next = 0;
};

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
    // The output slots are Float32 (the packer converts from DEST and accumulates in Float32), the S5 packs
    // intermediate-format: one packer switch per flush, around its P@V members. The pack MOP follows DEST's format,
    // so pk_w stays valid across it.
    bool pv_pack_fmt = false;
    while (st.s6_next < st.n_done && (final || static_cast<uint32_t>(st.cbatch[st.s6_next]) + 1u < st.batch_idx)) {
        const uint32_t g = st.s6_next;
        if (!pv_pack_fmt) {
            sdpa_maybe_pack_reconfig_data_format<cb_qkt_im, cb_prev_out>();
            pv_pack_fmt = true;
        }
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
    if (pv_pack_fmt) {
        sdpa_maybe_pack_reconfig_data_format<cb_prev_out, cb_qkt_im>();
    }
}

// Normalization of one slot, consumed from the front of the Float32 output and sum slots into cb_out. Per Q tile row:
// the sum tile goes through DEST into cb_recip_scratch (the matmul unpack does not read Float32 into the 16-bit
// DEST), col_identity collapses its partial sums and the reciprocal replaces it, then the output row is scaled by it.
// Unpacker in the intermediate format and packer in cb_recip_scratch's on entry and exit.
template <
    uint32_t Sq_chunk_t,
    uint32_t vDHt,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_out,
    uint32_t cb_sum_in,
    uint32_t cb_prev_out>
static __attribute__((noinline, noclone)) void res_normalize_slot() {
    constexpr uint32_t kDst = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t batch = vDHt < kDst ? vDHt : kDst;
    CircularBuffer(cb_col_identity).wait_front(1);
    sdpa_maybe_reconfig_data_format<cb_recip_scratch, cb_sum_in, cb_recip_scratch, cb_recip_scratch>();
    for (uint32_t s = 0; s < Sq_chunk_t; ++s) {
        copy_init(cb_sum_in);
        CircularBuffer(cb_sum_in).wait_front(1);
        CircularBuffer(cb_recip_scratch).reserve_back(1);
        tile_regs_acquire();
        copy_tile(cb_sum_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (s > 0) {
            sdpa_maybe_pack_reconfig_data_format<cb_out, cb_recip_scratch>();
        }
        configure_single_tile_pack(cb_recip_scratch);
        pack_tile(0, cb_recip_scratch);
        tile_regs_release();
        CircularBuffer(cb_recip_scratch).push_back(1);
        CircularBuffer(cb_sum_in).pop_front(1);

        sdpa_maybe_reconfig_data_format<cb_sum_in, cb_col_identity, cb_recip_scratch, cb_recip_scratch>();
        matmul_block_init(cb_recip_scratch, cb_col_identity, 0, 1, 1, 1);
        CircularBuffer(cb_recip_scratch).wait_front(1);
        tile_regs_acquire();
        matmul_block(cb_recip_scratch, cb_col_identity, 0, 0, 0, 0, 1, 1, 1);
        recip_tile_init();
#ifdef ARCH_BLACKHOLE
        MATH((recip_tile(0, VectorMode::C)));
#else
        MATH((recip_tile_first_column_wh_idst0_direct()));
#endif
        tile_regs_commit();
        CircularBuffer(cb_recip_scratch).pop_front(1);
        CircularBuffer(cb_recip_scratch).reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_recip_scratch);
        tile_regs_release();
        CircularBuffer(cb_recip_scratch).push_back(1);

        sdpa_maybe_reconfig_data_format<cb_col_identity, cb_prev_out, cb_recip_scratch, cb_recip_scratch>();
        mul_bcast_cols_init(cb_prev_out, cb_recip_scratch);
        sdpa_maybe_pack_reconfig_data_format<cb_recip_scratch, cb_out>();
        CircularBuffer(cb_prev_out).wait_front(vDHt);
        CircularBuffer(cb_recip_scratch).wait_front(1);
        CircularBuffer(cb_out).reserve_back(vDHt);
        for (uint32_t base = 0; base < vDHt; base += batch) {
            const uint32_t n = vDHt - base < batch ? vDHt - base : batch;
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                mul_tiles_bcast_cols(cb_prev_out, cb_recip_scratch, base + j, 0, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
        }
        CircularBuffer(cb_out).push_back(vDHt);
        CircularBuffer(cb_recip_scratch).pop_front(1);
        CircularBuffer(cb_prev_out).pop_front(vDHt);
    }
    sdpa_maybe_reconfig_data_format<cb_prev_out, cb_recip_scratch, cb_recip_scratch, cb_recip_scratch>();
    sdpa_maybe_pack_reconfig_data_format<cb_out, cb_recip_scratch>();
}

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
    uint32_t cb_sum_identity,
    uint32_t cb_max_scratch,
    uint32_t cb_exp_max_diff,
    uint32_t cb_max_in,
    uint32_t cb_sum_in,
    uint32_t cb_prev_out,
    uint32_t cb_out,
    uint32_t cb_pooled_bias,
    uint32_t cb_key_bias,
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
    const uint32_t n_entries,
    const uint8_t* bias_cols,
    const uint32_t n_bias) {
    constexpr uint32_t kDst = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t kBlockTiles = get_named_compile_time_arg_val("block_tiles");
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
        for (uint32_t g = 0; g < n_slots; ++g) {
            ASSERT(n_runs[g] == 1 && run_col(g, 0) == 0);
            CircularBuffer(cb_pooled_bias).wait_front(Sk_chunk_t);
#if defined(SPARSE_FAST_PBIAS) && SPARSE_FAST_PBIAS == 2
            configure_single_tile_pack(cb_qkt_im);
            reconfig_data_format_srcb(cb_qkt_im, cb_pooled_bias);
            unary_bcast_init<BroadcastType::ROW>(cb_pooled_bias);
            PACK((llk_pack_reconfig_l1_acc(1)));
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                const uint32_t out_base = sidx(g, r, 0);
                constexpr uint32_t kBatch = compute_kernel_lib::DEST_AUTO_LIMIT;
                for (uint32_t i = 0; i < W[g]; i += kBatch) {
                    const uint32_t batch = (W[g] - i) < kBatch ? (W[g] - i) : kBatch;
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < batch; ++j) {
                        unary_bcast<BroadcastType::ROW>(cb_pooled_bias, i + j, j);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < batch; ++j) {
                        pack_tile<true>(j, cb_qkt_im, out_base + i + j);
                    }
                    tile_regs_release();
                }
            }
            end_mask_l1_accumulate();
            unary_bcast_uninit<BroadcastType::ROW>(cb_pooled_bias);
            reconfig_data_format_srcb(cb_pooled_bias, cb_qkt_im);
#else
            begin_mask_l1_accumulate<false>(cb_qkt_im, cb_pooled_bias);
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                l1_acc_tile_run(cb_pooled_bias, 0, cb_qkt_im, sidx(g, r, 0), W[g]);
            }
            end_mask_l1_accumulate();
#endif
            CircularBuffer(cb_pooled_bias).pop_front(Sk_chunk_t);
        }
    }
    if (n_bias > 0) {
        begin_mask_l1_accumulate<false>(cb_qkt_im, cb_key_bias);
        for (uint32_t e = 0; e < n_bias; ++e) {
            const uint32_t bc = bias_cols[e];
            for (uint32_t g = 0; g < n_slots; ++g) {
                uint32_t col = 0;
                for (uint32_t s = 0; s < n_runs[g]; ++s) {
                    const uint32_t rc = run_col(g, s);
                    const uint32_t rw = run_wd(g, s);
                    if (bc >= rc && bc < rc + rw) {
                        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                            l1_acc_tile_run(cb_key_bias, e, cb_qkt_im, sidx(g, r, col + bc - rc), 1);
                        }
                        break;
                    }
                    col += rw;
                }
            }
        }
        end_mask_l1_accumulate();
    }
    cb_push_back_hold_wr_ptr(cb_qkt_im, qk_cap);

    constexpr float kScale = __builtin_bit_cast(float, scale_fp32);
    constexpr float kLazyThrRaw = 16.0f * 0.69314718055994531f / kScale;
    constexpr uint16_t kLazyThrBits = static_cast<uint16_t>(__builtin_bit_cast(uint32_t, kLazyThrRaw) >> 16);
    uint32_t n_init = 0;
    for (uint32_t g = 0; g < n_slots; ++g) {
        n_init += slot_init(g) ? 1u : 0u;
    }
    bool rescale = false;
#if defined(SPARSE_MEMBER_RESCALE)
    // TT_SPARSE_RING_MEMBER_RESCALE: bit g of rescale_mask = member g's own max grew past the threshold, and
    // SPARSE_RS(g) = member g is rescaled this step. Without the define the expressions reduce to the per-step rule.
    uint32_t rescale_mask = 0;
#define SPARSE_RS(g) (((rescale_mask >> (g)) & 1u) != 0)
#define SPARSE_RS_SKIP(g) (!slot_init(g) || !SPARSE_RS(g))
#else
#define SPARSE_RS(g) rescale
#define SPARSE_RS_SKIP(g) (!slot_init(g))
#endif
    CircularBuffer(cb_qkt_im).wait_front(qk_cap);
    {
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

        if (n_init > 0) {
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
#if defined(SPARSE_MEMBER_RESCALE)
            // Per member: the k-th initialized member's Sq_chunk_t d tiles (pages k * Sq_chunk_t ..) column-max-reduced
            // into its own DEST tile, packed to page k. A batch's result pages lie below the d pages it reads and at or
            // above those of the members already reduced, so no page is overwritten before it has been read.
            for (uint32_t k = 0; k < n_init;) {
                const uint32_t nk = n_init - k < kDst ? n_init - k : kDst;
                tile_regs_acquire();
                reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(
                    cb_exp_max_diff, cb_identity_scale_in, cb_exp_max_diff);
                for (uint32_t kk = 0; kk < nk; ++kk) {
                    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(
                            cb_exp_max_diff, cb_identity_scale_in, (k + kk) * Sq_chunk_t + r, 0, kk);
                    }
                }
                reduce_uninit();
                tile_regs_commit();
                tile_regs_wait();
                configure_single_tile_pack(cb_exp_max_diff);
                for (uint32_t kk = 0; kk < nk; ++kk) {
                    sdpa_pack_tile_ooo(kk, cb_exp_max_diff, k + kk);
                }
                tile_regs_release();
                k += nk;
            }
#else
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
#endif
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
            UNPACK({
                ckernel::tensix_sync();
                while (ckernel::semaphore_read(semaphore::PACK_DONE) == 0) {
                }
                t6_semaphore_get<>(semaphore::PACK_DONE);
                invalidate_l1_cache();
            });
#if defined(SPARSE_MEMBER_RESCALE)
            // The unpack thread reads every initialized member's growth (element [0][0] of its page) and hands the
            // mask to the math and pack threads (one mailbox word each, as read_tile_value does), so all three take
            // the same per-member branches.
            UNPACK({
                uint32_t k = 0;
                for (uint32_t g = 0; g < n_slots; ++g) {
                    if (!slot_init(g)) {
                        continue;
                    }
                    const uint32_t growth = reinterpret_cast<volatile uint32_t*>(
                        ckernel::get_tile_l1_byte_address(get_operand_id(cb_exp_max_diff), k++))[0];
                    if ((growth & 0xFFFFu) > kLazyThrBits) {
                        rescale_mask |= 1u << g;
                    }
                }
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, rescale_mask);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, rescale_mask);
            });
            MATH((rescale_mask = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
            PACK((rescale_mask = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
            rescale = rescale_mask != 0;
#else
            const uint32_t growth = ckernel::read_tile_value(cb_exp_max_diff, 0, 0);
            rescale = (growth & 0xFFFFu) > kLazyThrBits;
#endif
        }

        if (rescale) {
            exp_packthread_tile_init<EXP_APPROX_MODE>();
        }
        for (uint32_t g0 = 0; g0 < n_slots; g0 += alpha_slots_per_acq) {
            const uint32_t n_g = n_slots - g0 < alpha_slots_per_acq ? n_slots - g0 : alpha_slots_per_acq;
            uint32_t n_alpha = 0;
            uint32_t n_copy = 0;
            for (uint32_t gg = 0; gg < n_g; ++gg) {
                const bool init = slot_init(g0 + gg);
                n_alpha += (SPARSE_RS(g0 + gg) && init) ? 1u : 0u;
                n_copy += (SPARSE_RS(g0 + gg) || !init) ? 1u : 0u;
            }
            if (n_copy == 0) {
                continue;
            }
            tile_regs_acquire();
            if (n_alpha > 0) {
                sub_init(cb_max_in, cb_max_scratch);
                uint32_t a = 0;
                for (uint32_t gg = 0; gg < n_g; ++gg) {
                    if (SPARSE_RS_SKIP(g0 + gg)) {
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
                    if (!SPARSE_RS(g0 + gg) && slot_init(g0 + gg)) {
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
                    if (SPARSE_RS_SKIP(g0 + gg)) {
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
                    if (!SPARSE_RS(g0 + gg) && slot_init(g0 + gg)) {
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

        if (rescale) {
            sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_prev_out, cb_qkt_im, cb_exp_max_diff>();
            sdpa_maybe_pack_reconfig_data_format<cb_qkt_im, cb_prev_out>();
            for (uint32_t g = 0; g < n_slots; ++g) {
                if (SPARSE_RS_SKIP(g)) {
                    continue;
                }
                salad_correct_fused<Sq_chunk_t, vDHt, kDst>(
                    cb_prev_out, cb_sum_in, cb_exp_max_diff, cb_prev_out, cb_sum_in, slots[g], slots[g], slots[g]);
            }
            sdpa_maybe_reconfig_data_format<cb_prev_out, cb_qkt_im, cb_exp_max_diff, cb_qkt_im>();
            sdpa_maybe_pack_reconfig_data_format<cb_prev_out, cb_qkt_im>();
        }
    }

#undef SPARSE_RS
#undef SPARSE_RS_SKIP
    exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
    // Exp headroom: P = exp(scale * (s - m) - kExpOctaves ln 2). The fast exp computes 256 log2(e) scale x + (B - C)
    // as INT16; lowering its constant (LREG13) by 256 per octave scales P by exactly 2^-kExpOctaves (the error
    // pattern is unchanged, the scale cancels at normalization) and moves saturation from 0.72 to kExpOctaves ln 2 +
    // 0.72 above the stored max, past the rescale threshold. Underflow moves up by as much (to -68 here): still 0.
    PACK({
        constexpr uint32_t kExpOctaves = 28;
        constexpr float exp_c = 32500.818359375f - 256.0f * kExpOctaves;
        constexpr uint32_t exp_c_bits = __builtin_bit_cast(uint32_t, exp_c);
        TTI_SFPLOADI(0, 0xA, exp_c_bits & 0xFFFF);
        TTI_SFPLOADI(0, 0x8, exp_c_bits >> 16);
        TTI_SFPCONFIG(0, 13, 0);
    })
    CircularBuffer(cb_sum_identity).wait_front(1);
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
        st.pk_w = 0;
        for (uint32_t g = 0; g < n_slots; ++g) {
            const uint32_t w = W[g];
            for (uint32_t c0 = 0; c0 < w;) {
                const uint32_t cw = w - c0 < max_cols_per_acq ? w - c0 : max_cols_per_acq;
                if (st.uu_tiles + Sq_chunk_t * cw > kDst) {
                    s5_flush(st, false);
                }
                if (cw != st.init_cw) {
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

    sdpa_maybe_reconfig_data_format<cb_v_in, cb_sum_identity, cb_qkt_im, cb_qkt_im>();
    sdpa_maybe_pack_reconfig_data_format<cb_qkt_im, cb_sum_in>();
    configure_single_tile_pack(cb_sum_in);
    {
        uint32_t rs_stride = 0;
        for (uint32_t g0 = 0; g0 < n_slots; g0 += stat_slots_per_acq) {
            const uint32_t n_g = n_slots - g0 < stat_slots_per_acq ? n_slots - g0 : stat_slots_per_acq;
            tile_regs_acquire();
            for (uint32_t gg = 0; gg < n_g; ++gg) {
                const uint32_t g = g0 + gg;
                if (W[g] != rs_stride) {
                    sdpa_pv_mm_setup(cb_qkt_im, cb_sum_identity, false, 1, Sq_chunk_t, W[g]);
                    rs_stride = W[g];
                }
                for (uint32_t c = 0; c < W[g]; ++c) {
                    sdpa_matmul_block_no_mop(
                        cb_qkt_im, cb_sum_identity, sidx(g, 0, c), 0, gg * Sq_chunk_t, false, 1, Sq_chunk_t, W[g]);
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
    sdpa_maybe_reconfig_data_format<cb_sum_identity, cb_qkt_im, cb_qkt_im, cb_qkt_im>();
    sdpa_maybe_pack_reconfig_data_format<cb_sum_in, cb_qkt_im>();

    CircularBuffer(cb_qkt_im).pop_front(qk_cap);
    CircularBuffer(cb_max_scratch).pop_front(stats_cap);
    CircularBuffer(cb_exp_max_diff).pop_front(stats_cap);
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

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
    uint32_t kv_cb_pages,
    bool has_block_counts,
    uint32_t cb_key_bias,
    uint32_t key_bias_tiles,
    uint32_t cb_sum_identity>
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
    [[maybe_unused]] uint32_t& kb_pos,
    const bool is_last_ring_iter,
    const bool is_first_active_iter,
    [[maybe_unused]] const volatile uint8_t* block_counts) {
    static_assert(resident_slots >= 1 && resident_slots <= 32, "resident state is tracked in a 32-bit mask");
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    static_assert(
        !sdpa_unpack_format_changed<cb_qkt_im, cb_max_in>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_max_scratch>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_exp_max_diff>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_identity_scale_in>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_recip_scratch>() &&
            !sdpa_unpack_format_changed<cb_qkt_im, cb_col_identity>() &&
            !sdpa_unpack_format_changed<cb_prev_out, cb_sum_in>(),
        "resident mode: the max, scratch and scale CBs must use the step's intermediate format, the output and sum "
        "slots one format");
#endif
#ifdef TRISC_PACK
    static_assert(
        pack_dst_format[cb_prev_out] == static_cast<uint32_t>(DataFormat::Float32) &&
            pack_dst_format[cb_sum_in] == static_cast<uint32_t>(DataFormat::Float32) &&
            pack_dst_format[cb_max_in] == pack_dst_format[cb_qkt_im] &&
            pack_dst_format[cb_max_scratch] == pack_dst_format[cb_qkt_im] &&
            pack_dst_format[cb_exp_max_diff] == pack_dst_format[cb_qkt_im] &&
            pack_dst_format[cb_recip_scratch] == pack_dst_format[cb_qkt_im],
        "resident mode: the output and sum slots must be Float32, the max and scratch CBs the step's intermediate "
        "format");
#endif
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t kt_entry_tiles = DHt * Sk_chunk_t;
    constexpr uint32_t v_entry_tiles = Sk_chunk_t * v_cb_physical_width_t;

    constexpr uint32_t kBlockTiles = get_named_compile_time_arg_val("block_tiles");
    constexpr uint32_t kBlocksPerPiece = Sk_chunk_t / kBlockTiles;
    constexpr uint32_t kBW = plan_block_words;
    static_assert(Sk_chunk_t % kBlockTiles == 0, "a piece holds whole blocks");
    static_assert(kBW >= 1 && kBW + 3 <= 16, "the plan header holds the block words, the union bits and the flag");
    static_assert(kBlocksPerPiece <= 255 && Sk_chunk_t <= 255, "run tables are byte-indexed");

    constexpr uint32_t kGroupCap = resident_slots < 4 ? resident_slots : 4;
    constexpr uint32_t kQkTiles = Sq_chunk_t * Sk_chunk_t;
    // Pair steps (two K/V pieces per step) only with one Q pass, unless SPARSE_MULTIPASS_PAIRS
    // (TT_SPARSE_RING_MULTIPASS_PAIRS, whose host side builds the reader at -Os to make room). With several passes the
    // kernel otherwise takes single steps: the pair path costs ~1.4-1.5 KB of text on every TRISC, and the dense-prefix
    // 10 s geometry (2 passes) would not fit the kernel config buffer with it. The QK scratch keeps the host's CB size
    // (kCbPieces): every step reserves, pushes and pops exactly kQkCap tiles, which must equal the CB's fifo size.
#if defined(SPARSE_MULTIPASS_PAIRS)
    constexpr bool kMultipassPairs = true;
#else
    constexpr bool kMultipassPairs = false;
#endif
    constexpr uint32_t kCbPieces = pair_steps ? 2 : 1;
    constexpr uint32_t kStepPieces = (pair_steps && (num_q_passes == 1 || kMultipassPairs)) ? 2 : 1;
    constexpr uint32_t kRunStride = kBlocksPerPiece * kStepPieces;
    static_assert(kRunStride <= 255, "run tables are byte-indexed");
    static_assert(!pair_steps || kv_cb_pages >= 2, "a pair needs two K/V entries in the CB");
    constexpr uint32_t kQkCap = kQkTiles > kGroupCap * Sq_chunk_t * Sk_chunk_t * kCbPieces
                                    ? kQkTiles
                                    : kGroupCap * Sq_chunk_t * Sk_chunk_t * kCbPieces;
    constexpr uint32_t kStatsCap = resident_slots * Sq_chunk_t;
    constexpr uint32_t kOutCap = resident_slots * out_tiles;
#ifdef TRISC_PACK
    ASSERT(get_local_cb_interface(cb_qkt_im).fifo_num_pages == kQkCap);
    ASSERT(get_local_cb_interface(cb_kt_in).fifo_num_pages == kv_cb_pages * kt_entry_tiles);
    ASSERT(get_local_cb_interface(cb_v_in).fifo_num_pages == kv_cb_pages * v_entry_tiles);
    ASSERT(get_local_cb_interface(cb_max_scratch).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_exp_max_diff).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_max_in).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_sum_in).fifo_num_pages == kStatsCap);
    ASSERT(get_local_cb_interface(cb_prev_out).fifo_num_pages == kOutCap);
#endif

    const uint32_t n_res = global_q_end - global_q_start;
    ASSERT(n_res <= resident_slots);

    constexpr uint32_t kRec = kBW + 3;
    constexpr bool kPlanInL1 = plan_rec_base != 0;
    [[maybe_unused]] uint32_t plan_copy[kPlanInL1 ? 1 : resident_slots * kRec];
    [[maybe_unused]] const volatile uint32_t* plan_l1 = nullptr;
    if constexpr (kPlanInL1) {
        CircularBuffer(cb_sparse_plan).wait_front(1);
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
    [[maybe_unused]] const uint32_t do_pooled = n_res > 0 ? plan_word(0, kBW + 2) : 0u;
    const auto bit = [](uint32_t lo, uint32_t hi, uint32_t k) -> bool {
        return k < 32 ? ((lo >> k) & 1u) != 0 : ((hi >> (k - 32)) & 1u) != 0;
    };
    const auto block_kept = [&](uint32_t i, uint32_t b) -> bool {
        return b < kBW * 32 && ((plan_word(i, b >> 5) >> (b & 31)) & 1u) != 0;
    };
    constexpr uint32_t kMaskNb = get_named_compile_time_arg_val("mask_nb");
    [[maybe_unused]] constexpr uint32_t kMaskNbPad = get_named_compile_time_arg_val("mask_nb_pad");
    [[maybe_unused]] constexpr uint32_t kLogicalN = get_named_compile_time_arg_val("sparse_logical_n");
    const auto blk_count = [&](uint32_t b) -> uint32_t {
        if (b >= kMaskNb) {
            return 0;
        }
        if constexpr (has_block_counts) {
            return block_counts[ring_id * kMaskNbPad + b];
        } else {
            return sparse_blocks::geometric_len<kBlockTiles * 32>(local_padded_Nt * 32, kLogicalN, kMaskNb, ring_id, b);
        }
    };
    [[maybe_unused]] const auto piece_bias_tiles = [&](uint32_t k) -> uint32_t {
        uint32_t n = 0;
        for (uint32_t j = 0; j < kBlocksPerPiece; ++j) {
            n += blk_count(k * kBlocksPerPiece + j) % 32 != 0 ? 1u : 0u;
        }
        return n;
    };
    uint32_t initialized = is_first_active_iter ? 0u : initialized_slots;

    if (is_first_active_iter) {
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

    uint32_t step_entries = 1;
    const uint8_t* step_bias_cols = nullptr;
    uint32_t step_n_bias = 0;
    const auto run_step = [&](const uint8_t* slots,
                              const uint32_t n_slots,
                              const uint8_t* run_cols,
                              const uint8_t* run_ws,
                              const uint8_t* n_runs,
                              const bool apply_col_bias) {
        CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles * step_entries);
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
            cb_sum_identity,
            cb_max_scratch,
            cb_exp_max_diff,
            cb_max_in,
            cb_sum_in,
            cb_prev_out,
            cb_out,
            cb_pooled_bias,
            cb_key_bias,
            v_cb_physical_width_t,
            kQkCap,
            kStatsCap,
            kStepPieces>(
            slots,
            n_slots,
            run_cols,
            run_ws,
            n_runs,
            initialized,
            apply_col_bias,
            step_entries,
            step_bias_cols,
            step_n_bias);
        for (uint32_t g = 0; g < n_slots; ++g) {
            initialized |= 1u << slots[g];
        }
    };
    uint8_t m_chunk[resident_slots];
    uint8_t m_width[resident_slots];
    uint8_t m_cols[resident_slots * kRunStride];
    uint8_t m_ws[resident_slots * kRunStride];
    uint8_t m_nruns[resident_slots];
    const auto run_groups = [&](const uint32_t n, const bool apply_col_bias) {
        res_run_groups<kRunStride, Sq_chunk_t, kQkCap>(
            run_step, m_chunk, m_width, m_cols, m_ws, m_nruns, n, apply_col_bias);
    };
    const auto pop_kv = [&]() {
        CircularBuffer(cb_kt_in).wait_front(kt_entry_tiles);
        sdpa_cb_pop_front_out_of_line(cb_kt_in, kt_entry_tiles);
        CircularBuffer(cb_v_in).wait_front(v_entry_tiles);
        sdpa_cb_pop_front_out_of_line(cb_v_in, v_entry_tiles);
        kv_pos = kv_pos + 1 == kv_cb_pages ? 0 : kv_pos + 1;
    };

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
    const auto step_pieces = [&](uint32_t head_mask, const uint8_t* ks, uint32_t np) {
        uint8_t blk_tiles[kRunStride];
        uint8_t bias_cols[kRunStride];
        uint32_t n_bias = 0;
        for (uint32_t p = 0; p < np; ++p) {
            for (uint32_t j = 0; j < kBlocksPerPiece; ++j) {
                const uint32_t c = blk_count(ks[p] * kBlocksPerPiece + j);
                const uint32_t tiles = (c + 31) / 32;
                blk_tiles[p * kBlocksPerPiece + j] = static_cast<uint8_t>(tiles);
                if (c % 32 != 0) {
                    bias_cols[n_bias++] = static_cast<uint8_t>(p * Sk_chunk_t + j * kBlockTiles + tiles - 1);
                }
            }
        }
        uint32_t n_group = 0;
        for (uint32_t i = 0; i < n_res; ++i) {
            if (((head_mask >> i) & 1u) == 0) {
                continue;
            }
            uint32_t cnt = 0;
            uint32_t width = 0;
            for (uint32_t p = 0; p < np; ++p) {
                [[maybe_unused]] const uint32_t sk = active_sk(ks[p]);
                bool prev_open = false;
                for (uint32_t j = 0; j < kBlocksPerPiece; ++j) {
                    const bool kept = block_kept(i, ks[p] * kBlocksPerPiece + j);
                    const uint32_t tiles = blk_tiles[p * kBlocksPerPiece + j];
                    if (kept) {
                        ASSERT(tiles > 0 && j * kBlockTiles + tiles <= sk);
                        if (prev_open) {
                            m_ws[n_group * kRunStride + cnt - 1] += tiles;
                        } else {
                            m_cols[n_group * kRunStride + cnt] = static_cast<uint8_t>(p * Sk_chunk_t + j * kBlockTiles);
                            m_ws[n_group * kRunStride + cnt] = static_cast<uint8_t>(tiles);
                            ++cnt;
                        }
                        width += tiles;
                    }
                    prev_open = kept && tiles == kBlockTiles;
                }
            }
            if (cnt == 0) {
                continue;
            }
            m_chunk[n_group] = static_cast<uint8_t>(i);
            m_width[n_group] = static_cast<uint8_t>(width);
            m_nruns[n_group] = static_cast<uint8_t>(cnt);
            ++n_group;
        }
        if constexpr (key_bias_tiles > 0) {
            if (n_bias > 0) {
                ASSERT(n_bias <= kBlocksPerPiece);
                if (kb_pos + n_bias > key_bias_tiles) {
                    CircularBuffer(cb_key_bias).wait_front(key_bias_tiles - kb_pos);
                    CircularBuffer(cb_key_bias).pop_front(key_bias_tiles - kb_pos);
                    kb_pos = 0;
                }
                CircularBuffer(cb_key_bias).wait_front(n_bias);
            }
        }
        if (n_group > 0) {
            step_entries = np;
            step_bias_cols = bias_cols;
            step_n_bias = n_bias;
            run_groups(n_group, false);
            step_entries = 1;
            step_n_bias = 0;
        }
        if constexpr (key_bias_tiles > 0) {
            if (n_bias > 0) {
                CircularBuffer(cb_key_bias).pop_front(n_bias);
                kb_pos = kb_pos + n_bias == key_bias_tiles ? 0 : kb_pos + n_bias;
            }
        }
        for (uint32_t p = 0; p < np; ++p) {
            pop_kv();
        }
    };
    const auto piece_live = [&](uint32_t k) -> bool {
        return kv_chunk_starts_before_logical_end<false, false, local_padded_Nt, chunk_size_t, kv_region_Nt>(
            ring_id, k * Sk_chunk_t, logical_nt);
    };

    bool one_head = true;
    for (uint32_t i = 1; i < n_res; ++i) {
        one_head = one_head && res_head(i) == res_head(0);
    }
    if (kStepPieces == 2 && one_head && n_res > 0) {
        constexpr uint32_t kMaxPieces = 64;
        uint8_t ks[kMaxPieces];
        uint32_t n_ks = 0;
        for (uint32_t k = 0; k < num_local_k_chunks && n_ks < kMaxPieces; ++k) {
            if (piece_live(k) && bit(union_lo(0), union_hi(0), k)) {
                ks[n_ks++] = static_cast<uint8_t>(k);
            }
        }
        const uint32_t all = n_res >= 32 ? 0xFFFFFFFFu : (1u << n_res) - 1u;
        for (uint32_t idx = 0; idx < n_ks;) {
            bool pair = idx + 1 < n_ks && active_sk(ks[idx]) == Sk_chunk_t && active_sk(ks[idx + 1]) == Sk_chunk_t &&
                        kv_pos + 2 <= kv_cb_pages;
            if constexpr (key_bias_tiles > 0) {
                pair = pair && (piece_bias_tiles(ks[idx]) == 0 || piece_bias_tiles(ks[idx + 1]) == 0);
            }
            const uint32_t np = pair ? 2u : 1u;
            step_pieces(all, ks + idx, np);
            idx += np;
        }
    } else {
        for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
            if (!piece_live(k)) {
                continue;
            }
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
                if (!bit(union_lo(i0), union_hi(i0), k)) {
                    continue;
                }
                const uint8_t k8 = static_cast<uint8_t>(k);
                step_pieces(head_mask, &k8, 1);
            }
        }
    }

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
                    m_cols[n_p * kRunStride] = 0;
                    m_nruns[n_p] = 1;
                    ++n_p;
                }
                for (uint32_t pc = 0; pc < pooled_chunks; ++pc) {
                    const uint32_t active_width = (pc + 1 == pooled_chunks) ? pooled_last_tiles : Sk_chunk_t;
                    for (uint32_t g = 0; g < n_p; ++g) {
                        m_width[g] = static_cast<uint8_t>(active_width);
                        m_ws[g * kRunStride] = static_cast<uint8_t>(active_width);
                    }
                    run_groups(n_p, true);
                    pop_kv();
                }
            }
        }
    }
    initialized_slots = initialized;

    if (is_last_ring_iter) {
        for (uint32_t i = 0; i < n_res; ++i) {
            if (((initialized >> i) & 1u) == 0) {
                reconfig_data_format(cb_identity_scale_in, cb_identity_scale_in);
                pack_reconfig_data_format(cb_out);
                configure_single_tile_pack(cb_out);
                zero_block(cb_identity_scale_in, cb_out, out_tiles);
                pack_reconfig_data_format(cb_recip_scratch);
                sdpa_cb_pop_front_out_of_line(cb_sum_in, Sq_chunk_t);
                sdpa_cb_pop_front_out_of_line(cb_prev_out, out_tiles);
                continue;
            }
            res_normalize_slot<Sq_chunk_t, vDHt, cb_col_identity, cb_recip_scratch, cb_out, cb_sum_in, cb_prev_out>();
        }
        if (n_res < resident_slots) {
            sdpa_cb_pop_front_out_of_line(cb_sum_in, (resident_slots - n_res) * Sq_chunk_t);
            sdpa_cb_pop_front_out_of_line(cb_prev_out, (resident_slots - n_res) * out_tiles);
        }
        sdpa_cb_pop_front_out_of_line(cb_max_in, kStatsCap);
        if constexpr (num_q_passes > 1) {
            CircularBuffer(cb_q_in).wait_front(resident_slots * q_chunk_tiles);
            sdpa_cb_pop_front_out_of_line(cb_q_in, resident_slots * q_chunk_tiles);
        } else {
            sdpa_cb_pop_front_out_of_line(cb_q_in, n_res * q_chunk_tiles);
        }
    }
    if constexpr (kPlanInL1) {
        CircularBuffer(cb_sparse_plan).pop_front(1);
    }
}
