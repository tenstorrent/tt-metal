// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// RESEARCH PROTOTYPE (fused FP32 K chunk for BALANCED / ACCURATE). Included by recipe_streaming.hpp before
// sdpa_inner_loop_step, which dispatches every K chunk after a Q chunk's first to sdpa_fused_fp32_chunk.
//
//  - m_ref: the exact (TF32) row maxima in prev.max, kept for the whole Q chunk unless a group is redone.
//  - QK accumulates s - m_ref straight into FP32 DEST: one extra inner step [Q | M2] x [K^T ; -e0 - e1] at one
//    fidelity phase, inner half. M2 (CB 30, BF16) holds m_ref split as hi (col 0, 7 significant bits) and
//    lo = m_ref - hi (col 1, <= 4 bits for TF32 maxima), so srcB's 7 bits carry both exactly.
//  - The pack thread takes the FP32 score exp (grid + refine) in place and packs P once (FP32).
//  - The exp grid uses 9 fraction bits (FFP_GRID9, recipe_sfpu.hpp): its window is [-21.5, +22.9] nat around
//    m_ref instead of [-21.5, +0.72], so P may exceed 1 by e^22.
//  - Denominator: P . 1 (same fidelity rule as the reduce path), one tile per row into CB 14 (scratch).
//    UNPACK scans it: a row sum >= 2^24 (a P near saturation, or inf/nan) redoes the group on the reduce path
//    (real max, rescale of its O and l rows, new m_ref). Otherwise it is folded into l by L1 accumulation.
//  - Lag-2 pipeline over one-row groups: QK/exp of group g, denominator + check of g - 1, PV of g - 2.
//  - O and l accumulate in place (banks swapped as in the in-place numerator path); prev.max stays the reference.
#pragma once

#ifdef SDPA_RECIPE_FFP32_ACTIVE
constexpr uint32_t kFfpNegUnitCb = 29;  // sbw tiles, K format (BF16): columns 0 and 1 = -1
constexpr uint32_t kFfpRefCb = 30;      // Sq tiles, BF16: row m_ref split, col 0 = hi, col 1 = lo
#ifndef FFP_REDO_BITS
#define FFP_REDO_BITS 0x4B800000u  // 2^24
#endif
static bool ffp_static_ready = false;
static bool ffp_ref_valid = false;

#ifdef TRISC_UNPACK
// BF16 bits of m - hi, hi = m with its 17 low bits cleared (7 significant bits).
inline uint32_t ffp_lo_bf16(uint32_t u) {
    const uint32_t e = (u >> 23) & 0xffu;
    const uint32_t v = u & 0x1ffffu;
    if (v == 0 || e == 0 || e == 0xffu) {
        return 0;
    }
    const uint32_t p = 31u - static_cast<uint32_t>(__builtin_clz(v));
    const int32_t elo = static_cast<int32_t>(e) - 23 + static_cast<int32_t>(p);
    if (elo <= 0) {
        return 0;
    }
    const uint32_t mant = ((v << (23u - p)) & 0x7fffffu) >> 16;
    return ((u >> 16) & 0x8000u) | (static_cast<uint32_t>(elo) << 7) | mant;
}
// Word of row r's columns 0/1 in a BF16 tile (16 x 16 faces, rows 16-31 in face 2).
constexpr uint32_t ffp_bf16_pair_word(uint32_t r) { return (r / 16) * 256 + (r % 16) * 8; }
static __attribute__((noinline)) void ffp_build_ref(uint32_t max_cb, uint32_t row_begin, uint32_t row_end) {
    uint32_t sink = 0;
    for (uint32_t t = row_begin; t < row_end; ++t) {
        auto* m = reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(max_cb), t));
        auto* d = reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(kFfpRefCb), t));
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t u = m[sdpa_max_word(r)];
            d[ffp_bf16_pair_word(r)] = (u >> 16 & 0xfffeu) | (ffp_lo_bf16(u) << 16);
        }
        sink += d[ffp_bf16_pair_word(31)];
    }
    asm volatile("fence" ::"r"(sink) : "memory");
}
static __attribute__((noinline)) void ffp_build_static(uint32_t q_tiles, uint32_t sbw) {
    for (uint32_t t = 0; t < sbw; ++t) {
        auto* d = reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(kFfpNegUnitCb), t));
        for (uint32_t w = 0; w < 512; ++w) {
            d[w] = 0;
        }
        for (uint32_t r = 0; r < 32; ++r) {
            d[ffp_bf16_pair_word(r)] = 0xBF80BF80u;
        }
    }
    for (uint32_t t = 0; t < q_tiles; ++t) {
        auto* d = reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(kFfpRefCb), t));
        for (uint32_t w = 0; w < 512; ++w) {
            d[w] = 0;
        }
    }
    asm volatile("fence" ::: "memory");
}
#endif

#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// dest tiles [0, w): state; tile w: column-broadcast correction c. state *= c.
template <int w>
inline void ffp_state_scale() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat c = sfpi::dst_reg[32 * w];
#pragma GCC unroll 4
        for (int t = 0; t < w; ++t) {
            sfpi::dst_reg[32 * t] = sfpi::dst_reg[32 * t] * c;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

#ifdef SDPA_RECIPE_ACCURATE
constexpr bool kFfpAccurateExp = true;
constexpr int kFfpDenPhases = 3;  // phases 0-2 of the HiFi4 image: all of P's srcB bits against the exact ones
#else
constexpr bool kFfpAccurateExp = false;
constexpr int kFfpDenPhases = 1;  // LoFi, as the reduce path
#endif

template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_lsum,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    bool independent_q_release>
static __attribute__((noinline)) void sdpa_fused_fp32_chunk(
    AccumulatorHalf& prev, AccumulatorHalf& cur, bool is_last_iter, bool release_q) {
    constexpr uint32_t KT = Sk_chunk_t;
    constexpr uint32_t G = Sq_chunk_t;
    constexpr uint32_t sbw = qkt_subblock_w;
    constexpr uint32_t n_kb = KT / sbw;
    constexpr uint32_t n_pieces = n_kb % 2 == 0 ? 2 : 1;
    constexpr uint32_t kb_per_piece = n_kb / n_pieces;
    constexpr uint32_t piece_k = kb_per_piece * sbw;
    constexpr uint32_t pv_w = qktv_subblock_w;
    constexpr uint32_t n_pv = vDHt / pv_w;
    constexpr uint32_t n_o = Sq_chunk_t * vDHt;
    static_assert(KT % sbw == 0 && vDHt % pv_w == 0 && sbw <= 4 && pv_w <= 4);

    // One-time constants (UNPACK writes L1 directly; the other threads never read these CBs' bytes).
    if (!ffp_static_ready) {
        UNPACK((ffp_build_static(Sq_chunk_t, sbw)));
        ffp_static_ready = true;
    }
    CircularBuffer(prev.max).wait_front(Sq_chunk_t);
    if (!ffp_ref_valid) {
        UNPACK((ffp_build_ref(prev.max, 0, Sq_chunk_t)));
        ffp_ref_valid = true;
    }

    // O and l accumulate in place: pop the published banks and reserve them again (same slots, the CBs hold
    // exactly one Q chunk), as the in-place numerator path does.
    CircularBuffer(prev.out).wait_front(n_o);
    CircularBuffer(prev.out).pop_front(n_o);
    std::swap(prev.out, cur.out);
    CircularBuffer(prev.sum).wait_front(Sq_chunk_t);
    CircularBuffer(prev.sum).pop_front(Sq_chunk_t);
    std::swap(prev.sum, cur.sum);
    sdpa_skip_prev_sum_pop = true;
    const uint32_t out_cb = cur.out;
    const uint32_t sum_cb = cur.sum;
    CircularBuffer(out_cb).reserve_back(n_o);
    CircularBuffer(sum_cb).reserve_back(Sq_chunk_t);
    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT);
    CircularBuffer(cb_lsum).reserve_back(Sq_chunk_t);
    CircularBuffer(cb_kt_in).wait_front(DHt * KT);
    CircularBuffer(cb_q_in).wait_front(Sq_chunk_t * DHt);

    // Source format tracking (srcA, srcB). INVALID_CB forces a reconfiguration.
    uint32_t src_a = INVALID_CB, src_b = INVALID_CB;
    auto set_src = [&](uint32_t a, uint32_t b) {
        if (a != src_a || b != src_b) {
            reconfig_data_format_skip_int8(a, b);
            src_a = a;
            src_b = b;
        }
    };
    auto arm_exp = []() {
        exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
        PACK((ckernel::sfpu::init_sdpa_refine_loadmacros()));
    };
    arm_exp();
    bool qk_ready = false;
    bool relu_on = false;
    auto leave_qk = [&]() {
        qk_ready = false;
        if (relu_on) {
            PACK((llk_pack_relu_config(ReluConfig::none())));
            relu_on = false;
        }
    };

    // QK subblock (g, kb): DEST = -m_ref + Q K^T (FP32), exp in place, P packed.
    auto qk_subblock = [&](uint32_t g, uint32_t kb) {
        if (!qk_ready) {
            set_src(cb_kt_in, cb_q_in);
            recipe_mm_reinit(cb_q_in, cb_kt_in, true, sbw, 1, DHt);
            MATH((llk_math_matmul_reinit_no_mop<FFP_QK_FID, MM_THROTTLE>(cb_q_in, cb_kt_in, true, sbw, 1)));
            qk_ready = true;
        }
        tile_regs_acquire();
        uint32_t in0 = g * DHt, in1 = kb * sbw;
        for (uint32_t inner = 0; inner < DHt; ++inner) {
            UNPACK((llk_unpack_AB_matmul(cb_q_in, cb_kt_in, in0, in1, sbw, 1, DHt)));
            MATH((llk_math_matmul_no_mop<FFP_QK_FID, MM_THROTTLE>(cb_q_in, cb_kt_in, 0, sbw, 1)));
            in0++;
            in1 += KT;
        }
#ifndef FFP_KO_MSTEP
        UNPACK((llk_unpack_AB_matmul(kFfpRefCb, kFfpNegUnitCb, g, 0, sbw, 1, 1)));
        MATH((llk_math_matmul_no_mop<FFP_QK_FID, MM_THROTTLE, 1, true>(kFfpRefCb, kFfpNegUnitCb, 0, sbw, 1)));
#endif
        tile_regs_commit();
        tile_regs_wait();
#if defined(SDPA_RECIPE_K_PRIMARY_ROWS)
        mask_recipe_tail(kb * sbw, sbw, 1);
#endif
        PACK((ckernel::sfpu::restore_sdpa_grid_macro_instructions()));
        PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
        PACK((SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_grid_batch, (32 * sbw), 0, VectorMode::None)));
        PACK((SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sdpa_exp_refine_loadmacro,
            (32 * sbw, kFfpAccurateExp),
            0,
            VectorMode::None)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        if (!relu_on) {
            PACK((llk_pack_relu_config(ReluConfig::zero())));
            relu_on = true;
        }
        configure_row_pack_width(cb_qkt_im, sbw);
        pack_contiguous_rows_nocfg(cb_qkt_im, g, 1, KT, kb * sbw, sbw);
        tile_regs_release();
    };

    // Row sums of group g (P . 1) into the scratch row tile g.
    auto denominator = [&](uint32_t g, uint32_t target_cb, uint32_t index, bool acc) {
        leave_qk();
        CircularBuffer(cb_qkt_im).wait_front((g + 1) * KT);
        set_src(cb_col_identity, cb_qkt_im);
        recipe_mm_reinit(cb_qkt_im, cb_col_identity, false, 1, 1, KT);
        tile_regs_acquire();
        for (uint32_t k = 0; k < KT; ++k) {
            UNPACK((llk_unpack_AB_matmul(cb_qkt_im, cb_col_identity, g * KT + k, 0, 1, 1, KT)));
            MATH((llk_math_matmul_no_mop<MATH_FIDELITY, MM_THROTTLE, kFfpDenPhases>(
                cb_qkt_im, cb_col_identity, 0, 1, 1)));
        }
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(target_cb);
        if (acc) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        pack_tile<true>(0, target_cb, index);
        if (acc) {
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
        tile_regs_release();
    };

    // Verdict for group g: does any row sum reach the redo threshold.
    auto read_check = [&](uint32_t g) -> bool {
        uint32_t redo = 0;
        UNPACK({
            CircularBuffer(cb_lsum).wait_front(g + 1);
            auto* s = reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(cb_lsum), g));
            for (uint32_t r = 0; r < 32; ++r) {
                redo |= (s[sdpa_max_word(r)] & 0x7fffffffu) >= FFP_REDO_BITS ? 1u : 0u;
            }
#ifdef FFP_FORCE_REDO
            redo = 1;
#endif
            mailbox_write(ckernel::ThreadId::MathThreadId, redo);
            mailbox_write(ckernel::ThreadId::PackThreadId, redo);
        })
        MATH(redo = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(redo = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        return redo != 0;
    };

    bool v_ready = false;
    // PV of group g over K tiles [k0, k0 + k_len), L1-accumulated onto O row g (the next unpublished row).
    auto pv_piece = [&](uint32_t g, uint32_t k0, uint32_t k_len) {
        leave_qk();
        if (!v_ready) {
            CircularBuffer(cb_v_in).wait_front(KT * vDHt);
            v_ready = true;
        }
        CircularBuffer(cb_qkt_im).wait_front((g + 1) * KT);
        set_src(cb_v_in, cb_qkt_im);
        recipe_mm_reinit(cb_qkt_im, cb_v_in, false, pv_w, 1, KT);
        configure_row_pack_width(out_cb, pv_w);
        PACK((llk_pack_reconfig_l1_acc(1)));
        for (uint32_t v = 0; v < n_pv; ++v) {
            blocked_matmul_and_pack<false, vDHt, vDHt>(
                cb_qkt_im,
                cb_v_in,
                out_cb,
                g * KT + k0,
                k0 * vDHt + v * pv_w,
                0,
                v * pv_w,
                pv_w,
                1,
                k_len,
                KT,
                /*skip_pack_configure=*/true);
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
    };

    // l row g += this chunk's sums.
    auto fold = [&](uint32_t g) {
        leave_qk();
        set_src(cb_lsum, cb_qkt_im);
        copy_init(cb_lsum);
        tile_regs_acquire();
        copy_tile(cb_lsum, g, 0);
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(sum_cb);
        PACK((llk_pack_reconfig_l1_acc(1)));
        pack_tile<true>(0, sum_cb, 0);
        PACK((llk_pack_reconfig_l1_acc(0)));
        tile_regs_release();
    };

    auto finish_group = [&](uint32_t g) {
        leave_qk();
        CircularBuffer(sum_cb).push_back(1);
        CircularBuffer(out_cb).push_back(vDHt);
        if (is_last_iter) {
            normalize_row_streaming<
                vDHt,
                compute_kernel_lib::DEST_AUTO_LIMIT,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                scale_fp32>(sum_cb, out_cb, 1);
            src_a = src_b = INVALID_CB;
            PACK(sdpa_pack.invalidate();)
            arm_exp();
        }
    };

    // Redo group g on the reduce path: scores, real max, P, then O, l *= c, PV and P . 1 accumulated, m_ref = m.
    auto redo_group = [&](uint32_t g) __attribute__((noinline)) {
        leave_qk();
        const uint32_t read_row = is_last_iter ? 0 : g;
        // Scores (FP32) over this row's P.
        set_src(cb_kt_in, cb_q_in);
        recipe_mm_init(cb_q_in, cb_kt_in, true, sbw, 1, DHt);
        MATH((llk_math_matmul_reinit_no_mop<FFP_QK_FID, MM_THROTTLE>(cb_q_in, cb_kt_in, true, sbw, 1)));
        configure_row_pack_width(cb_qkt_im, sbw);
        for (uint32_t kb = 0; kb < n_kb; ++kb) {
            blocked_matmul_and_pack<true, KT, KT>(
                cb_q_in, cb_kt_in, cb_qkt_im, g * DHt, kb * sbw, g, kb * sbw, sbw, 1, DHt, DHt, true);
        }
        recipe_pack_to_unpack_barrier();
        // m = max(m_ref, rowmax) into scratch row g of cur.max.
        reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT, scale_fp32>(
            cur.max, prev.max, g, true, 1, KT, false, false, g);
        PACK(sdpa_pack.invalidate();)
        recipe_pack_to_unpack_barrier();
        // c = exp(scale (m_ref - m))
        exp_packthread_tile_init<EXP_APPROX_MODE>();
        reconfig_data_format_skip_int8(prev.max, cur.max);
        sub_init(prev.max, cur.max);
        tile_regs_acquire();
        sub_tiles(prev.max, cur.max, g, g, 0);
        tile_regs_commit();
        CircularBuffer(cb_recip_scratch).reserve_back(1);
        tile_regs_wait();
        PACK((SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_correction, (scale_fp32), 0, VectorMode::C)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        configure_single_tile_pack(cb_recip_scratch);
        pack_tile(0, cb_recip_scratch);
        tile_regs_release();
        CircularBuffer(cb_recip_scratch).push_back(1);
        // P against m
        arm_exp();
        for (uint32_t kb = 0; kb < n_kb; ++kb) {
            sub_exp_block_bcast_cols<scale_fp32>(cb_qkt_im, cur.max, cb_lsum, KT, g, kb * sbw, 1, sbw, false, false);
        }
        PACK(sdpa_pack.invalidate();)
        recipe_pack_to_unpack_barrier();
        // O, l *= c in place
        CircularBuffer(cb_recip_scratch).wait_front(1);
        PACK((llk_pack_reconfig_l1_acc(0)));
        for (uint32_t j = 0; j < vDHt; j += 2) {
            const bool pair = j + 1 < vDHt;
            tile_regs_acquire();
            reconfig_data_format_skip_int8(out_cb, out_cb);
            copy_init(out_cb);
            copy_tile(out_cb, read_row * vDHt + j, 0);
            if (pair) {
                copy_tile(out_cb, read_row * vDHt + j + 1, 1);
            }
            reconfig_data_format_skip_int8(cb_recip_scratch, cb_recip_scratch);
            unary_bcast_init<BroadcastType::COL>(cb_recip_scratch);
            unary_bcast<BroadcastType::COL>(cb_recip_scratch, 0, pair ? 2 : 1);
            unary_bcast_uninit<BroadcastType::COL>(cb_recip_scratch);
            tile_regs_commit();
            tile_regs_wait();
            if (pair) {
                PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, ffp_state_scale, (2), 0, VectorMode::None)));
            } else {
                PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, ffp_state_scale, (1), 0, VectorMode::None)));
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            configure_single_tile_pack(out_cb);
            pack_tile<true>(0, out_cb, j);
            if (pair) {
                pack_tile<true>(1, out_cb, j + 1);
            }
            tile_regs_release();
        }
        {
            tile_regs_acquire();
            reconfig_data_format_skip_int8(sum_cb, sum_cb);
            copy_init(sum_cb);
            copy_tile(sum_cb, read_row, 0);
            reconfig_data_format_skip_int8(cb_recip_scratch, cb_recip_scratch);
            unary_bcast_init<BroadcastType::COL>(cb_recip_scratch);
            unary_bcast<BroadcastType::COL>(cb_recip_scratch, 0, 1);
            unary_bcast_uninit<BroadcastType::COL>(cb_recip_scratch);
            tile_regs_commit();
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, ffp_state_scale, (1), 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            configure_single_tile_pack(sum_cb);
            pack_tile<true>(0, sum_cb, 0);
            tile_regs_release();
        }
        CircularBuffer(cb_recip_scratch).pop_front(1);
        src_a = src_b = INVALID_CB;
        recipe_pack_to_unpack_barrier();
        // O += P V, l += P . 1
        pv_piece(g, 0, KT);
        denominator(g, sum_cb, 0, true);
        // m_ref = m for this row
        reconfig_data_format_skip_int8(cur.max, cur.max);
        copy_init(cur.max);
        tile_regs_acquire();
        copy_tile(cur.max, g, 0);
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(prev.max);
        pack_tile<true>(0, prev.max, g);
        tile_regs_release();
        src_a = src_b = INVALID_CB;
        recipe_pack_to_unpack_barrier();
        UNPACK((ffp_build_ref(prev.max, g, g + 1)));
        arm_exp();
    };

    // Iteration it: QK of group it, denominator + check of group it - 1, PV of group it - 2.
    for (uint32_t it = 0; it < G + 2; ++it) {
        const bool do_qk = it < G;
        const bool do_den = it >= 1 && it <= G;
        bool do_pv = it >= 2;
        const uint32_t pg = it - 2;
        if (do_pv && read_check(pg)) {
            redo_group(pg);
            finish_group(pg);
            do_pv = false;
        }
        for (uint32_t piece = 0; piece < n_pieces; ++piece) {
            if (do_qk) {
                for (uint32_t kb = piece * kb_per_piece; kb < (piece + 1) * kb_per_piece; ++kb) {
                    qk_subblock(it, kb);
                }
                if (piece == n_pieces - 1) {
                    leave_qk();
                    cb_push_back_hold_wr_ptr(cb_qkt_im, KT);
                }
            }
            if (do_den && piece == 0) {
                denominator(it - 1, cb_lsum, it - 1, false);
                cb_push_back_hold_wr_ptr(cb_lsum, 1);
            }
            if (do_pv) {
                pv_piece(pg, piece * piece_k, piece_k);
            }
        }
        if (do_pv) {
            fold(pg);
            finish_group(pg);
        }
    }
    leave_qk();
    CircularBuffer(cb_kt_in).pop_front(DHt * KT);
    if (independent_q_release ? release_q : is_last_iter) {
        sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
    }
    CircularBuffer(cb_lsum).pop_front(Sq_chunk_t);
    CircularBuffer(cb_v_in).pop_front(KT * vDHt);
    CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT);
    PACK(sdpa_pack.invalidate();)
    // The unfused step reconfigures from (cb_qkt_im, ...) defaults; leave srcA/srcB at K/Q.
    set_src(cb_kt_in, cb_q_in);
}
#endif
