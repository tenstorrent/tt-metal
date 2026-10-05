// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// STANDARD's and LOW_PRECISION's fused K chunk (sdpa_fused_chunk). Included by recipe_streaming.hpp after the helpers it uses
// (reduce_c_row_group, sub_exp_block_bcast_cols, normalize_row_streaming, blocked_matmul_and_pack) and before
// sdpa_inner_loop_step, which dispatches to it.
//
// Code size. Every recipe kernel must fit the 70656 B kernel config buffer, and the fused chunk is the largest
// addition. What keeps the ring and exp ring builds inside it:
//   - SDPA_RECIPE_COLD (recipe_streaming.hpp): with fused chunks, the reduce path (a Q chunk's first K chunk,
//     redone groups), normalization and sdpa_inner_loop_step are out of line and -Os.
//   - SDPA_FUSED_CHUNK_ATTR (below): the ring kernels' unpack copy of this chunk is -Os (unpack only issues and
//     waits here, and its image is the one nearest the limit).
//   - ring_joint_sdpa_recipe.cpp / exp_ring_joint_sdpa_recipe.cpp: unpack and pack drop from -O3 to -O2 in
//     fused builds (math is -O2 everywhere).
//   - The ring and exp ring reader/writer dataflow kernels are -Os.
// The dense kernel fits at its default -O3.

#pragma once

#ifdef SDPA_RECIPE_FUSED_ACTIVE
/**
 * Fused STANDARD / LOW_PRECISION K chunk (every chunk after a Q chunk's first). The reference max m_ref is already
 * known, so the row max is not needed before the exp:
 *   - QK accumulates s - m_ref straight into DEST through one extra inner step, [Q | M] x [K^T ; -e0]
 *     (M: the reference-max tile, m_ref in column 0; -e0 in K's format). The pack thread takes the fast exp
 *     in place and packs P once, L1-accumulating it onto QK-subblock-width partial row sums.
 *   - Row groups are software-pipelined: QK/exp of group g, the saturation check of group g - 1 and the
 *     PV of group g - 2 are interleaved piece by piece, so the FPU's PV overlaps the pack thread's exp.
 *   - Saturation check: a group whose chunk row sums reach kFusedRedoSum may have saturated the exp (P
 *     saturates near 1.66 at this scale). Rows are independent, so only that group is redone on the
 *     reduce path (real max, theta select, rescale of its O and l rows); every other group keeps m_ref.
 * Same CB protocol as sdpa_inner_loop_step: scores, chunk sums, O and l banks, the max ping-pong.
 */
// Ring kernels: unpack only issues and waits here, so its copy is size-optimized (their unpack image is the one
// nearest the kernel config buffer limit).
#if defined(TRISC_UNPACK) && defined(SDPA_RECIPE_RING)
#define SDPA_FUSED_CHUNK_ATTR __attribute__((noinline, optimize("Os")))
#else
#define SDPA_FUSED_CHUNK_ATTR __attribute__((noinline))
#endif
static bool fused_neg_unit_ready = false;
#ifdef SDPA_RING_STREAM_STATE
#include "../../recipe_state_layout.hpp"
// Streamed ring checkpoints (recipe_ring.hpp, StateTransfer): a restored Q block's O rows land row by row while
// its first fused chunk runs, and the last fused chunk of a saved block hands its O rows to the writer group by
// group. kNoStreamSlot: no save in flight; restored rows >= every row end: no restore in flight.
constexpr uint32_t kStreamRequestCb = 17;
constexpr uint32_t kStreamAckCb = 18;
constexpr uint32_t kNoStreamSlot = 0xffffffffu;
static uint32_t recipe_stream_save_slot = kNoStreamSlot;
static uint32_t recipe_stream_saved_rows = 0;
static uint32_t recipe_stream_restored_rows = 0xffffffffu;
// Out of line throughout: the ring kernels sit at the kernel config buffer limit.
// UNPACK tells MATH and PACK it has passed a handshake (the threads leave it together).
static __attribute__((noinline)) void recipe_stream_sync_threads() {
    UNPACK({
        mailbox_write(ckernel::ThreadId::MathThreadId, 1);
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })
    MATH((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
}
static __attribute__((noinline)) void recipe_stream_wait_rows_slow(uint32_t row_end) {
    UNPACK({
        for (uint32_t r = recipe_stream_restored_rows; r < row_end; ++r) {
            llk_wait_tiles(kStreamAckCb, 1);
            llk_pop_tiles(kStreamAckCb, 1);
        }
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })
    PACK((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    recipe_stream_restored_rows = row_end;
}
// O rows [0, row_end) must have landed before any thread touches them (PACK accumulates PV onto them, UNPACK
// reads them in a redo). Every thread runs this with the same counters, so the mailbox calls pair up.
ALWI void recipe_stream_wait_rows(uint32_t row_end) {
    if (recipe_stream_restored_rows < row_end) {
        recipe_stream_wait_rows_slow(row_end);
    }
}
// Hand O rows [saved, row_end) of the block being saved to the writer (the push lands after their packs).
static __attribute__((noinline)) void recipe_stream_save_rows(uint32_t numerator_cb, uint32_t row_end) {
    CircularBuffer request(kStreamRequestCb);
    request.reserve_back(1);
    PACK({
        using Transfer = sdpa::streaming::StateTransfer;
        auto* words = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(kStreamRequestCb).fifo_wr_ptr << 4);
        words[Transfer::Operation] = Transfer::SaveRows;
        words[Transfer::Slot] = recipe_stream_save_slot;
        words[Transfer::Numerator] = numerator_cb;
        words[Transfer::Row0] = recipe_stream_saved_rows;
        words[Transfer::Rows] = row_end - recipe_stream_saved_rows;
    })
    request.push_back(1);
    recipe_stream_saved_rows = row_end;
}
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
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    bool independent_q_release>
static SDPA_FUSED_CHUNK_ATTR void sdpa_fused_chunk(
    AccumulatorHalf& prev, AccumulatorHalf& cur, bool is_last_iter, bool release_q) {
    constexpr uint32_t KT = Sk_chunk_t;
    // Rows per group. Dense STANDARD (math-bound) runs a short Q chunk in one-row groups, so the lagged QK / check /
    // PV pipeline has enough groups to overlap (1 core Q128: 1.93 -> 2.03 TF). LOW_PRECISION is pack-bound and keeps
    // two-row groups (one-row groups double its per-group pack work), and so do the ring kernels (H3 4x8 ring
    // block: one-row groups were 1-4% slower).
#if defined(SDPA_RECIPE_LOFI) || defined(SDPA_RECIPE_RING)
    constexpr uint32_t kH1MaxQTiles = 0;
#else
    constexpr uint32_t kH1MaxQTiles = 6;
#endif
    constexpr uint32_t H = Sq_chunk_t <= kH1MaxQTiles ? 1 : 2;
    constexpr uint32_t sbw = qkt_subblock_w;
    constexpr uint32_t n_kb = Sk_chunk_t / sbw;
    static_assert(Sk_chunk_t % sbw == 0 && vDHt % qktv_subblock_w == 0 && DHt == vDHt);
    // PV of a pipelined group runs in n_pieces K pieces, each after QK subblocks of the next group.
    constexpr uint32_t n_pieces = n_kb % 2 == 0 ? 2 : 1;
    constexpr uint32_t kb_per_piece = n_kb / n_pieces;
    constexpr uint32_t piece_k = kb_per_piece * sbw;
    constexpr uint32_t pv_w = qktv_subblock_w;
    constexpr uint32_t n_pv = vDHt / pv_w;
    constexpr bool tail = Sq_chunk_t % H != 0;
    constexpr uint32_t G = Sq_chunk_t / H + (tail ? 1 : 0);
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t neg_unit_cb = kFusedNegUnitCb;
    constexpr uint32_t lsum_cb = kRefMaxChunkSumCb;  // tile-shaped sums of a redone group
    constexpr uint32_t psum_cb = kFusedSumCb;
    constexpr uint32_t check_cb = kFusedCheckCb;
    const uint32_t out_cb = cur.out;
    auto rows = [](uint32_t g) -> uint32_t { return tail && g == G - 1 ? 1 : H; };
    // Group index in units of its own height (the helpers address rows as index * height).
    auto gindex = [](uint32_t g) -> uint32_t { return tail && g == G - 1 ? H * g : g; };

    // Runtime unpack-A / pack format tracking: the chunk interleaves broadcasts, matmuls, reduces and folds.
    uint32_t srca = cb_qkt_im;      // step entry state
    uint32_t pack_cb = cb_qkt_im;
    auto set_srca = [&](uint32_t cb) {
        if (cb != srca) {
            reconfig_data_format_srca(srca, cb);
            srca = cb;
        }
    };
    auto set_pack = [&](uint32_t cb) {
        if (cb != pack_cb) {
            pack_reconfig_data_format(pack_cb, cb);
            pack_cb = cb;
        }
    };
    // The fast exp's SFPU program and constants (28-octave headroom). Other SFPU users interleaved with the
    // exps (normalization's reciprocal, a redone group's correction and rescale) reprogram the shared
    // SFPU, so the exp is re-armed after them.
    auto arm_exp = []() {
        exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
        PACK({
            constexpr float exp_c = 32500.818359375f - 256.0f * kRefMaxExpOctaves;
            constexpr uint32_t exp_c_bits = __builtin_bit_cast(uint32_t, exp_c);
            TTI_SFPLOADI(0, 0xA, exp_c_bits & 0xFFFF);
            TTI_SFPLOADI(0, 0x8, exp_c_bits >> 16);
            TTI_SFPCONFIG(0, 13, 0);
        })
    };
    auto pack_to_unpack_barrier = []() {
        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
    };

    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT);
    CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);
    CircularBuffer(lsum_cb).reserve_back(Sq_chunk_t);
    CircularBuffer(psum_cb).reserve_back(Sq_chunk_t * sbw);
    CircularBuffer(out_cb).reserve_back(Sq_chunk_t * vDHt * sdpa_out_stride);
    CircularBuffer(cb_kt_in).wait_front(DHt * KT);
    CircularBuffer(cb_q_in).wait_front(Sq_chunk_t * DHt);

    // The reference max stays in prev.max for every fused chunk; a redone group updates its rows in place.
    {
    CircularBuffer(prev.max).wait_front(Sq_chunk_t);
    // -e0 in K's format, built once per program from the column identity (the packer converts the format).
    if (!fused_neg_unit_ready) {
        CircularBuffer(cb_col_identity).wait_front(1);
        set_srca(cb_col_identity);
        copy_init(cb_col_identity);
        CircularBuffer(neg_unit_cb).reserve_back(sbw);
        tile_regs_acquire();
        copy_tile(cb_col_identity, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_negate_tile, 0, VectorMode::None)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        set_pack(neg_unit_cb);
        configure_single_tile_pack(neg_unit_cb);
        for (uint32_t j = 0; j < sbw; ++j) {
            pack_tile(0, neg_unit_cb);
        }
        tile_regs_release();
        CircularBuffer(neg_unit_cb).push_back(sbw);
        CircularBuffer(neg_unit_cb).wait_front(sbw);
        fused_neg_unit_ready = true;
    }
    }

    // QK matmul setup is kept across back-to-back subblocks of the same height (the m step's replay ends
    // with the counter reset, so it leaves nothing behind), and in STANDARD so is the P pack setup (format,
    // row width, ReLU; LOW_PRECISION's exp ring builds have no room for it); every other step calls leave_qk.
    uint32_t qk_setup_h = 0;
#ifdef SDPA_RECIPE_LOFI
    // A plain statement: lambda calls from the -Os cold steps cost code, and LOW_PRECISION's exp ring builds sit
    // at the kernel config buffer limit.
#define SDPA_FUSED_LEAVE_QK() (qk_setup_h = 0)
#else
    bool qk_pack_on = false;
    // Out of line: it is called from every other step.
    auto leave_qk = [&]() __attribute__((noinline)) {
        qk_setup_h = 0;
        if (qk_pack_on) {
            PACK((llk_pack_relu_config(ReluConfig::none())));
            qk_pack_on = false;
        }
    };
#define SDPA_FUSED_LEAVE_QK() leave_qk()
#endif
    // QK subblock (g, kb): DEST = -m_ref + Q K^T, exp in place, P and its row sums packed.
    auto qk_subblock = [&](uint32_t g, uint32_t kb) {
        const uint32_t h = rows(g);
        const uint32_t row0 = H * g;
        const uint32_t col0 = kb * sbw;
        if (qk_setup_h != h) {
            set_srca(cb_kt_in);
            recipe_mm_reinit(cb_q_in, cb_kt_in, true, sbw, h, DHt);
            qk_setup_h = h;
        }
        tile_regs_acquire();
        uint32_t in0_index = row0 * DHt;
        uint32_t in1_index = col0;
        for (uint32_t inner = 0; inner < DHt; ++inner) {
            matmul_block_no_mop(cb_q_in, cb_kt_in, in0_index, in1_index, 0, true, sbw, h, DHt);
            in0_index++;
            in1_index += KT;
        }
        // - m_ref: M (row r's m in column 0) x (-e0 transposed: -1 in row 0). Both operands are exact in one
        // fidelity phase (m_ref has 7 significant bits for srcB, -1 has one for srcA), so a HiFi build replays
        // its image once here, and only its inner 0-15 half (the rest multiplies zeros).
        UNPACK((llk_unpack_AB_matmul(prev.max, neg_unit_cb, row0, 0, sbw, h, 1)));
#ifdef SDPA_RECIPE_LOFI
        // A LoFi image has no halves; this keeps LOW_PRECISION's matmul instantiations (and code size) as they were.
        constexpr bool m_inner_half = false;
#else
        constexpr bool m_inner_half = true;
#endif
        MATH((llk_math_matmul_no_mop<MATH_FIDELITY, MM_THROTTLE, 1, m_inner_half>(prev.max, neg_unit_cb, 0, sbw, h)));
        tile_regs_commit();
        tile_regs_wait();
#if defined(SDPA_RECIPE_K_PRIMARY_ROWS) || defined(SDPA_RECIPE_RING)
#ifdef SDPA_RECIPE_RING
        if (recipe_k_valid_rows < recipe_k_chunk_rows)
#endif
        {
            mask_recipe_tail(col0, sbw, h);
            qk_setup_h = 0;
        }
#endif
#ifdef SDPA_RECIPE_LOFI
        PACK((llk_pack_relu_config(ReluConfig::zero())));
        for (uint32_t t = 0; t < h * sbw; ++t) {
            exp_packthread_tile<true, false, InputClamping::None, 32>(t, VectorMode::None);
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        set_pack(cb_qkt_im);
        configure_row_pack_width(cb_qkt_im, sbw);
        pack_contiguous_rows_nocfg(cb_qkt_im, row0, h, KT, col0, sbw);
        PACK((llk_pack_reconfig_l1_acc(kb > 0 ? 1 : 0)));
        pack_contiguous_rows_nocfg(psum_cb, row0, h, sbw, 0, sbw);
        PACK((llk_pack_reconfig_l1_acc(0)));
        PACK((llk_pack_relu_config(ReluConfig::none())));
#else
        if (!qk_pack_on) {
            PACK((llk_pack_relu_config(ReluConfig::zero())));
        }
        for (uint32_t t = 0; t < h * sbw; ++t) {
            exp_packthread_tile<true, false, InputClamping::None, 32>(t, VectorMode::None);
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        if (!qk_pack_on) {
            set_pack(cb_qkt_im);
            configure_row_pack_width(cb_qkt_im, sbw);
            PACK((llk_pack_reconfig_l1_acc(0)));
            qk_pack_on = true;
        }
        pack_contiguous_rows_nocfg(cb_qkt_im, row0, h, KT, col0, sbw);
        if (kb > 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        pack_contiguous_rows_nocfg(psum_cb, row0, h, sbw, 0, sbw);
        if (kb > 0) {
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
#endif
        tile_regs_release();
    };

    // Saturation check of groups [g0, g0 + n): max over their chunk row-sum tiles, into the one-tile check CB.
    // STANDARD (math-bound) reduces columns only, LOW_PRECISION (pack-bound) to a scalar; read_check takes the max.
#ifdef SDPA_RECIPE_LOFI
    constexpr ReduceDim check_dim = ReduceDim::REDUCE_SCALAR;
#else
    constexpr ReduceDim check_dim = ReduceDim::REDUCE_COL;
#endif
    // Check unit size (groups per check). STANDARD is math-bound, so halving the checks' fixed cost pays;
    // LOW_PRECISION checks every group.
#ifdef SDPA_RECIPE_LOFI
    constexpr uint32_t CG = 1;
#else
    constexpr uint32_t CG = 2;
#endif
    auto check_groups = [&](uint32_t g0, uint32_t n) {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t last = g0 + n - 1;
        const uint32_t row_end = H * last + rows(last);
        CircularBuffer(cb_qkt_im).wait_front(row_end * KT);  // the groups' sums were packed before their push
        set_srca(psum_cb);
        reduce_init<PoolType::MAX, check_dim>(psum_cb, cb_identity_scale_in, check_cb);
        tile_regs_acquire();
        for (uint32_t t = H * g0 * sbw; t < row_end * sbw; ++t) {
            reduce_tile<PoolType::MAX, check_dim>(psum_cb, cb_identity_scale_in, t, 0, 0);
        }
        tile_regs_commit();
        reduce_uninit();
#ifdef SDPA_RECIPE_LOFI
        // The reduce may replace the matmul replay image: the next matmul init must re-record it.
        recipe_mm_reuse_a = false;
#endif
        CircularBuffer(check_cb).reserve_back(1);
        tile_regs_wait();
        set_pack(check_cb);
        configure_single_tile_pack(check_cb);
        pack_tile(0, check_cb);
        tile_regs_release();
        CircularBuffer(check_cb).push_back(1);
    };
    // The verdict of the oldest issued check: does any of its groups' partial sums reach the redo threshold.
    auto read_check = [&]() -> bool {
        uint32_t redo = 0;
        UNPACK({
            CircularBuffer(check_cb).wait_front(1);
            auto* check =
                reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(check_cb), 0));
            if constexpr (check_dim == ReduceDim::REDUCE_SCALAR) {
                redo = (check[0] & 0x7fffu) >= kFusedRedoSumBf16 ? 1u : 0u;
            } else {
                // Column maxima: row 0 of faces 0 and 1, BF16 pairs at uint32 words 0-7 and 128-135.
                for (uint32_t i = 0; i < 8; ++i) {
                    const uint32_t a = check[i], b = check[128 + i];
                    redo |= ((a & 0x7fffu) >= kFusedRedoSumBf16) | (((a >> 16) & 0x7fffu) >= kFusedRedoSumBf16) |
                            ((b & 0x7fffu) >= kFusedRedoSumBf16) | (((b >> 16) & 0x7fffu) >= kFusedRedoSumBf16);
                }
            }
            CircularBuffer(check_cb).pop_front(1);
            mailbox_write(ckernel::ThreadId::MathThreadId, redo);
            mailbox_write(ckernel::ThreadId::PackThreadId, redo);
        })
        MATH(redo = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(redo = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        return redo != 0;
    };

    // PV of group g over K tiles [k0, k0 + k_len), L1-accumulated into O plane `plane` (0: O, vDHt: a redone
    // group's chunk PV, written by the first piece).
    bool v_ready = false;
    auto pv_piece = [&](uint32_t g, uint32_t k0, uint32_t k_len, uint32_t plane, bool accumulate) {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t h = rows(g);
        const uint32_t row0 = H * g;
#ifdef SDPA_RING_STREAM_STATE
        if (plane == 0) {
            recipe_stream_wait_rows(row0 + h);
        }
#endif
        if (!v_ready) {
            CircularBuffer(cb_v_in).wait_front(KT * vDHt);
            v_ready = true;
        }
        CircularBuffer(cb_qkt_im).wait_front((row0 + h) * KT);
        set_srca(cb_v_in);
        recipe_mm_reinit(cb_qkt_im, cb_v_in, false, pv_w, h, KT);
        set_pack(out_cb);
        PACK((llk_pack_reconfig_l1_acc(accumulate ? 1 : 0)));
        configure_row_pack_width(out_cb, pv_w);
        for (uint32_t v = 0; v < n_pv; ++v) {
            blocked_matmul_and_pack<false, vDHt, vDHt>(
                cb_qkt_im,
                cb_v_in,
                out_cb,
                row0 * KT + k0,
                k0 * vDHt + v * pv_w,
                0,
                v * pv_w + plane,
                pv_w,
                h,
                k_len,
                KT,
                /*skip_pack_configure=*/true);
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
    };

    // l += this chunk's sums for an unchanged group.
    auto fold_sums = [&](uint32_t g) {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t h = rows(g);
        static_assert(H * qkt_subblock_w <= 8 || true);
        set_srca(psum_cb);
        copy_init(psum_cb);
        tile_regs_acquire();
        for (uint32_t t = 0; t < h * sbw; ++t) {
            copy_tile(psum_cb, H * g * sbw + t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        set_pack(cur.sum);
        configure_single_tile_pack(cur.sum);
        PACK((llk_pack_reconfig_l1_acc(1)));
        for (uint32_t i = 0; i < h; ++i) {
            for (uint32_t j = 0; j < sbw; ++j) {
                pack_tile<true>(i * sbw + j, cur.sum, i);
            }
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        tile_regs_release();
    };

    // Publish group g's O and l rows; the last chunk normalizes them.
    auto finish_group = [&](uint32_t g) {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t h = rows(g);
        CircularBuffer(cur.sum).push_back(h);
        CircularBuffer(out_cb).push_back(h * vDHt * sdpa_out_stride);
#ifdef SDPA_RING_STREAM_STATE
        // The last chunk of a saved block: these O rows are final for this ring iteration.
        if (recipe_stream_save_slot != kNoStreamSlot) {
            recipe_stream_save_rows(out_cb, H * g + h);
        }
#endif
        if (is_last_iter) {
            normalize_row_streaming<false, vDHt, dst_size, cb_col_identity, cb_recip_scratch, cb_normalized_out>(
                cur.sum, out_cb, h);
            srca = cb_normalized_out;
            pack_cb = cb_recip_scratch;
            arm_exp();
        }
    };

    // Redo group g on the reduce path (rare): scores, the real max with the theta select, P and sums,
    // its PV into plane 1, then O = O * c + PV and l = l * c + l_chunk row by row.
    auto redo_group = [&](uint32_t g) SDPA_RECIPE_COLD {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t h = rows(g);
        const uint32_t row0 = H * g;
#ifdef SDPA_RING_STREAM_STATE
        recipe_stream_wait_rows(row0 + h);
#endif
        const uint32_t gi = gindex(g);
        set_srca(cb_kt_in);
        recipe_mm_init(cb_q_in, cb_kt_in, true, sbw, h, DHt);
        set_pack(cb_qkt_im);
        configure_row_pack_width(cb_qkt_im, sbw);
        for (uint32_t kb = 0; kb < n_kb; ++kb) {
            blocked_matmul_and_pack<true, KT, KT>(
                cb_q_in, cb_kt_in, cb_qkt_im, row0 * DHt, kb * sbw, gi, kb * sbw, sbw, h, DHt, DHt, true);
        }
        pack_to_unpack_barrier();
        set_srca(cb_qkt_im);
        set_pack(cur.max);
        configure_single_tile_pack(cur.max);
        reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT, scale_fp32>(
            cur.max, prev.max, gi, true, h, KT, false, false, row0);
        pack_to_unpack_barrier();
        for (uint32_t kb = 0; kb < n_kb; ++kb) {
            sub_exp_block_bcast_cols<false, scale_fp32>(
                cb_qkt_im, cur.max, lsum_cb, KT, gi, kb * sbw, h, sbw, false, /*wait_max=*/false);
        }
        pack_cb = cb_qkt_im;
        srca = cb_qkt_im;
        pack_to_unpack_barrier();
        pv_piece(g, 0, KT, vDHt, false);
        pack_to_unpack_barrier();
        const uint32_t read_base = is_last_iter ? 0 : row0;
        for (uint32_t r = 0; r < h; ++r) {
            // c = exp(scale * (m_old - m_new)) for this Q tile row, column-broadcast below.
            set_srca(prev.max);
            sub_init(prev.max, cur.max);
            tile_regs_acquire();
            sub_tiles(prev.max, cur.max, row0 + r, row0 + r, 0);
            tile_regs_commit();
            CircularBuffer(cb_recip_scratch).reserve_back(1);
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_correction, (scale_fp32), 0, VectorMode::C)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            set_pack(cb_recip_scratch);
            configure_single_tile_pack(cb_recip_scratch);
            pack_tile(0, cb_recip_scratch);
            tile_regs_release();
            CircularBuffer(cb_recip_scratch).push_back(1);
            CircularBuffer(cb_recip_scratch).wait_front(1);
            // l = l * c + l_chunk
            tile_regs_acquire();
            set_srca(cur.sum);
            copy_init(cur.sum);
            copy_tile(cur.sum, read_base + r, 0);
            set_srca(lsum_cb);
            copy_init(lsum_cb);
            copy_tile(lsum_cb, row0 + r, 1);
            set_srca(cb_recip_scratch);
            unary_bcast_init<BroadcastType::COL>(cb_recip_scratch);
            unary_bcast<BroadcastType::COL>(cb_recip_scratch, 0, 2);
            unary_bcast_uninit<BroadcastType::COL>(cb_recip_scratch);
            tile_regs_commit();
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_rescale_add, (1), 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            set_pack(cur.sum);
            configure_single_tile_pack(cur.sum);
            pack_tile<true>(0, cur.sum, r);
            tile_regs_release();
            // O = O * c + PV, two head-dim tiles at a time
            for (uint32_t j = 0; j < vDHt; j += 2) {
                const bool pair = j + 1 < vDHt;
                tile_regs_acquire();
                set_srca(out_cb);
                copy_init(out_cb);
                const uint32_t rbase = (read_base + r) * 2 * vDHt + j;
                copy_tile(out_cb, rbase, 0);
                if (pair) {
                    copy_tile(out_cb, rbase + 1, 1);
                }
                copy_tile(out_cb, rbase + vDHt, pair ? 2 : 1);
                if (pair) {
                    copy_tile(out_cb, rbase + vDHt + 1, 3);
                }
                set_srca(cb_recip_scratch);
                unary_bcast_init<BroadcastType::COL>(cb_recip_scratch);
                unary_bcast<BroadcastType::COL>(cb_recip_scratch, 0, pair ? 4 : 2);
                unary_bcast_uninit<BroadcastType::COL>(cb_recip_scratch);
                tile_regs_commit();
                tile_regs_wait();
                if (pair) {
                    PACK((SFPU_UNARY_CALL(
                        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_rescale_add, (2), 0, VectorMode::None)));
                } else {
                    PACK((SFPU_UNARY_CALL(
                        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_rescale_add, (1), 0, VectorMode::None)));
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                set_pack(out_cb);
                configure_single_tile_pack(out_cb);
                const uint32_t wbase = r * 2 * vDHt + j;
                pack_tile<true>(0, out_cb, wbase);
                if (pair) {
                    pack_tile<true>(1, out_cb, wbase + 1);
                }
                tile_regs_release();
            }
            CircularBuffer(cb_recip_scratch).pop_front(1);
        }
        // The new maxima become this group's reference (cur.max was scratch, prev.max is the state).
        set_srca(cur.max);
        copy_init(cur.max);
        tile_regs_acquire();
        for (uint32_t r = 0; r < h; ++r) {
            copy_tile(cur.max, row0 + r, r);
        }
        tile_regs_commit();
        tile_regs_wait();
        set_pack(prev.max);
        configure_single_tile_pack(prev.max);
        for (uint32_t r = 0; r < h; ++r) {
            pack_tile<true>(r, prev.max, row0 + r);
        }
        tile_regs_release();
        pack_to_unpack_barrier();
        arm_exp();
    };

    // Non-last chunks fold every unchanged group's sums once at the end (one pack-format switch); the last
    // chunk folds per group, ahead of each group's normalization.
    const bool late_fold = !is_last_iter;
    uint32_t redone = 0;
    // Groups are checked in units of CG: one reduce, one verdict and one round of handshakes per unit; a unit
    // whose verdict fires is re-checked group by group, so only the groups that need it are redone. Iteration it runs QK of group it, the check of the unit ending at group it - 1 and PV of group
    // it - CG - 1, so a unit's verdicts are read one iteration after its check was issued and before its first
    // group's PV touches O.
    uint32_t unit_redo = 0;  // the current check unit's verdicts, bit j for its group j
    for (uint32_t it = 0; it < G + CG + 1; ++it) {
        const bool do_qk = it < G;
        const uint32_t cg_last = it - 1;
        const bool do_check = it >= 1 && it <= G && ((cg_last + 1) % CG == 0 || cg_last == G - 1);
        bool do_pv = it >= CG + 1;
        const uint32_t pg = it - CG - 1;
        if (do_pv && pg % CG == 0) {
            const uint32_t n = G - pg < CG ? G - pg : CG;
            unit_redo = read_check() ? 1u : 0u;
            if (unit_redo && n > 1) {
                // Rare: find which groups need the redo (redoing a neighbour would change its numerics).
                unit_redo = 0;
                for (uint32_t j = 0; j < n; ++j) {
                    check_groups(pg + j, 1);
                    unit_redo |= (read_check() ? 1u : 0u) << j;
                }
            }
        }
        if (do_pv && ((unit_redo >> (pg % CG)) & 1u)) {
            redo_group(pg);
            finish_group(pg);
            redone |= 1u << pg;
            do_pv = false;
        }
        for (uint32_t piece = 0; piece < n_pieces; ++piece) {
            if (do_qk) {
                for (uint32_t kb = piece * kb_per_piece; kb < (piece + 1) * kb_per_piece; ++kb) {
                    qk_subblock(it, kb);
                }
                if (piece == n_pieces - 1) {
                    cb_push_back_hold_wr_ptr(cb_qkt_im, rows(it) * KT);
                }
            }
            if (do_check && piece == 0) {
                check_groups(cg_last - cg_last % CG, cg_last % CG + 1);
            }
            if (do_pv) {
                pv_piece(pg, piece * piece_k, piece_k, 0, true);
            }
        }
        if (do_pv) {
            if (!late_fold) {
                fold_sums(pg);
            }
            finish_group(pg);
        }
    }
#ifndef SDPA_RECIPE_LOFI
    leave_qk();
#endif
    if (late_fold) {
        // cur.sum has wrapped back to this chunk's first row (one push per row); fold unchanged groups in place.
        set_srca(psum_cb);
        copy_init(psum_cb);
        set_pack(cur.sum);
        configure_single_tile_pack(cur.sum);
        PACK((llk_pack_reconfig_l1_acc(1)));
        for (uint32_t g = 0; g < G; ++g) {
            if ((redone >> g) & 1u) {
                continue;
            }
            const uint32_t h = rows(g);
            tile_regs_acquire();
            for (uint32_t t = 0; t < h * sbw; ++t) {
                copy_tile(psum_cb, H * g * sbw + t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < h; ++i) {
                for (uint32_t j = 0; j < sbw; ++j) {
                    pack_tile<true>(i * sbw + j, cur.sum, H * g + i);
                }
            }
            tile_regs_release();
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
    // K and Q stay until here: a redone group recomputes its scores.
    if (true) {
        CircularBuffer(cb_kt_in).pop_front(DHt * KT);
        if (independent_q_release ? release_q : is_last_iter) {
            sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
        }
    }
    // Chunk sums were read in place; release them as a whole (pointers wrap back to the same slots).
    CircularBuffer(lsum_cb).push_back(Sq_chunk_t);
    CircularBuffer(lsum_cb).wait_front(Sq_chunk_t);
    CircularBuffer(lsum_cb).pop_front(Sq_chunk_t);
    CircularBuffer(psum_cb).push_back(Sq_chunk_t * sbw);
    CircularBuffer(psum_cb).wait_front(Sq_chunk_t * sbw);
    CircularBuffer(psum_cb).pop_front(Sq_chunk_t * sbw);
    CircularBuffer(cb_v_in).pop_front(KT * vDHt);
    CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT);
    // Leave the unpack/pack formats where the step expects them.
    set_srca(cb_qkt_im);
    set_pack(cb_qkt_im);
}
#undef SDPA_FUSED_LEAVE_QK
#endif
