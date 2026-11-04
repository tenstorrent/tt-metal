// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// The op's compute kernel with the key block's score phases STREAMED in q tile-rows and the softmax exp issued from
// the PACK thread (TRISC2), so the SFPU exp of row i runs while the MATH thread (TRISC1) does FPU work of other rows.
// bf16 DEST only; fp32 DEST keeps the op's helper body verbatim (stock exp_tile, different algorithm).
//
// RAW-LLK JUSTIFICATION (capability, not ergonomics). Bypassed helpers: matmul_block (Q.K^T, P.V),
// reduce<MAX/SUM, REDUCE_ROW>, binary_sfpu<BinaryMax>, eltwise_chain(Sub bcast col -> ScaledExp -> PackTile).
//  (1) The exp must run on TRISC2 between tile_regs_wait() and the pack of the SAME DEST half while TRISC1 fills the
//      other half: no helper can place an SFPU op on the pack thread (eltwise_chain runs every op on MATH).
//  (2) The overlap needs several different ops per DEST section packed to different CBs (copy + block max reduce +
//      row sum + P.V in one section -> m_blk, l, O) and Q.K^T / P rows written out of order into a held-wr_ptr score
//      block (pack_tile<true>, blocked pack width); helpers own one phase over one whole block, one output CB.
//  (3) The running max is folded into the row max by pre-loading m_acc (within-face transposed) into the max
//      reduce's DEST accumulator, replacing BinaryMax (an SFPU op on MATH that would race the pack-thread SFPU
//      stream's LREG/LOADMACRO state); the row max uses reduce_block_max_row (one MOP per row; bf16 operands only).
//  The exp itself is ScaledExp's instruction stream unchanged (bit-identical to exp_tile<false, true>), issued for a
//  whole DEST section (sbw tiles) with ONE STALLWAIT(SFPU, MATH) + set_dst_write_addr and ONE RWC reset per section
//  (each tile's stream advances the DEST RWC by exactly one tile). Without any SFPU-side stall the per-tile SETRWC
//  (a math-pipe instruction) races the next tile's SFPU loads: measured INCORRECT (cand_nostall).
//
// Per key block (R rows, skv valid key tiles, sbw = DEST-half column chunk), steps i = -2 .. R:
//   E0  exp chunk 0 of row i: sub (S_i - m_i) on MATH -> exp(s .) on PACK -> pack in place      [overlaps S2]
//   S2  m_blk(i+1) = max(m_acc, rowmax S_{i+1}); l(i-1) = rowsum P_{i-1}; O(i-1) = P_{i-1}.V     [MATH, FPU]
//   E1  exp chunk 1 of row i (a one-section row is exp'd in two halves)                 [overlaps QK]
//   QK  S_{i+2} = Q_{i+2} . K^T (+ key-tail mask via dest-reuse add)                             [MATH, FPU]
// P rows are written in place; a credit-only token CB (cb_p_tok) orders "P row packed" (pack) before "P row
// unpacked" (P.V / rowsum). Score rows are made visible with push-and-hold-wr_ptr (the CB holds exactly one block).
// Output is bit-identical to the op's kernel (all regimes tested).

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/reduce_custom.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ring_joint_sdpa_common.hpp"
#include "ring_joint_sdpa_scaled_exp.hpp"

using namespace compute_kernel_lib;

// Sampled per-block zone window (key blocks of the first unit): the marker budget (~125 zone executions per RISC)
// cannot hold every block.
constexpr uint32_t ZONE_SAMPLE_KB0 = 4;
constexpr uint32_t ZONE_SAMPLE_NKB = 6;

struct ValidKeyTiles {
    uint32_t n;
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t /*block_k*/) const { return n; }
};

inline uint32_t largest_divisor_le(uint32_t n, uint32_t cap) {
    uint32_t w = n < cap ? n : cap;
    while (n % w) {
        --w;
    }
    return w;
}

inline uint32_t subblock_w(uint32_t n) { return largest_divisor_le(n, DEST_AUTO_LIMIT); }

// Push n tiles (visible to UNPACK) but keep the pack write pointer on the block base, so later pack_tile<true>
// indices stay relative to the block (the CB holds exactly one block).
ALWI void push_hold_wr_ptr(uint32_t cb, uint32_t n) {
    cb_push_back(cb, n);
    PACK(({
        auto& intf = get_local_cb_interface(cb);
        intf.fifo_wr_ptr -= n * intf.fifo_page_size;
        const uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;
        }
    }));
}

uint32_t g_pack_width = 1;  // PACK thread: the packer MOP's current tile count
// Re-program the pack MOP for w tiles per pack call (no addrmod / stride reconfig: same formats, MOP-only).
ALWI void set_pack_width(uint32_t cb, uint32_t w) {
    PACK(({
        if (g_pack_width != w) {
            llk_pack_init<PackMode::Default, false, true, true>(cb, w);
            g_pack_width = w;
        }
    }));
}
// Pack n contiguous DEST tiles (from dst0) to n contiguous tiles of cb at out-of-order index idx (relative to the
// reserved write pointer): one blocked pack call (measured faster than n single-tile calls).
ALWI void pack_row(uint32_t cb, uint32_t dst0, uint32_t idx, uint32_t n) {
    set_pack_width(cb, n);
    pack_tile<true>(dst0, cb, idx);
}

// Unpacker source tracker: (srcA cb, srcB cb) currently configured; reconfig only on a format change.
struct Src {
    uint32_t a, b;
    ALWI void use(uint32_t na, uint32_t nb) {
        reconfig_data_format(a, na, b, nb);
        a = na;
        b = nb;
    }
};

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(3);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scores = get_compile_time_arg_val(6);
    constexpr uint32_t cb_max_blk = get_compile_time_arg_val(7);
    constexpr uint32_t cb_max_acc = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sum_blk = get_compile_time_arg_val(9);
    constexpr uint32_t cb_sum_acc = get_compile_time_arg_val(10);
    constexpr uint32_t cb_row_factor = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out_blk = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out_acc = get_compile_time_arg_val(13);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t Sq = get_compile_time_arg_val(15);
    constexpr uint32_t Sk = get_compile_time_arg_val(16);
    constexpr uint32_t DHt = get_compile_time_arg_val(17);
    constexpr uint32_t DVt = get_compile_time_arg_val(18);
    constexpr uint32_t NQ = get_compile_time_arg_val(19);  // unit row space (q_heads_per_unit heads stacked)
    constexpr uint32_t LQ = get_compile_time_arg_val(20);
    constexpr uint32_t n_kv_t = get_compile_time_arg_val(21);
    constexpr uint32_t C = get_compile_time_arg_val(23);
    constexpr uint32_t n_q_t = get_compile_time_arg_val(28);  // per-head q tiles (causal row positions)
    constexpr bool causal = get_compile_time_arg_val(30) != 0;
    constexpr bool balanced = get_compile_time_arg_val(31) != 0;
    constexpr uint32_t G = get_compile_time_arg_val(32);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(33);
    constexpr uint32_t cb_p_tok = get_compile_time_arg_val(34);
    constexpr uint32_t mask_tiles = causal ? 2 : 1;  // causal: [diagonal, all -inf]; else the key-tail mask

    const uint32_t core_idx = get_arg_val<uint32_t>(0);
    const uint32_t num_segments = get_arg_val<uint32_t>(1);
    constexpr uint32_t SEG = 3;  // per segment: valid_tiles, masked, src

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_scores);

    CircularBuffer q_buf(cb_q), k_buf(cb_k), v_buf(cb_v), scores_buf(cb_scores);
    CircularBuffer out_acc_buf(cb_out_acc), out_blk_buf(cb_out_blk);

    const uint32_t tail_idx = 3 + SEG * num_segments;  // has_tail, bh, row0, rows, joint
    const uint32_t W = get_arg_val<uint32_t>(2 + SEG * num_segments);
    const uint32_t my_units = W + get_arg_val<uint32_t>(tail_idx);
    const uint32_t tail_rows = get_arg_val<uint32_t>(tail_idx + 5);
    [[maybe_unused]] const uint32_t my_pos = get_arg_val<uint32_t>(tail_idx + 6);  // group position p

    // ---- per-(row, key tile) mask of the current block: 0 none, else mask tile index + 1 ----
    // causal: the global tile of every compute row (filled per unit) against the block's key tiles (shared plan in
    // ring_joint_sdpa_common.hpp); non-causal: the key-tail padding mask on the segment's last valid tile only.
    uint32_t row_gq[Sq];
    struct BlockMask {
        bool any;          // the block needs any mask tile
        bool pad;          // non-causal: key-tail mask on column skv - 1
        uint32_t src, t0;  // causal: source device, first key tile of the block
    };
    auto mask_of = [&](const BlockMask& bm, uint32_t r, uint32_t c, uint32_t skv) -> uint32_t {
        if constexpr (causal) {
            return causal_mask_class(row_gq[r], global_tile<balanced>(bm.src, bm.t0 + c, n_kv_t, G));
        } else {
            return (bm.pad && c + 1 == skv) ? 1 : 0;
        }
    };
    const uint32_t pv_sbw = subblock_w(DVt);

    // P.V output chunk width inside the S2 section (2 DEST slots go to the max and the row sum).
    constexpr uint32_t PV_W = DEST_AUTO_LIMIT > 2 ? DEST_AUTO_LIMIT - 2 : 1;
    const uint32_t pv_w = largest_divisor_le(DVt, PV_W);

    using PExp = PackThreadScaledExp<scale_bits, false>;

    // ------------------------------------------------------------------------------------------------------------
    // Streamed score phases of one key block (bf16 DEST). Leaves cb_max_blk (R, fronted), cb_sum_{acc|blk} (R),
    // cb_out_{acc|blk} (R*DVt) exactly as the op's phase-by-phase body does; cb_scores is consumed.
    // ------------------------------------------------------------------------------------------------------------
    // m_old / m_new: the running max (previous blocks) and this block's max ping-pong between cb_max_acc and
    // cb_max_blk (both Sq pages, same format) instead of a per-block m_acc <- m_blk copy.
    auto stream_block = [&](uint32_t R, uint32_t skv, const BlockMask& bm, bool first, uint32_t m_old, uint32_t m_new) {
        const uint32_t sbw = subblock_w(skv);
        // Single accumulators: block 0 fills l_acc / O_acc; every later block corrects them in place inside S2
        // (l_acc = rowsum P + a*l_acc, O_acc = P.V + a*O_acc), so there is no l_blk / O_blk and no rescale pass.
        constexpr uint32_t sum_cb = cb_sum_acc;
        constexpr uint32_t out_cb = cb_out_acc;
        Src src{cb_k, cb_q};
        reconfig_data_format(cb_k, cb_q);  // unconditional: the previous block's helpers left any state
        pack_reconfig_data_format(cb_scores);
        PACK((g_pack_width = 0));  // unknown after the previous block's helpers: the first set_pack_width re-programs

        PExp::init();
        cb_wait_front(cb_k, Sk * DHt);
        cb_wait_front(cb_q, Sq * DHt);
        if (!first) {
            cb_wait_front(m_old, R);
        }
        if (bm.any) {
            cb_wait_front(cb_mask, mask_tiles);
        }
        cb_reserve_back(cb_scores, Sq * Sk);

        // ---- S3 chunk: score row r cols [c0, c0+sbw) = Q_r . K^T (+ key-tail mask) -> cb_scores (absolute) ----
        auto qk_chunk = [&](uint32_t r, uint32_t c0) {
            src.use(cb_k, cb_q);
            matmul_block_init(cb_q, cb_k, 1, sbw, 1, DHt);
            tile_regs_acquire();
            for (uint32_t k = 0; k < DHt; ++k) {
                matmul_block(cb_q, cb_k, r * DHt + k, k * Sk + c0, 0, 1, sbw, 1, DHt);
            }
            if (bm.any) {
                bool add_ready = false;
                for (uint32_t j = 0; j < sbw; ++j) {
                    const uint32_t mt = mask_of(bm, r, c0 + j, skv);
                    if (mt == 0) {
                        continue;
                    }
                    if (!add_ready) {
                        src.use(src.a, cb_mask);
                        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask);
                        add_ready = true;
                    }
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask, mt - 1, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_row(cb_scores, 0, r * Sk + c0, sbw);
            tile_regs_release();
            if (c0 + sbw >= skv) {  // row complete
                push_hold_wr_ptr(cb_scores, Sk);
                if (r + 1 == R) {
                    cb_pop_front(cb_k, Sk * DHt);
                    if (R < Sq) {
                        push_hold_wr_ptr(cb_scores, (Sq - R) * Sk);  // nominal credit (tail piece)
                    }
                }
            }
        };

        // ---- S1 chunk: P_i cols [c0, c0+w) = exp(s (S_i - m_i)) in place; exp on the PACK thread ----
        // with_a: the row's correction factor a_i = exp(s (m_acc_i - m_blk_i)) rides in DEST slot w of the same
        // section (same pack-thread exp stream) and is packed to cb_row_factor for S2's in-place l / O update.
        auto exp_chunk = [&](uint32_t i, uint32_t c0, uint32_t w, bool with_a) {
            if (c0 == 0) {
                cb_wait_front(m_new, i + 1);
            }
            tile_regs_acquire();
            if (w) {
                src.use(cb_scores, m_new);
                sub_bcast_cols_init(cb_scores, m_new);
                for (uint32_t j = 0; j < w; ++j) {
                    sub_tiles_bcast_cols(cb_scores, m_new, i * Sk + c0 + j, i, j);
                }
            }
            if (with_a) {
                src.use(m_old, m_new);
                sub_init(m_old, m_new);
                sub_tiles(m_old, m_new, i, i, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            PExp::tiles<p_stall::MATH>(w + (with_a ? 1 : 0));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            if (w) {
                pack_row(cb_scores, 0, i * Sk + c0, w);
            }
            if (with_a) {
                cb_reserve_back(cb_row_factor, 1);
                set_pack_width(cb_row_factor, 1);
                pack_tile(w, cb_row_factor);
                cb_push_back(cb_row_factor, 1);
            }
            tile_regs_release();
            if (w && c0 + w >= skv) {  // P row complete -> token for P.V / rowsum
                cb_reserve_back(cb_p_tok, 1);
                cb_push_back(cb_p_tok, 1);
            }
        };

        // ---- S2: m_blk(mrow) = max(m_acc(mrow), rowmax S_mrow); l(prow) = rowsum P_prow; O(prow) = P_prow . V ----
        auto s2 = [&](uint32_t mrow, bool do_max, uint32_t prow, bool do_pv) {
            tile_regs_acquire();
            if (do_max) {
                cb_wait_front(cb_scores, (mrow + 1) * Sk);
                if (!first) {
                    src.use(m_old, src.b);
                    copy_init(m_old, 0, 1);  // within-face transpose: col 0 -> the GMPOOL row-0 accumulator
                    copy_tile(m_old, mrow, 0);
                }
                src.use(cb_scores, cb_max_scaler);
                // (stock reduce_tile MAX on top of the preloaded m_acc measured INCORRECT: inf output)
                reduce_block_max_row_init_runtime(m_new, skv, cb_scores, cb_max_scaler);
                reduce_block_max_row_runtime(cb_scores, cb_max_scaler, mrow * Sk, 0);
                reduce_block_max_row_uninit_runtime(cb_scores);
            }
            if (do_pv) {
                cb_wait_front(cb_p_tok, 1);
                if (prow == 0) {
                    cb_wait_front(cb_v, Sk * DVt);
                }
                src.use(cb_sum_scaler, cb_scores);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scores, cb_sum_scaler, sum_cb);
                for (uint32_t k = 0; k < skv; ++k) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scores, cb_sum_scaler, prow * Sk + k, 0, 1);
                }
                reduce_uninit();
                src.use(cb_v, cb_scores);
                matmul_block_init(cb_scores, cb_v, 0, pv_w, 1, Sk);
                for (uint32_t k = 0; k < skv; ++k) {
                    matmul_block(cb_scores, cb_v, prow * Sk + k, k * DVt, 2, 0, pv_w, 1, Sk);
                }
                if (!first) {
                    // l and O chunk 0 corrections: ELWMUL accumulates into DEST (DST += A*B), so a*l_acc lands on
                    // the fresh row sum and a*O_acc on the fresh P.V (matmul first: the P.V keeps its own rounding)
                    cb_wait_front(cb_row_factor, prow + 1);
                    cb_wait_front(cb_sum_acc, 1);
                    cb_wait_front(cb_out_acc, pv_w);
                    src.use(cb_sum_acc, cb_row_factor);
                    mul_bcast_cols_init(cb_sum_acc, cb_row_factor);
                    mul_tiles_bcast_cols(cb_sum_acc, cb_row_factor, 0, prow, 1);
                    src.use(cb_out_acc, cb_row_factor);
                    for (uint32_t c = 0; c < pv_w; ++c) {
                        mul_tiles_bcast_cols(cb_out_acc, cb_row_factor, c, prow, 2 + c);
                    }
                    cb_pop_front(cb_sum_acc, 1);
                    cb_pop_front(cb_out_acc, pv_w);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(m_new)));
            set_pack_width(m_new, 1);
            if (do_max) {
                cb_reserve_back(m_new, 1);
                pack_tile(0, m_new);
                cb_push_back(m_new, 1);
            }
            if (do_pv) {
                cb_reserve_back(sum_cb, 1);
                pack_tile(1, sum_cb);
                cb_push_back(sum_cb, 1);
            }
            PACK((llk_pack_reduce_mask_clear()));
            // O_acc rotates through its CB one pv_w chunk at a time (pop the corrected chunk at the front, push it
            // back): in place for a full accumulator, into the free window for a tail piece, order preserved.
            if (do_pv) {
                cb_reserve_back(out_cb, pv_w);
                pack_row(out_cb, 2, 0, pv_w);
                cb_push_back(out_cb, pv_w);
            }
            tile_regs_release();
            if (do_pv) {
                for (uint32_t c0 = pv_w; c0 < DVt; c0 += pv_w) {  // remaining P.V column chunks (DVt > PV_W)
                    tile_regs_acquire();
                    src.use(cb_v, cb_scores);
                    matmul_block_init(cb_scores, cb_v, 0, pv_w, 1, Sk);
                    for (uint32_t k = 0; k < skv; ++k) {
                        matmul_block(cb_scores, cb_v, prow * Sk + k, k * DVt + c0, 0, 0, pv_w, 1, Sk);
                    }
                    if (!first) {
                        cb_wait_front(cb_out_acc, pv_w);
                        src.use(cb_out_acc, cb_row_factor);
                        mul_bcast_cols_init(cb_out_acc, cb_row_factor);
                        for (uint32_t c = 0; c < pv_w; ++c) {
                            mul_tiles_bcast_cols(cb_out_acc, cb_row_factor, c, prow, c);
                        }
                        cb_pop_front(cb_out_acc, pv_w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(out_cb, pv_w);
                    pack_row(out_cb, 0, 0, pv_w);
                    cb_push_back(out_cb, pv_w);
                    tile_regs_release();
                }
                cb_pop_front(cb_p_tok, 1);
                if (prow + 1 == R) {
                    cb_pop_front(cb_v, Sk * DVt);
                }
            }
        };

        // exp chunk width: split a single-section row in two so each half's exp overlaps a different MATH section
        // (S2, then the Q.K^T of row i+2) instead of one long exp against one section (measured -1.2 us per block).
        const uint32_t n_qk = skv / sbw;
        const uint32_t ew = (n_qk == 1 && (sbw % 2) == 0) ? sbw / 2 : sbw;
        const uint32_t n_ex = skv / ew;
        // a_i rides in exp chunk 0's DEST section when a slot is free, else in its own one-tile section
        const bool a_in_chunk = !first && ew < DEST_AUTO_LIMIT;
        const bool a_alone = !first && !a_in_chunk;
        const int32_t Ri = static_cast<int32_t>(R);
        for (int32_t i = -2; i <= Ri; ++i) {
            const bool has_exp = i >= 0 && i < Ri;
            const int32_t m = i + 1, p = i - 1, q = i + 2;
            const bool do_max = m >= 0 && m < Ri;
            const bool do_pv = p >= 0 && p < Ri;
            const bool has_qk = q >= 0 && q < Ri;
            const uint32_t ne = has_exp ? n_ex : 0, nq = has_qk ? n_qk : 0;
            if (ne) {
                exp_chunk(i, 0, ew, a_in_chunk);
                if (a_alone) {
                    exp_chunk(i, 0, 0, true);
                }
            }
            if (do_max || do_pv) {
                s2(do_max ? m : 0, do_max, do_pv ? p : 0, do_pv);
            }
            for (uint32_t t = 1; t < ne || t <= nq; ++t) {
                if (t < ne) {
                    exp_chunk(i, t * ew, ew, false);
                }
                if (t <= nq) {
                    qk_chunk(q, (t - 1) * sbw);
                }
            }
        }
        cb_wait_front(cb_scores, Sq * Sk);
        cb_pop_front(cb_scores, Sq * Sk);
        if (!first) {  // the rescale pass used to consume these
            cb_wait_front(cb_row_factor, R);
            cb_pop_front(cb_row_factor, R);
            cb_pop_front(m_old, R);
        }
        set_pack_width(cb_scores, 1);  // the helpers that follow pack one tile per call
    };

    // fp32-DEST causal mask: scores (row r, cols [c, c + n)) += mask tile `mt` (in place, caller-managed credit).
    [[maybe_unused]] auto apply_mask_run = [&](uint32_t r, uint32_t c, uint32_t n, uint32_t mt) {
        eltwise_chain(
            IterationShape::grid(1, n),
            BinaryFpu<
                BinaryFpuOp::Add,
                input(
                    cb_scores,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Strided),
                input(
                    cb_mask,
                    BroadcastDim::None,
                    WaitPolicy::Upfront,
                    PopPolicy::None,
                    InputTileMapping::Scalar,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Offset)>{StridedTileRange{r * Sk + c, Sk}, mt},
            PackTile<output(
                cb_scores,
                ReservePolicy::None,
                PushPolicy::None,
                DataFormatReconfig::Enabled,
                TileAddressing::Strided)>{StridedTileRange{r * Sk + c, Sk}});
    };

    // One unit's key walk + finalize on R compute rows (Sq for whole units, tail_rows for the tail piece).
    auto run_unit = [&](uint32_t R, uint32_t i) {
        MaybeDeviceZoneScope("compute_unit");
        uint32_t kb = 0;  // processed key blocks of this unit
        uint32_t m_old = cb_max_acc, m_new = cb_max_blk;  // bf16 DEST swaps them per block; fp32 DEST copies
        [[maybe_unused]] uint32_t max_gq = 0, min_gq = 0;
        if constexpr (causal) {
            const AttnUnit unit = i < W ? decode_unit<Sq, NQ, LQ>(core_idx + i * C) : tail_piece(tail_idx);
            max_gq = unit_max_gtile<balanced>(unit, my_pos, n_q_t, G);
            min_gq = max_gq;
            for (uint32_t r = 0; r < R; ++r) {
                row_gq[r] = unit_row_gtile<balanced>(unit, r, my_pos, n_q_t, G);
                min_gq = row_gq[r] < min_gq ? row_gq[r] : min_gq;
            }
        }
        for (uint32_t s = 0; s < num_segments; ++s) {
            uint32_t valid = get_arg_val<uint32_t>(2 + SEG * s);
            const bool seg_masked = get_arg_val<uint32_t>(3 + SEG * s) != 0;
            [[maybe_unused]] const uint32_t seg_src = get_arg_val<uint32_t>(4 + SEG * s);
            if constexpr (causal) {  // same truncation as the reader (keys past the unit's latest query)
                valid = causal_visible_tiles<balanced>(valid, seg_src, max_gq, n_kv_t, G);
            }
            for (uint32_t t0 = 0; t0 < valid; t0 += Sk, ++kb) {
                const uint32_t skv = (valid - t0) < Sk ? (valid - t0) : Sk;
                BlockMask bm{false, false, seg_src, t0};
                if constexpr (causal) {
                    bm.any = global_tile<balanced>(seg_src, t0 + skv - 1, n_kv_t, G) >= min_gq;
                } else {
                    bm.pad = seg_masked && (t0 + skv == valid);
                    bm.any = bm.pad;
                }
                const bool zs = (i == 0) && (kb >= ZONE_SAMPLE_KB0) && (kb < ZONE_SAMPLE_KB0 + ZONE_SAMPLE_NKB);
                if constexpr (!DST_ACCUM_MODE) {
                    // One zone over the whole streamed block: its steps interleave QK / max / exp / rowsum / P.V
                    // rows across the three TRISCs, so per-phase zones no longer measure separable work.
                    MaybeDeviceZoneScopeIf("compute_stream", zs);
                    stream_block(R, skv, bm, kb == 0, m_old, m_new);
                    const uint32_t m_t = m_old;
                    m_old = m_new;
                    m_new = m_t;
                } else {
                    // ---- fp32 DEST: the op's phase-by-phase body (helpers), unchanged ----
                    {
                        MaybeDeviceZoneScopeIf("compute_k_wait", zs);
                        k_buf.wait_front(Sk * DHt);
                    }
                    {
                        MaybeDeviceZoneScopeIf("compute_qk", zs);
                        const uint32_t sbw = subblock_w(skv);
                        matmul_block<
                            true,
                            false,
                            LastBlockTarget::Out,
                            OutputCBLayout::TileRowMajor,
                            matmul_config::InitMode::Short,
                            InputPolicy::WaitAndRetainOnLastBlock,
                            InputPolicy::NoWaitNoPop>(
                            q_buf,
                            k_buf,
                            scores_buf,
                            scores_buf,
                            MatmulBlockShape::of(R, skv / sbw, 1, sbw, DHt, 1),
                            NoPostCompute{},
                            NoPreKBlock{},
                            Sk,
                            Sk);
                    }
                    k_buf.pop_front(Sk * DHt);
                    if (R < Sq) {
                        cb_reserve_back(cb_scores, (Sq - R) * Sk);
                        cb_push_back(cb_scores, (Sq - R) * Sk);
                    }
                    cb_wait_front(cb_scores, Sq * Sk);
                    if (bm.any) {
                        MaybeDeviceZoneScopeIf("compute_mask", zs);
                        if constexpr (causal) {
                            // per row: the diagonal tile (mask tile 0) then the dead run (mask tile 1); the visible
                            // keys of a row are a prefix of the block (positions grow along the source)
                            for (uint32_t r = 0; r < R; ++r) {
                                uint32_t c = 0;
                                while (c < skv && mask_of(bm, r, c, skv) == 0) {
                                    ++c;
                                }
                                if (c < skv && mask_of(bm, r, c, skv) == 1) {
                                    apply_mask_run(r, c, 1, 0);
                                    ++c;
                                }
                                if (c < skv) {
                                    apply_mask_run(r, c, skv - c, 1);
                                }
                            }
                        } else {
                            eltwise_chain(
                                IterationShape::grid(R, 1),
                                BinaryFpu<
                                    BinaryFpuOp::Add,
                                    input(
                                        cb_scores,
                                        WaitPolicy::None,
                                        PopPolicy::None,
                                        InputTileMapping::Block,
                                        DataFormatReconfig::Enabled,
                                        TileAddressing::Strided),
                                    input(cb_mask, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::None)>{
                                    StridedTileRange{skv - 1, Sk}},
                                PackTile<output(
                                    cb_scores,
                                    ReservePolicy::None,
                                    PushPolicy::None,
                                    DataFormatReconfig::Enabled,
                                    TileAddressing::Strided)>{StridedTileRange{skv - 1, Sk}});
                        }
                        cb_pop_front(cb_scores, Sq * Sk);
                        cb_reserve_back(cb_scores, Sq * Sk);
                        cb_push_back(cb_scores, Sq * Sk);
                        cb_wait_front(cb_scores, Sq * Sk);
                    }
                    {
                        MaybeDeviceZoneScopeIf("compute_rowmax", zs);
                        reduce<
                            PoolType::MAX,
                            ReduceDim::REDUCE_ROW,
                            cb_scores,
                            cb_max_scaler,
                            cb_max_blk,
                            ReduceInputPolicy::NoWaitNoPop>(
                            ReduceInputBlockShape::of(R, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                        if (kb > 0) {
                            binary_sfpu<
                                BinaryMax<>,
                                input(cb_max_blk),
                                input(cb_max_acc, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
                                output(cb_max_blk)>(IterationShape::tiles(R));
                        }
                    }
                    {
                        MaybeDeviceZoneScopeIf("compute_sub_exp", zs);
                        eltwise_chain(
                            IterationShape::grid(R, skv),
                            BinaryFpu<
                                BinaryFpuOp::Sub,
                                input(
                                    cb_scores,
                                    WaitPolicy::None,
                                    PopPolicy::None,
                                    InputTileMapping::Block,
                                    DataFormatReconfig::Enabled,
                                    TileAddressing::Strided),
                                input(
                                    cb_max_blk,
                                    BroadcastDim::Col,
                                    WaitPolicy::Upfront,
                                    PopPolicy::None,
                                    InputTileMapping::Col)>{StridedTileRange{0, Sk}},
                            ScaledExp<scale_bits>{},
                            PackTile<output(
                                cb_scores,
                                ReservePolicy::None,
                                PushPolicy::None,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Strided)>{StridedTileRange{0, Sk}});
                        cb_pop_front(cb_scores, Sq * Sk);
                        cb_reserve_back(cb_scores, Sq * Sk);
                        cb_push_back(cb_scores, Sq * Sk);
                        cb_wait_front(cb_scores, Sq * Sk);
                    }
                    {
                        MaybeDeviceZoneScopeIf("compute_rowsum", zs);
                        if (kb == 0) {
                            reduce<
                                PoolType::SUM,
                                ReduceDim::REDUCE_ROW,
                                cb_scores,
                                cb_sum_scaler,
                                cb_sum_acc,
                                ReduceInputPolicy::NoWaitNoPop>(
                                ReduceInputBlockShape::of(R, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                        } else {
                            reduce<
                                PoolType::SUM,
                                ReduceDim::REDUCE_ROW,
                                cb_scores,
                                cb_sum_scaler,
                                cb_sum_blk,
                                ReduceInputPolicy::NoWaitNoPop>(
                                ReduceInputBlockShape::of(R, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                        }
                    }
                    {
                        MaybeDeviceZoneScopeIf("compute_v_wait", zs);
                        v_buf.wait_front(Sk * DVt);
                    }
                    {
                        MaybeDeviceZoneScopeIf("compute_pv", zs);
                        matmul_block<
                            false,
                            false,
                            LastBlockTarget::Out,
                            OutputCBLayout::TileRowMajor,
                            matmul_config::InitMode::Short,
                            InputPolicy::WaitAndPopPerKBlock,
                            InputPolicy::NoWaitNoPop>(
                            scores_buf,
                            v_buf,
                            kb == 0 ? out_acc_buf : out_blk_buf,
                            kb == 0 ? out_acc_buf : out_blk_buf,
                            MatmulBlockShape::of(R, DVt / pv_sbw, 1, pv_sbw, Sk, 1),
                            NoPostCompute{},
                            NoPreKBlock{},
                            DVt,
                            DVt,
                            NoPostKBlock{},
                            ValidKeyTiles{skv});
                    }
                    v_buf.pop_front(Sk * DVt);
                    if (R < Sq) {
                        cb_pop_front(cb_scores, (Sq - R) * Sk);
                    }
                }

                // ---- rescale_block + m_acc <- m_blk (the op's helpers, unchanged) ----
                auto rescale_copy = [&](auto rows_t) {
                    const uint32_t RR = rows_t;
                    if (kb > 0) {  // bf16 DEST corrects l / O in place inside the stream
                        MaybeDeviceZoneScopeIf("compute_rescale", zs);
                        eltwise_chain(
                            IterationShape::tiles(RR),
                            BinaryFpu<
                                BinaryFpuOp::Sub,
                                input(cb_max_acc),
                                input(cb_max_blk, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block)>{},
                            ScaledExp<scale_bits>{},
                            PackTile<output(cb_row_factor)>{});
                        eltwise_chain(
                            IterationShape::tiles(RR),
                            BinaryFpu<
                                BinaryFpuOp::Mul,
                                input(cb_sum_acc),
                                input(
                                    cb_row_factor,
                                    BroadcastDim::Col,
                                    WaitPolicy::Upfront,
                                    PopPolicy::None,
                                    InputTileMapping::Block)>{},
                            DestReuseBinary<BinaryFpuOp::Add, input(cb_sum_blk), DestReuseType::DEST_TO_SRCA>{},
                            PackTile<output(cb_sum_acc)>{});
                        eltwise_chain(
                            IterationShape::grid(RR, DVt),
                            BinaryFpu<
                                BinaryFpuOp::Mul,
                                input(cb_out_acc),
                                input(
                                    cb_row_factor,
                                    BroadcastDim::Col,
                                    WaitPolicy::Upfront,
                                    PopPolicy::AtEnd,
                                    InputTileMapping::Col)>{},
                            DestReuseBinary<BinaryFpuOp::Add, input(cb_out_blk), DestReuseType::DEST_TO_SRCA>{},
                            PackTile<output(cb_out_acc)>{});
                    }
                    MaybeDeviceZoneScopeIf("compute_copy_max", zs);
                    copy<input(cb_max_blk), output(cb_max_acc)>(IterationShape::tiles(RR));
                };
                if constexpr (DST_ACCUM_MODE) {
                    if (R == Sq) {
                        rescale_copy(std::integral_constant<uint32_t, Sq>{});
                    } else {
                        rescale_copy(R);
                    }
                }
            }
        }

        {
            MaybeDeviceZoneScope("compute_finalize");
            unary<Recip<>, input(cb_sum_acc), output(cb_row_factor)>(IterationShape::tiles(R));
            mul<input(cb_out_acc),
                input(cb_row_factor, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                output(cb_out_tiles)>(IterationShape::grid(R, DVt));
        }
        cb_wait_front(m_old, R);
        cb_pop_front(m_old, R);
        cb_pop_front(cb_q, Sq * DHt);
    };

    for (uint32_t i = 0; i < my_units; ++i) {
        run_unit(i < W ? Sq : tail_rows, i);
    }
}
