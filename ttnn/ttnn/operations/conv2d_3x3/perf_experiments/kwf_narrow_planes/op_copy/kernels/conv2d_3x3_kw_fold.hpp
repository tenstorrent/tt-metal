// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// kw-fold + output-shift compute (HEIGHT-sharded TILE input, stride 1, padding == dilation, W % 32 == 0, whole-tile
// C_in, one M-block per core): no untilize, no im2col gather, no tilize.
//
// With W % 32 == 0 a tile-row of 32 output pixels lies in one image row and the vertical tap offset (kh - 1) d W is
// R = d W / 32 whole tile-rows, so for each kw
//     Q_kw[o] = sum_kh in[o + (kh - 1) d W] . Wt[kh, kw]
// is a matmul whose in0 tiles ARE the input's TILE pages (the slab the reader fills: input tile-rows
// [max(0, t0 - 1 - R), ...), consecutive). One matmul with K' = 3 C_in ((kh, c), c fastest) and N' = 3 C_out
// ((kw, n), n fastest) gives [Q_0 | Q_1 | Q_2] at exactly the conv's MAC count (plus one halo tile-row each side whose
// Q_0 / Q_2 spills d rows into this core's range), and out[o] = bias + Q_1[o] + [w >= d] Q_0[o - d] + [w < W - d]
// Q_2[o + d]. The weights stream in column rounds, kw = 1 first, then 0, then 2: round kw is W'[(kh, c), kw, n]
// (3 C_in x C_out tiles, (kh, c, n) order), the prepared [K_p, C_out] tensor read tile-permuted (slot (kh, c, n) is
// prepared tile ((kh * 3 + kw) Ct + c, n)). Each Q_kw tile-row accumulates all its in-image kernel rows in DEST and is
// packed once.
//
// The shift: each Q_kw tile-row is pack-untilized (fp32 rows) into one fp32 accumulator ACC (cb_acc) at row
// rowbase(t) + (1 - kw) d, so the kw shift sums in fp32 in the packer (K in fp32 DEST). ACC leaves a d-row gap after
// every image row: the wrapped rows of Q_0 (w >= W - d) and Q_2 (w < d) land there and are never read (the masks cost
// nothing). Q_1 covers every valid ACC row exactly once and its round comes first, so it packs as an overwrite (packer
// L1 accumulation off) and Q_0 / Q_2, which spill d rows into the neighbouring row regions, accumulate onto it. Gap and
// halo rows take accumulations onto garbage, and nothing reads them. ACC is N planes one output column tile wide
// (128 B rows): row r of a Q tile lands at (32 t + r) * 128 B, exactly PACR emission order, so a run of a subblock's
// tiles down one column is one packer stream with no ADC instruction inside (pack_zst). Last, each output tile is
// tilized from ACC (srcA, tf32)
// plus the bias (srcB) and packed with ReLU. The bias therefore has no reader seed and no wait before the first pack:
// the writer reads it after the weights and the reader decodes it into fp32 rows (cb_bias_rows) once its fill is done.
//
// Raw LLK (capability): compute_kernel_lib::matmul_block reads in0 as one contiguous [M, K] block and packs at a CB's
// write pointer; here a tile-row's K' comes from three slab tile-rows (whole input rows apart, one in0 base per kernel
// row, all accumulated in DEST before the one pack of a column round), out-of-image kernel rows are skipped, and each
// subblock is pack-untilized at a computed ACC row offset (a replayed PACR stream with an explicit base, below).
// compute_kernel_lib::tilize reads at the CB read pointer; the final pass tilizes from computed ACC rows.
// The final pass's bias add (capability): compute_kernel_lib::tilize has no second operand, and an add after it
// (binary_dest_reuse_tiles) needs an unpacker re-init per DEST batch and a second unpack + FPU pass per tile. The
// fp32-DEST tilize copy is already an ELWADD of the tilized srcA and a zeroed srcB, so the tilize MOP's srcB op becomes
// a real unpack of 8 rows of fp32 bias (raw UNPACR at unpacker 1's base address); the math is the stock copy.
// Measured (Blackhole p150b, LoFi, against the reader-seeded accumulator): stem.conv2 40.6 -> 37.6, stem.conv3 60.0 ->
// 53.3 us; the final pass took 3.0 us on stem.conv2 with and without the add.
// The pack stream (raw PACR replay, bypassing llk_pack_untilize's per-row MOP and the 2/4-wide stream MOP it replaced):
// fp32 untilize into a 1-wide plane runs at ~75 cycles per tile isolated (1 core, 4-tile DEST sections, L1
// accumulation on or off) against 100 for a 2-wide stream MOP and 122 for a 1-wide one; in the op (matmul stubbed,
// stem.conv2) 83 against 144 cycles per tile. That lets N = 2 run 4 x 1 subblocks (one unpack context per call; 2 x 2
// needs two): stem.conv2 34.3 -> 32.2 us, where 4 x 1 on the old 1-wide stream MOP was pack-bound (40.4 us). N = 4
// keeps 1 x 4 (as 4 x 1, with four srcB unpacks per 4 tile-MACs instead of one: +2% to +22%).

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace conv2d_3x3::kw_fold {

constexpr uint32_t FINAL_BATCH = 4;  // output tiles per DEST section in the final pass

// Column rounds stream kw = 1 first: Q_1 covers every valid ACC row once, so it packs as an overwrite that Q_0 / Q_2
// accumulate onto (no bias seed). The writer and the weight server read the rounds in this order.
constexpr uint32_t KW_ROUND_ORDER[3] = {1, 0, 2};
constexpr uint32_t kw_round_slot(uint32_t kw) { return kw == 1 ? 0 : (kw == 0 ? 1 : 2); }

// ACC row of tile-row index i (tile-row t0 - 1 + i): d leading rows, 32 rows per tile-row, a d-row gap per image row.
template <uint32_t d, uint32_t tpr>
inline uint32_t rowbase(uint32_t t0, uint32_t i) {
    const uint32_t img0 = (t0 - 1 + tpr) / tpr;
    const uint32_t img = (t0 - 1 + i + tpr) / tpr;
    return d + 32 * i + d * (img - img0);
}

#ifdef TRISC_PACK
// One fp32 untilize stream down a 1-wide plane: 15 PACRs step y (ADDR_MOD_1, y += 1, from the untilize init) and the
// 16th clears y and steps z (ADDR_MOD_2). The untilize init's Z stride is two faces, so z walks F0|F1, F2|F3 of tile W,
// then tile W + 1 at z = 2: a run of tiles is replays only, and only its last PACR closes the stream.
constexpr uint32_t ZST_REPLAY = ckernel::packer::replay_buf_offset;
inline void zst_init() {
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_2);
    load_replay_buf(ZST_REPLAY, 16, [] {
        for (uint32_t i = 0; i < 15; ++i) {
            TTI_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_STRIDED_MODE,
                ADDR_MOD_1,
                p_pacr::ADDR_CNT_CTXT_0,
                0,
                p_pacr::TWO_INTFS_ACTIVE,
                0,
                0,
                p_pacr::NO_CTXT_CTRL,
                0,
                0);
        }
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::TWO_INTFS_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0);
    });
}
// DEST tiles t0 .. t0 + g - 1 into consecutive ACC rows from byte address addr: one stream. Exit state: ch0 X / Y / Z
// and W at 0 (the BH tilize pack of the final pass does not reset Y at tile start).
inline void pack_zst(uint32_t addr, uint32_t t0, uint32_t g) {
    program_packer_destination((addr >> 4) - 1);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, t0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
        lltt::replay(ZST_REPLAY, 16);
    }
    lltt::replay(ZST_REPLAY, 15);
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_2,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    set_dst_write_addr(0);
}
#endif

#ifdef TRISC_UNPACK
// The tilize MOP (_llk_unpack_tilize_mop_config_, non-8-bit) with its srcB op a real unpack instead of the zeroed srcB
// the fp32-DEST copy (ELWADD, srcB rows 0-7 for every 8-row step) adds. The BH tilize unpack leaves tile row r in srcA
// rows 2r (columns 0-15) and 2r + 1 (16-31), so 8 srcB rows of the row-major bias row [b_0..b_31] (4 rows of it) add
// the bias to every row.
inline void tilize_bias_mop_config() {
    static constexpr uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint32_t unpack_srcb =
        TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    ckernel_template tmp(1, 1, unpack_srcb);
    tmp.set_start_op(unpack_srca);
    tmp.program();
    TTI_SETADCXX(p_setadc::UNP_B, 8 * FACE_C_DIM - 1, 0x0);
}
// _llk_unpack_tilize_ (non-8-bit, srcA) for column tile col_tile of the rows at addr, plus srcB from addr_b.
template <uint32_t cb_acc>
inline void unpack_tilize_bias_at(uint32_t addr, uint32_t col_tile, uint32_t addr_b) {
    const uint32_t id = get_operand_id(cb_acc);
    const uint32_t a = (addr >> 4) - 1 + (SCALE_DATUM_SIZE(unpack_src_format[id], col_tile) << 1);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(a, (addr_b >> 4) - 1, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel::ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

// Slab row of input tile-row r read for kernel row kh is r + base[kh]: the reader's tile_slab_plan
// (conv2d_3x3_slab_dataflow.hpp; that header is dataflow-only) for output pixels [o_first, o_end), stride 1,
// padding == dilation, band_last_tap 2. Bands that overlap or touch share consecutive rows; a band past a gap (a short
// ragged last core) is packed right after the previous one.
template <int32_t W, int32_t num_in_px, int32_t d>
inline void slab_bases(uint32_t o_first, uint32_t o_end, int32_t (&base)[3]) {
    const int32_t f = static_cast<int32_t>(o_first) - d * W - d;
    const int32_t band_len = static_cast<int32_t>(o_end - 1 - o_first) + 2 * d + 1;
    int32_t next_free = 0;
    bool have_prev = false;
    int32_t prev_t_lo = 0, prev_t_hi = 0, prev_row = 0;
    for (int32_t kh = 0; kh < 3; ++kh) {
        const int32_t v_lo = f + kh * d * W;
        const int32_t lo = v_lo > 0 ? v_lo : 0;
        const int32_t hi = v_lo + band_len < num_in_px ? v_lo + band_len : num_in_px;
        if (lo >= hi) {
            base[kh] = 0;
            continue;
        }
        const int32_t t_lo = lo / 32;
        const int32_t t_hi = (hi + 31) / 32;
        const bool shares = have_prev && t_lo < prev_t_hi;
        const int32_t row = shares ? prev_row + (t_lo - prev_t_lo) : next_free;
        base[kh] = row - t_lo;
        next_free = row + (t_hi - t_lo);
        have_prev = true;
        prev_t_lo = t_lo;
        prev_t_hi = t_hi;
        prev_row = row;
    }
}

template <
    uint32_t cb_in,
    uint32_t cb_w,
    uint32_t cb_bias,
    uint32_t cb_out,
    uint32_t cb_acc,
    uint32_t cb_bias_rows,
    uint32_t cb_acc_ready,
    uint32_t N,
    uint32_t Ct,
    uint32_t bm,
    uint32_t d,
    uint32_t tpr,
    uint32_t sbw,
    uint32_t sbh,
    bool relu,
    uint32_t slab_pages,
    uint32_t m_tiles,
    bool kblocked,
    uint32_t plane_bytes,
    bool noinline>
ALWI void run(uint32_t m_tile_start, uint32_t actual_m_tiles, uint32_t num_blocks) {
    constexpr uint32_t R = d * tpr;
    // ACC is N column planes of [rows, 32] fp32 (one output column tile each). Subblocks are sbh x 1 (a column run:
    // one stream per run of ACC-consecutive rows) or 1 x sbw (one 1-tile stream per column).
    constexpr uint32_t PITCH = 32 * 4;
    constexpr uint32_t PLANE = plane_bytes;
    static_assert(N % sbw == 0 && sbw * sbh <= 4 && (sbw == 1 || sbh == 1) && d >= 1 && d <= 32);

    CircularBuffer in(cb_in);
    CircularBuffer w(cb_w);
    CircularBuffer out(cb_out);
    CircularBuffer bias_rows(cb_bias_rows);
    CircularBuffer acc_ready(cb_acc_ready);

    compute_kernel_hw_startup(cb_w, cb_in, cb_acc);
    [[maybe_unused]] uint32_t acc_base = 0;
    PACK((acc_base = get_local_cb_interface(get_output_id(cb_acc)).fifo_wr_ptr << 4));
    // M-blocks of bm tile-rows (one ACC: a block's Q_1 overwrite packs follow the previous block's final pass on the
    // same thread). Whole-K keeps the weights resident across blocks (streamed once).
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t t0 = m_tile_start + blk * bm;
        const uint32_t valid = actual_m_tiles - blk * bm < bm ? actual_m_tiles - blk * bm : bm;
        int32_t base[3];
        slab_bases<static_cast<int32_t>(tpr * 32), static_cast<int32_t>(m_tiles * 32), static_cast<int32_t>(d)>(
            t0 * 32, (t0 + valid) * 32, base);
        const bool halo_top = t0 % tpr != 0;
        const bool halo_bot = (t0 + valid) % tpr != 0 && t0 + valid < m_tiles;
        {
            MaybeDeviceZoneScope("compute_kwf_matmul_shift");
            // One pass per weight column round, in round order kw = 1, 0, 2, as the round lands: Q_kw's rows each
            // accumulate every in-image kernel row in DEST and are packed onto ACC once (Q_1: the valid rows, as an
            // overwrite; Q_0: tile-rows t0 - 1 .. t_end - 1 and Q_2: t0 .. t_end, accumulating). K-blocked: one round
            // at a time (double-buffered, popped after its pass). Whole-K: the rounds stay resident across M-blocks;
            // pass s waits only for rounds 0 .. s.
            constexpr uint32_t round_tiles = 3 * Ct * N;
            reconfig_data_format(cb_w, cb_in);
            pack_reconfig_data_format(cb_acc);
            pack_untilize_dest_init<1, 1>(cb_acc);
            PACK((zst_init()));
            uint32_t init_rows = sbh;
            matmul_block_init(cb_in, cb_w, false, sbw, sbh, Ct);
            uint32_t landed = 0;
            // Rows [i0, i0 + h) (tile-row t0 - 1 + i), one kw, columns [n0, n0 + sbw).
            auto subblock_body = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t n0) {
                const uint32_t t_first = t0 - 1 + i0;
                // Slab pages up to the last in0 row this subblock reads (the highest in-image kernel row), clamped to
                // the slab.
                {
                    const uint32_t last_kh = t_first + h - 1 + R < m_tiles ? 2 : 1;
                    uint32_t last = t_first + h - 1 + last_kh * R - R;
                    last = last < m_tiles ? last : m_tiles - 1;
                    uint32_t pages = (static_cast<uint32_t>(static_cast<int32_t>(last) + base[last_kh]) + 1) * Ct;
                    pages = pages < slab_pages ? pages : slab_pages;
                    if (pages > landed) {
                        in.wait_front(pages);
                        landed = pages;
                    }
                }
                tile_regs_acquire();
                for (uint32_t kh = 0; kh < 3; ++kh) {
                    // Rows of the group whose kernel row kh is inside the image: a contiguous [a, b).
                    uint32_t a = 0;
                    uint32_t b = h;
                    while (a < b && t_first + a + kh * R < R) {
                        ++a;
                    }
                    while (b > a && t_first + (b - 1) + kh * R >= m_tiles + R) {
                        --b;
                    }
                    if (a == b) {
                        continue;
                    }
                    if (b - a != init_rows) {
                        matmul_block_init(cb_in, cb_w, false, sbw, b - a, Ct);
                        init_rows = b - a;
                    }
                    const uint32_t in0 =
                        static_cast<uint32_t>(static_cast<int32_t>(t_first + a + kh * R - R) + base[kh]) * Ct;
                    // Whole-K: round kw sits at slot kw_round_slot(kw) of the resident rounds.
                    const uint32_t in1 = (kblocked ? 0 : kw_round_slot(kw) * round_tiles) + kh * Ct * N + n0;
                    for (uint32_t c = 0; c < Ct; ++c) {
                        matmul_block(cb_in, cb_w, in0 + c, in1 + c * N, a * sbw, false, sbw, b - a, Ct);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                // One stream per run of rows that are consecutive in ACC (an image-row gap splits it); a 1 x sbw
                // subblock packs each of its tiles as its own stream into its column's plane.
                for (uint32_t r = 0; r < h;) {
                    uint32_t e = r + 1;
                    while (e < h && rowbase<d, tpr>(t0, i0 + e) == rowbase<d, tpr>(t0, i0 + r) + 32 * (e - r)) {
                        ++e;
                    }
                    [[maybe_unused]] const uint32_t row = acc_base + (rowbase<d, tpr>(t0, i0 + r) + d - kw * d) * PITCH;
                    for (uint32_t j = 0; j < sbw; ++j) {
                        PACK((pack_zst(row + (n0 + j) * PLANE, r * sbw + j, sbw == 1 ? e - r : 1)));
                    }
                    r = e;
                }
                tile_regs_release();
            };
            // TRISC code size is a measured lever here: with noinline the subblock code exists once (stem.conv2 45.6
            // -> 40.6 us, an unrolled kernel-row loop instead: 49.2 us), but on 1-row subblocks the call cost more than
            // it saved before column rounds (inst res2.fuse_conv.1 36.2 -> 42.0 us) and is flat with them (28.1 vs
            // 28.3 us). The descriptor picks it (noinline): whole-K with multi-row subblocks.
            auto subblock_out_of_line = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t n0)
                                            __attribute__((noinline)) { subblock_body(i0, h, kw, n0); };
            auto subblock = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t n0) {
                if constexpr (noinline) {
                    subblock_out_of_line(i0, h, kw, n0);
                } else {
                    subblock_body(i0, h, kw, n0);
                }
            };
            for (uint32_t step = 0; step < 3; ++step) {
                const uint32_t kw = KW_ROUND_ORDER[step];
                {
                    // K-blocked: this round at the front (the previous one was popped), so its in1 tiles start at
                    // index 0. Whole-K: rounds 0 .. step.
                    MaybeDeviceZoneScope("compute_kwf_weight_wait");
                    w.wait_front(kblocked ? round_tiles : (step + 1) * round_tiles);
                }
                // Q_1 (step 0) overwrites; Q_0 and Q_2 accumulate onto it.
                if (step < 2) {
                    pack_reconfig_l1_acc(step);
                }
                // Q_0 reaches back to the halo tile-row above (t0 - 1), Q_2 to the one below (t_end).
                const uint32_t i_lo = kw == 0 && halo_top ? 0 : 1;
                const uint32_t i_end = valid + (kw == 2 && halo_bot ? 2 : 1);
                for (uint32_t i0 = i_lo; i0 < i_end; i0 += sbh) {
                    const uint32_t h = i_end - i0 < sbh ? i_end - i0 : sbh;
                    for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
                        subblock(i0, h, kw, n0);
                    }
                }
                if constexpr (kblocked) {
                    w.pop_front(round_tiles);
                }
            }
            pack_untilize_uninit(cb_acc);
            pack_reconfig_l1_acc(0);
            // The final pass unpacks ACC: its rows must have landed. A CB push from the pack thread is stored only
            // after the packer drained (llk_push_to_brisc), so the unpacker's wait below orders its reads after every
            // ACC write.
            acc_ready.reserve_back(1);
            acc_ready.push_back(1);
        }

        // ---- output tile = tilize(ACC rows) + bias, ReLU on the pack.
        {
            MaybeDeviceZoneScope("compute_kwf_final");
            acc_ready.wait_front(1);
            [[maybe_unused]] uint32_t acc_rd = 0;
            [[maybe_unused]] uint32_t rows_rd = 0;
            UNPACK((acc_rd = get_local_cb_interface(get_operand_id(cb_acc)).fifo_rd_ptr << 4));
            UNPACK((rows_rd = get_local_cb_interface(get_operand_id(cb_bias_rows)).fifo_rd_ptr << 4));
            // The reader's fp32 bias rows (one 512 B page per output column tile: 4 copies of [b_0 .. b_31]).
            bias_rows.wait_front(N);
            reconfig_data_format(cb_acc, cb_bias_rows);
            pack_reconfig_data_format(cb_out);
            if constexpr (relu) {
                pack_relu_config(ReluConfig::zero());
            }
            tilize_init(cb_acc, 1, cb_out);
            UNPACK((tilize_bias_mop_config()));
            constexpr uint32_t ROWS_BYTES = 4 * 32 * 4;  // one column tile's 4 fp32 bias rows
            out.reserve_back(bm * N);
            const uint32_t total = valid * N;
            for (uint32_t f0 = 0; f0 < total; f0 += FINAL_BATCH) {
                const uint32_t cnt = total - f0 < FINAL_BATCH ? total - f0 : FINAL_BATCH;
                MATH((llk_math_wait_for_dest_available()));
                PACK((llk_packer_wait_for_math_done()));
                for (uint32_t j = 0; j < cnt; ++j) {
                    const uint32_t u = (f0 + j) / N;
                    const uint32_t n = (f0 + j) - u * N;
                    UNPACK((unpack_tilize_bias_at<cb_acc>(
                        acc_rd + n * PLANE + rowbase<d, tpr>(t0, u + 1) * PITCH, 0, rows_rd + n * ROWS_BYTES)));
                    MATH((llk_math_eltwise_unary_datacopy<
                          DataCopyType::A2D,
                          DST_ACCUM_MODE,
                          BroadcastType::NONE,
                          UnpackToDestEn>(j, cb_acc)));
                }
                for (uint32_t j = 0; j < cnt; ++j) {
                    PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(j, cb_out, f0 + j)));
                }
                MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
            }
            out.push_back(bm * N);
            acc_ready.pop_front(1);
            tilize_uninit(cb_acc, cb_out);
            if constexpr (relu) {
                pack_relu_config(ReluConfig::none());
            }
        }
        in.wait_front(slab_pages);
        in.pop_front(slab_pages);
    }
    if constexpr (!kblocked) {
        w.pop_front(9 * Ct * N);
    }
    bias_rows.pop_front(N);
}

}  // namespace conv2d_3x3::kw_fold
