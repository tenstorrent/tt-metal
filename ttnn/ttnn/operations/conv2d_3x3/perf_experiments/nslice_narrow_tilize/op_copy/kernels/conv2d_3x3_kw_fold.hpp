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
// Q_2[o + d]. The weights are the prepared [K_p, C_out] tensor read tile-permuted by the writer (W'[kh Ct + c, kw N +
// n] is prepared tile ((kh * 3 + kw) Ct + c, n)).
//
// The shift: each Q_kw tile-row is pack-untilized (fp32 rows) into one fp32 accumulator ACC (cb_acc) at row
// rowbase(t) + (1 - kw) d with packer L1 accumulation, so bias, K sum and kw shift all sum in fp32 in the packer.
// ACC leaves a d-row gap after every image row: the wrapped rows of Q_0 (w >= W - d) and Q_2 (w < d) land there and are
// never read (the masks cost nothing). The reader seeds each valid ACC region with the fp32 bias rows compute packs
// once (cb_bias_rm) and hands it over (cb_tok), so no pack ever overwrites. With a full-width subblock (sbw == N) each
// subblock is one packer stream (one close, not one per row). Last, each output tile is tilized from ACC (srcA, tf32:
// the same rounding today's bias add applies to the fp32 partials) and packed with ReLU.
//
// Raw LLK (capability): compute_kernel_lib::matmul_block reads in0 as one contiguous [M, K] block and packs at a CB's
// write pointer; here a tile-row's K' comes from three slab tile-rows, out-of-image kernel rows are skipped, and each
// subblock is pack-untilized at a computed ACC row offset (llk_pack_untilize_impl / a stream MOP with an explicit
// base). compute_kernel_lib::tilize reads at the CB read pointer; the final pass tilizes from computed ACC rows.

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

// ACC row of tile-row index i (tile-row t0 - 1 + i): d leading rows, 32 rows per tile-row, a d-row gap per image row.
template <uint32_t d, uint32_t tpr>
inline uint32_t rowbase(uint32_t t0, uint32_t i) {
    const uint32_t img0 = (t0 - 1 + tpr) / tpr;
    const uint32_t img = (t0 - 1 + i + tpr) / tpr;
    return d + 32 * i + d * (img - img0);
}

#ifdef TRISC_PACK
constexpr uint32_t MOP_LAST_OUTER_CFG_INDEX = 7;
inline uint32_t stream_pacr(uint32_t addr_mod, uint32_t last) {
    return TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        last);
}
// _llk_pack_untilize_mop_config_'s PACRs (2 interfaces = one 32-datum row of one tile, W walks the block's tiles)
// without the per-row close and L1 address bump: a full-width block's rows are consecutive in L1.
template <uint32_t sbw>
inline void stream_mop_config() {
    ckernel_template tmp(16, sbw, TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0), stream_pacr(ADDR_MOD_0, 0));
    tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010));
    tmp.set_last_inner_loop_instr(stream_pacr(ADDR_MOD_1, 0));
    tmp.set_last_outer_loop_instr(stream_pacr(ADDR_MOD_1, 0));
    tmp.program();
}
inline void stream_patch_close(bool close) {
    volatile uint32_t* mop_cfg = reinterpret_cast<volatile uint32_t*>(TENSIX_MOP_CFG_BASE);
    ckernel::mop_sync();
    mop_cfg[MOP_LAST_OUTER_CFG_INDEX] = stream_pacr(ADDR_MOD_1, close ? 1 : 0);
}
// Tile-rows r0 .. r0 + h - 1 of the DEST section into consecutive ACC rows from byte address addr: one stream.
template <uint32_t sbw>
inline void pack_stream(uint32_t addr, uint32_t r0, uint32_t h) {
    program_packer_destination((addr >> 4) - 1);
    for (uint32_t r = r0; r < r0 + h; ++r) {
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, (15 + r * sbw) & 0xF);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        for (uint32_t face = 0; face < 2; ++face) {
            const bool close = r + 1 == r0 + h && face == 1;
            if (close) {
                stream_patch_close(true);
            }
            ckernel_template::run();
            if (close) {
                stream_patch_close(false);
            }
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
            TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
        }
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    }
    set_dst_write_addr(0);
}
#endif

#ifdef TRISC_UNPACK
template <uint32_t cb_acc>
inline void unpack_tilize_at(uint32_t addr, uint32_t col_tile) {
    const uint32_t id = get_operand_id(cb_acc);
    llk_unpack_tilize_impl(
        (addr >> 4) - 1,
        col_tile,
        unpack_src_format[id],
        unpack_dst_format[id],
        get_operand_face_r_dim(id),
        get_operand_num_faces(id),
        get_operand_narrow_tile(id));
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
    uint32_t cb_bias_rm,
    uint32_t cb_tok,
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
    constexpr uint32_t N3 = 3 * N;
    // ACC is N / sbw column planes of [rows, sbw * 32] fp32: a subblock covers the full width of its plane, so every
    // subblock pack is one contiguous stream.
    constexpr uint32_t PITCH = sbw * 32 * 4;
    constexpr uint32_t PLANE = plane_bytes;
    static_assert(N % sbw == 0 && sbw * sbh <= 4 && d >= 1 && d <= 32);

    CircularBuffer in(cb_in);
    CircularBuffer w(cb_w);
    CircularBuffer bias(cb_bias);
    CircularBuffer out(cb_out);
    CircularBuffer bias_rm(cb_bias_rm);
    CircularBuffer tok(cb_tok);
    CircularBuffer acc_ready(cb_acc_ready);

    // ---- bias block: [32, N * 32] fp32 rows = bias (row broadcast in DEST, exact), pack-untilized once.
    compute_kernel_hw_startup(cb_bias, cb_bias_rm);
    {
        MaybeDeviceZoneScope("compute_kwf_bias_block");
        bias.wait_front(N);
        unary_bcast_init<BroadcastType::ROW>(cb_bias);
        pack_reconfig_data_format(cb_bias_rm);
        pack_untilize_dest_init<sbw, sbw>(cb_bias_rm);
        PACK((stream_mop_config<sbw>()));
        bias_rm.reserve_back(1);
        [[maybe_unused]] uint32_t bias_rm_base = 0;
        PACK((bias_rm_base = get_local_cb_interface(get_output_id(cb_bias_rm)).fifo_wr_ptr << 4));
        for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < sbw; ++j) {
                unary_bcast<BroadcastType::ROW>(cb_bias, n0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            // Plane n0 / sbw of the block: [32, sbw * 32] rows, planes back to back.
            PACK((pack_stream<sbw>(bias_rm_base + (n0 / sbw) * 32 * PITCH, 0, 1)));
            tile_regs_release();
        }
        bias_rm.push_back(1);
        pack_untilize_uninit(cb_bias_rm);
    }

    [[maybe_unused]] uint32_t acc_base = 0;
    PACK((acc_base = get_local_cb_interface(get_output_id(cb_acc)).fifo_wr_ptr << 4));
    // M-blocks of bm tile-rows (one ACC, re-seeded per block: the reader waits for the previous block's tokens to be
    // popped, which happens after its final pass). Whole-K keeps the weights resident across blocks (streamed once).
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
            // Whole-K: one pass over K' once every weight K-block landed (fewest packs). K-blocked (weight-paced
            // shapes): one pass per kernel row kh as its weight K-block lands, each pack accumulating onto ACC.
            constexpr uint32_t num_passes = kblocked ? 3 : 1;
            reconfig_data_format(cb_w, cb_in);
            pack_reconfig_data_format(cb_acc);
            pack_untilize_dest_init<sbw, sbw>(cb_acc);
            PACK((stream_mop_config<sbw>()));
            pack_reconfig_l1_acc(1);
            uint32_t init_rows = sbh;
            matmul_block_init(cb_in, cb_w, false, sbw, sbh, Ct);
            uint32_t ready = 0;
            uint32_t landed = 0;
            // Rows [i0, i0 + h) (tile-row t0 - 1 + i), one kw, columns [n0, n0 + sbw).
            uint32_t kh_lo = 0;
            uint32_t kh_hi = 3;
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
                // Q_0 of the last row spills d rows into the next region: it must be seeded first.
                {
                    uint32_t need = i0 + h;
                    need = need < 1 ? 1 : (need > valid ? valid : need);
                    if (need > ready) {
                        tok.wait_front(need);
                        ready = need;
                    }
                }
                tile_regs_acquire();
                for (uint32_t kh = kh_lo; kh < kh_hi; ++kh) {
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
                    const uint32_t in1 = (kblocked ? 0 : kh * Ct * N3) + kw * N + n0;
#ifdef CONV2D_3X3_ABLATE_MATMUL
                    for (uint32_t c = 0; c < 0; ++c) {
#else
                    for (uint32_t c = 0; c < Ct; ++c) {
#endif
                        matmul_block(cb_in, cb_w, in0 + c, in1 + c * N3, a * sbw, false, sbw, b - a, Ct);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                // One stream per run of rows that are consecutive in ACC (an image-row gap splits it).
                for (uint32_t r = 0; r < h;) {
                    uint32_t e = r + 1;
                    while (e < h && rowbase<d, tpr>(t0, i0 + e) == rowbase<d, tpr>(t0, i0 + r) + 32 * (e - r)) {
                        ++e;
                    }
                    PACK((pack_stream<sbw>(
                        acc_base + (n0 / sbw) * PLANE + (rowbase<d, tpr>(t0, i0 + r) + d - kw * d) * PITCH, r, e - r)));
                    r = e;
                }
                tile_regs_release();
            };
            // TRISC code size is a measured lever here: with noinline the subblock code exists once for its three call
            // sites (stem.conv2 45.6 -> 40.6 us, an unrolled kernel-row loop instead: 49.2 us), but the call costs more
            // than it saves on 1-row subblocks and on the K-blocked walk (inst res2.fuse_conv.1 36.2 -> 42.0 us). The
            // descriptor picks it (noinline): whole-K with multi-row subblocks.
            auto subblock_out_of_line = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t n0)
                                            __attribute__((noinline)) { subblock_body(i0, h, kw, n0); };
            auto subblock = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t n0) {
                if constexpr (noinline) {
                    subblock_out_of_line(i0, h, kw, n0);
                } else {
                    subblock_body(i0, h, kw, n0);
                }
            };
            for (uint32_t pass = 0; pass < num_passes; ++pass) {
                if constexpr (kblocked) {
                    kh_lo = pass;
                    kh_hi = pass + 1;
                }
                {
                    // Whole-K: every K-block, popped at the end. K-blocked: this pass's K-block at the front (the
                    // previous one was popped), so its in1 tiles start at index 0.
                    MaybeDeviceZoneScope("compute_kwf_weight_wait");
                    w.wait_front(kblocked ? Ct * N3 : 3 * Ct * N3);
                }
                if (halo_top) {
                    for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
                        subblock(0, 1, 0, n0);
                    }
                }
                for (uint32_t i0 = 1; i0 <= valid; i0 += sbh) {
                    const uint32_t h = valid + 1 - i0 < sbh ? valid + 1 - i0 : sbh;
                    for (uint32_t kw = 0; kw < 3; ++kw) {
                        for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
                            subblock(i0, h, kw, n0);
                        }
                    }
                }
                if (halo_bot) {
                    for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
                        subblock(valid + 1, 1, 2, n0);
                    }
                }
                if constexpr (kblocked) {
                    w.pop_front(Ct * N3);
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

        // ---- output tile = tilize(ACC rows) + ReLU on the pack.
        {
            MaybeDeviceZoneScope("compute_kwf_final");
            acc_ready.wait_front(1);
            [[maybe_unused]] uint32_t acc_rd = 0;
            UNPACK((acc_rd = get_local_cb_interface(get_operand_id(cb_acc)).fifo_rd_ptr << 4));
            reconfig_data_format_srca(cb_acc);
            pack_reconfig_data_format(cb_out);
            if constexpr (relu) {
                pack_relu_config(ReluConfig::zero());
            }
            tilize_init(cb_acc, sbw, cb_out);
            out.reserve_back(bm * N);
            const uint32_t total = valid * N;
            for (uint32_t f0 = 0; f0 < total; f0 += FINAL_BATCH) {
                const uint32_t cnt = total - f0 < FINAL_BATCH ? total - f0 : FINAL_BATCH;
                MATH((llk_math_wait_for_dest_available()));
                PACK((llk_packer_wait_for_math_done()));
                for (uint32_t j = 0; j < cnt; ++j) {
                    const uint32_t u = (f0 + j) / N;
                    const uint32_t n = (f0 + j) - u * N;
                    UNPACK((unpack_tilize_at<cb_acc>(
                        acc_rd + (n / sbw) * PLANE + rowbase<d, tpr>(t0, u + 1) * PITCH, n % sbw)));
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
        tok.wait_front(valid);
        tok.pop_front(valid);
        in.wait_front(slab_pages);
        in.pop_front(slab_pages);
    }
    if constexpr (!kblocked) {
        w.pop_front(3 * Ct * N3);
    }
    bias.pop_front(N);
}

}  // namespace conv2d_3x3::kw_fold
