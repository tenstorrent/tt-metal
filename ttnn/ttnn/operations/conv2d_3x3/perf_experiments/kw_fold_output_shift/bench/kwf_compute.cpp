// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// kw-fold + output-shift conv2d_3x3 compute (isolated per-core bench; stride 1, p == d, W % 32 == 0, TILE input).
//
// With W % 32 == 0 a tile-row of 32 output pixels lies in one image row, and the vertical tap offset (kh - 1) * d * W
// is R = d * W / 32 whole tile-rows. So for each kw the partial
//     Q_kw[o] = sum_kh in[o + (kh - 1) d W] . Wt[kh, kw]
// is a plain matmul whose in0 tiles ARE the input's TILE pages (slab tile-row i + kh * R): no untilize, no im2col
// gather, no tilize. One matmul with K' = 3 C_in ((kh, c), c fastest) and N' = 3 C_out ((kw, n), n fastest) yields
// [Q_0 | Q_1 | Q_2] with exactly the conv's MAC count, plus one halo tile-row on each side whose Q_0 (top) / Q_2
// (bottom) spills d rows into this core's range. Then
//     out[o] = bias + Q_1[o] + [w(o) >= d] Q_0[o - d] + [w(o) < W - d] Q_2[o + d].
// The shift: every Q_kw tile-row is pack-untilized (fp32, row-major) into one fp32 accumulator ACC at row
// rowbase(t) + (1 - kw) * d, with packer L1 accumulation on, so the shifted sum is formed in fp32 by the packer.
// The masks are free: ACC leaves a gap of d rows after every image row, so the wrapped rows of Q_0 (w >= W - d) and
// Q_2 (w < d) land in a gap that is never read. ACC's valid regions are seeded with the bias (fp32 rows, written by
// the reader from a bias block compute packs once), so bias, the K sum and the kw shift all sum in fp32. Last, each
// output tile is tilized from ACC straight into DEST (Float32 unpack-to-dest: lossless) and packed to the output
// format with ReLU.
//
// Raw LLK (capability, no helper covers it): compute_kernel_lib::matmul_block reads in0 as one contiguous [M, K]
// block and packs to a CB's write pointer; here in0 rows of one output tile-row come from three slab tile-rows, and
// each subblock row is pack-untilized at a computed L1 row offset (llk_pack_untilize_impl with an explicit base).
// compute_kernel_lib::tilize reads at the CB read pointer; the final pass tilizes from computed ACC addresses.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

constexpr uint32_t cb_in = 0;       // slab: input tile-rows [t0 - 1 - R, t0 + bm + 1 + R), zero tiles outside the image
constexpr uint32_t cb_w = 1;        // weights [3 C_in, 3 C_out] tiles, k-major / n-fastest
constexpr uint32_t cb_bias = 2;     // bias tile row 0
constexpr uint32_t cb_out = 3;      // output shard [bm, N] tiles
constexpr uint32_t cb_acc = 4;      // fp32 row-major accumulator (scratch, addressed directly)
constexpr uint32_t cb_bias_rm = 5;  // one page: [32, N * 32] fp32 rows of the bias (compute -> reader)
constexpr uint32_t cb_tok = 6;      // one page per seeded ACC region (reader -> compute)
constexpr uint32_t cb_acc_ready = 7;  // ACC complete (compute pack -> compute unpack)

constexpr uint32_t N = get_compile_time_arg_val(0);    // C_out tiles
constexpr uint32_t Ct = get_compile_time_arg_val(1);   // C_in tiles
constexpr uint32_t bm = get_compile_time_arg_val(2);   // output tile-rows on this core
constexpr uint32_t d = get_compile_time_arg_val(3);    // dilation (= padding)
constexpr uint32_t tpr = get_compile_time_arg_val(4);  // tile-rows per image row (W / 32)
constexpr uint32_t sbw = get_compile_time_arg_val(5);  // subblock width (divides N)
constexpr uint32_t sbh = get_compile_time_arg_val(6);  // subblock height (sbw * sbh <= 4)
constexpr bool relu = get_compile_time_arg_val(7) != 0;
constexpr uint32_t FINAL_BATCH = get_compile_time_arg_val(8);  // output tiles per DEST section in the final pass
// Contiguous-stream pack (sbw == N: a subblock row covers a whole ACC row, so consecutive ACC rows are consecutive L1).
constexpr bool STREAM = get_compile_time_arg_val(9) != 0;
constexpr uint32_t R = d * tpr;  // slab tile-rows per kh step
constexpr uint32_t N3 = 3 * N;
constexpr uint32_t PITCH = N * 32 * 4;  // ACC row bytes (fp32)
static_assert(N % sbw == 0 && sbw * sbh <= 4 && d >= 1 && d <= 32);

// ACC row of output tile-row index i (tile-row t = t0 - 1 + i): d leading rows, then 32 rows per tile-row plus a gap
// of d rows after every image row.
inline uint32_t rowbase(uint32_t t0, uint32_t i) {
    const uint32_t img0 = (t0 - 1 + tpr) / tpr;  // image row of t0 - 1, + 1 (t0 may be 0)
    const uint32_t img = (t0 - 1 + i + tpr) / tpr;
    return d + 32 * i + d * (img - img0);
}

#ifdef TRISC_PACK
inline void pack_row_at(uint32_t addr, uint32_t dst_tile) {
    const uint32_t id = get_output_id(cb_acc);
    llk_pack_untilize_impl<sbw, N>(
        1,
        (addr >> 4) - 1,
        pack_src_format[id],
        pack_dst_format[id],
        0,
        get_output_face_r_dim(id),
        get_output_num_faces(id),
        0,
        dst_tile);
}
#endif

#ifdef TRISC_PACK
// Stream variant of _llk_pack_untilize_mop_config_ for a full-width block: the same PACRs (2 interfaces = one 32-datum
// row of one tile, W counter walks the block's tiles), but no Last=1 close and no L1 address bump per row: the rows
// of a full-width block are consecutive in L1, so a whole subblock is ONE packer stream (one drain instead of 32 per
// tile-row). The run that ends the stream gets a closing last-outer PACR (patched in the MOP config).
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
// h tile-rows of the section (DEST tiles r * sbw ..) into consecutive ACC rows from byte address addr.
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

}  // namespace

void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const bool halo_top = get_arg_val<uint32_t>(1) != 0;  // t0 not at an image-row start: Q_0 of t0 - 1 spills in
    const bool halo_bot = get_arg_val<uint32_t>(2) != 0;  // t0 + bm not at an image-row start: Q_2 of it spills in

    CircularBuffer bias_rm(cb_bias_rm);
    CircularBuffer tok(cb_tok);
    CircularBuffer acc_ready(cb_acc_ready);

    // ---- bias block: [32, N * 32] fp32 rows = bias, broadcast in DEST (exact), pack-untilized once.
    compute_kernel_hw_startup(cb_bias, cb_bias_rm);
    {
        MaybeDeviceZoneScope("kwf_bias_block");
        unary_bcast_init<BroadcastType::ROW>(cb_bias);
        pack_reconfig_data_format(cb_bias_rm);
        pack_untilize_dest_init<sbw, N>(cb_bias_rm);
        bias_rm.reserve_back(1);
        for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < sbw; ++j) {
                unary_bcast<BroadcastType::ROW>(cb_bias, n0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<sbw, N>(cb_bias_rm, 1, n0 / sbw);
            tile_regs_release();
        }
        bias_rm.push_back(1);
    }

    // ---- matmul [Q_0 | Q_1 | Q_2] per tile-row, each Q_kw tile-row pack-untilized with L1 accumulation at
    // rowbase + (1 - kw) d.
    [[maybe_unused]] uint32_t acc_base = 0;
    PACK((acc_base = get_local_cb_interface(get_output_id(cb_acc)).fifo_wr_ptr << 4));
    {
        MaybeDeviceZoneScope("kwf_matmul_shift");
        reconfig_data_format(cb_w, cb_in);
        pack_reconfig_data_format(cb_acc);
        pack_untilize_dest_init<sbw, N>(cb_acc);
        if constexpr (STREAM) {
            PACK((stream_mop_config()));
        }
        pack_reconfig_l1_acc(1);
        uint32_t init_rows = sbh;
        matmul_block_init(cb_in, cb_w, false, sbw, sbh, Ct);
        uint32_t ready = 0;
        // Rows [i0, i0 + h) (index i = t - t0 + 1), one kw, columns [n0, n0 + sbw).
        auto subblock = [&](uint32_t i0, uint32_t h, uint32_t kw, uint32_t n0) {
            if (h != init_rows) {
                matmul_block_init(cb_in, cb_w, false, sbw, h, Ct);
                init_rows = h;
            }
            // Q_0 of the last row spills d rows into the next region: it must be seeded first.
            uint32_t need = i0 + h;
            need = need < 1 ? 1 : (need > bm ? bm : need);
            if (need > ready) {
                tok.wait_front(need);
                ready = need;
            }
            tile_regs_acquire();
            for (uint32_t kh = 0; kh < 3; ++kh) {
                const uint32_t in0 = (i0 + kh * R) * Ct;
                const uint32_t in1 = kh * Ct * N3 + kw * N + n0;
                for (uint32_t c = 0; c < Ct; ++c) {
                    matmul_block(cb_in, cb_w, in0 + c, in1 + c * N3, 0, false, sbw, h, Ct);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (STREAM) {
                // One stream per run of rows that are consecutive in ACC (an image-row gap splits it).
                for (uint32_t r = 0; r < h;) {
                    uint32_t e = r + 1;
                    while (e < h && rowbase(t0, i0 + e) == rowbase(t0, i0 + r) + 32 * (e - r)) {
                        ++e;
                    }
                    PACK((pack_stream(acc_base + (rowbase(t0, i0 + r) + d - kw * d) * PITCH, r, e - r)));
                    r = e;
                }
            } else {
                for (uint32_t r = 0; r < h; ++r) {
                    PACK((pack_row_at(acc_base + (rowbase(t0, i0 + r) + d - kw * d) * PITCH + n0 * 128, r * sbw)));
                }
            }
            tile_regs_release();
        };
        if (halo_top) {
            for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
                subblock(0, 1, 0, n0);
            }
        }
        for (uint32_t i0 = 1; i0 <= bm; i0 += sbh) {
            const uint32_t h = bm + 1 - i0 < sbh ? bm + 1 - i0 : sbh;
            for (uint32_t kw = 0; kw < 3; ++kw) {
                for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
                    subblock(i0, h, kw, n0);
                }
            }
        }
        if (halo_bot) {
            for (uint32_t n0 = 0; n0 < N; n0 += sbw) {
                subblock(bm + 1, 1, 2, n0);
            }
        }
        pack_untilize_uninit(cb_acc);
        pack_reconfig_l1_acc(0);
        // The final pass unpacks ACC only after every pack into it landed (a pack-side push is stored after the
        // packer drained).
        acc_ready.reserve_back(1);
        acc_ready.push_back(1);
    }

    // ---- out tile = tilize(ACC rows) (fp32 unpack-to-dest, lossless), ReLU on the pack.
    {
        MaybeDeviceZoneScope("kwf_final");
        acc_ready.wait_front(1);
        [[maybe_unused]] uint32_t acc_rd = 0;
        UNPACK((acc_rd = get_local_cb_interface(get_operand_id(cb_acc)).fifo_rd_ptr << 4));
        reconfig_data_format_srca(cb_acc);
        pack_reconfig_data_format(cb_out);
        if constexpr (relu) {
            pack_relu_config(ReluConfig::zero());
        }
        tilize_init(cb_acc, N, cb_out);
        // FINAL_BATCH output tiles (flat (u, n) order) per DEST section.
        constexpr uint32_t total = bm * N;
        for (uint32_t f0 = 0; f0 < total; f0 += FINAL_BATCH) {
            const uint32_t cnt = total - f0 < FINAL_BATCH ? total - f0 : FINAL_BATCH;
            MATH((llk_math_wait_for_dest_available()));
            PACK((llk_packer_wait_for_math_done()));
            for (uint32_t j = 0; j < cnt; ++j) {
                const uint32_t u = (f0 + j) / N;
                const uint32_t n = (f0 + j) - u * N;
                UNPACK((unpack_tilize_at(acc_rd + rowbase(t0, u + 1) * PITCH, n)));
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
        tilize_uninit(cb_acc, cb_out);
        if constexpr (relu) {
            pack_relu_config(ReluConfig::none());
        }
    }
    tok.wait_front(bm);
    tok.pop_front(bm);
    acc_ready.pop_front(1);
}
