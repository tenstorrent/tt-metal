// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for the multi-core DRAM-sharded decode matmul (cores_per_bank > 0); the work decode and
// the block order match reader_dram_sharded_multicore.cpp.
//
// fp32_dest_acc_en: the whole K of a column pass stays in fp32 Dest (full sync, at most 8 tiles), with
// no partial sums in L1. Without it (DSMC_L1ACC) the pass accumulates G streamed blocks at a time in
// 16-bit Dest and the packer adds each group into a Float16_b accumulator in L1, the numerics of the
// single-reader path's packer L1 accumulation; the accumulator is copied into the output at the end. A core's columns
// are taken in P passes; the activation is resident for all of them and each pass re-streams only its own weight
// columns. With CK > 1 the core's partials go to L1 once, are exchanged inside the column group by the stream-1 kernel,
// and the owner of each column range adds the CK slices.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#ifdef DSMC_REDUCE
#include "api/compute/eltwise_binary.h"
#endif

namespace {
constexpr uint32_t Nbt = get_arg(args::Nbt);
constexpr uint32_t KT = get_arg(args::KT);
constexpr uint32_t KB = get_arg(args::KB);
constexpr uint32_t CN = get_arg(args::CN);
constexpr uint32_t CK = get_arg(args::CK);
constexpr uint32_t P = get_arg(args::P);
constexpr uint32_t NCP = get_arg(args::NCP);
constexpr uint32_t NGRP = get_arg(args::NGRP);
constexpr uint32_t ROT = get_arg(args::ROT);
constexpr uint32_t GX = get_arg(args::GX);
constexpr uint32_t MAXOWN = get_arg(args::MAXOWN);
constexpr uint32_t BT = KB * NCP;
constexpr uint32_t DST_TILES = 8;  // full-sync Dest tiles used per pass
#ifdef DSMC_L1ACC
constexpr uint32_t G = get_arg(args::G);  // streamed blocks per packer L1 accumulation step
#endif
constexpr uint8_t kRole[] = {DSMC_ROLE};
}  // namespace

void kernel_main() {
    const uint32_t role = kRole[get_absolute_logical_y() * GX + get_absolute_logical_x()];
    if (role == 0) {
        return;
    }
    const uint32_t wi = role - 1, kidx = wi % CK, grp = wi / CK, cn = grp % CN;
    const uint32_t n0 = Nbt * cn / CN, n1 = Nbt * (cn + 1) / CN;
    const uint32_t k0 = KT * kidx / CK, k1 = KT * (kidx + 1) / CK;
    const uint32_t nc = n1 - n0, rows = k1 - k0, nblk = (rows + KB - 1) / KB;
    const uint32_t rot = ROT ? grp * nblk / NGRP : 0;

    DataflowBuffer dfb_x(dfb::x);
    DataflowBuffer dfb_w0(dfb::w0);
#ifdef DSMC_TWO_STREAMS
    DataflowBuffer dfb_w1(dfb::w1);
#endif
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::x, dfb::w0, dfb::out);

#ifndef DSMC_REDUCE
#ifdef DSMC_L1ACC
    DataflowBuffer dfb_acc(dfb::acc);
    constexpr uint32_t sink = dfb::acc;
    dfb_acc.reserve_back(MAXOWN);
#else
    constexpr uint32_t sink = dfb::out;
    dfb_out.reserve_back(MAXOWN);
#endif
    pack_reconfig_data_format(sink);
    for (uint32_t q = 0; q < P; q++) {
        const uint32_t qa = nc * q / P, ncq = nc * (q + 1) / P - qa;
        matmul_block_init(dfb::x, dfb::w0, false, ncq, 1, 1);
#ifndef DSMC_L1ACC
        tile_regs_acquire();
#endif
        for (uint32_t p = 0; p < nblk; p++) {
#ifdef DSMC_L1ACC
            if (p % G == 0) {
                tile_regs_acquire();
            }
#endif
            const uint32_t j = (p + rot) % nblk, rb = (rows - j * KB) < KB ? (rows - j * KB) : KB;
            if (q == 0) {
                dfb_x.wait_front((p + 1) * KB);
            }
#ifdef DSMC_TWO_STREAMS
            const bool odd = (p & 1) != 0;
            DataflowBuffer& dfb_w = odd ? dfb_w1 : dfb_w0;
            const uint32_t w_id = odd ? static_cast<uint32_t>(dfb::w1) : static_cast<uint32_t>(dfb::w0);
#else
            DataflowBuffer& dfb_w = dfb_w0;
            const uint32_t w_id = dfb::w0;
#endif
            dfb_w.wait_front(BT);
            for (uint32_t r = 0; r < rb; r++) {
                matmul_block(dfb::x, w_id, p * KB + r, r * ncq, 0, false, ncq, 1, 1);
            }
            dfb_w.pop_front(BT);
#ifdef DSMC_L1ACC
            if ((p + 1) % G == 0 || p + 1 == nblk) {
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_l1_acc(p >= G ? 1 : 0);
                for (uint32_t jj = 0; jj < ncq; jj++) {
                    pack_tile<true>(jj, dfb::acc, qa + jj);
                }
                tile_regs_release();
            }
#endif
        }
#ifndef DSMC_L1ACC
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t jj = 0; jj < ncq; jj++) {
            pack_tile<true>(jj, dfb::out, qa + jj);
        }
        tile_regs_release();
#endif
    }
#ifdef DSMC_L1ACC
    pack_reconfig_l1_acc(0);
    dfb_acc.push_back(MAXOWN);
    dfb_acc.wait_front(MAXOWN);
    dfb_out.reserve_back(MAXOWN);
    reconfig_data_format_srca(dfb::w0, dfb::acc);
    reconfig_data_format_srcb(dfb::x, dfb::acc);
    pack_reconfig_data_format(dfb::out);
    copy_init(dfb::acc);
    for (uint32_t j0 = 0; j0 < nc; j0 += DST_TILES) {
        const uint32_t m = (nc - j0) < DST_TILES ? (nc - j0) : DST_TILES;
        tile_regs_acquire();
        for (uint32_t j = 0; j < m; j++) {
            copy_tile(dfb::acc, j0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < m; j++) {
            pack_tile<true>(j, dfb::out, j0 + j);
        }
        tile_regs_release();
    }
    dfb_acc.pop_front(MAXOWN);
#endif
    dfb_out.push_back(MAXOWN);
#else
    static_assert(P == 1, "column passes need CK == 1");
    DataflowBuffer dfb_part(dfb::part);
    DataflowBuffer dfb_recv(dfb::recv);
    matmul_block_init(dfb::x, dfb::w0, false, nc, 1, 1);
    dfb_part.reserve_back(NCP);
    pack_reconfig_data_format(dfb::part);
#ifndef DSMC_L1ACC
    tile_regs_acquire();
#endif
    for (uint32_t p = 0; p < nblk; p++) {
        const uint32_t j = (p + rot) % nblk, rb = (rows - j * KB) < KB ? (rows - j * KB) : KB;
#ifdef DSMC_L1ACC
        if (p % G == 0) {
            tile_regs_acquire();
        }
#endif
        dfb_x.wait_front((p + 1) * KB);
#ifdef DSMC_TWO_STREAMS
        const bool odd = (p & 1) != 0;
        DataflowBuffer& dfb_w = odd ? dfb_w1 : dfb_w0;
        const uint32_t w_id = odd ? static_cast<uint32_t>(dfb::w1) : static_cast<uint32_t>(dfb::w0);
#else
        DataflowBuffer& dfb_w = dfb_w0;
        const uint32_t w_id = dfb::w0;
#endif
        dfb_w.wait_front(BT);
        for (uint32_t r = 0; r < rb; r++) {
            matmul_block(dfb::x, w_id, p * KB + r, r * nc, 0, false, nc, 1, 1);
        }
        dfb_w.pop_front(BT);
#ifdef DSMC_L1ACC
        if ((p + 1) % G == 0 || p + 1 == nblk) {
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(p >= G ? 1 : 0);
            for (uint32_t jj = 0; jj < nc; jj++) {
                pack_tile<true>(jj, dfb::part, jj);
            }
            tile_regs_release();
        }
#endif
    }
#ifdef DSMC_L1ACC
    pack_reconfig_l1_acc(0);
#else
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t jj = 0; jj < nc; jj++) {
        pack_tile<true>(jj, dfb::part, jj);
    }
    tile_regs_release();
#endif
    dfb_part.push_back(NCP);

    // Owner: add the CK slices of its columns (slot q holds member q's partials of them).
    const uint32_t nown = n0 + nc * (kidx + 1) / CK - (n0 + nc * kidx / CK);
    dfb_recv.wait_front(CK * MAXOWN);
    dfb_out.reserve_back(MAXOWN);
    reconfig_data_format_srca(dfb::w0, dfb::recv);
    reconfig_data_format_srcb(dfb::x, dfb::recv);
    pack_reconfig_data_format(dfb::out);
    for (uint32_t j0 = 0; j0 < nown; j0 += DST_TILES) {
        const uint32_t m = (nown - j0) < DST_TILES ? (nown - j0) : DST_TILES;
        tile_regs_acquire();
        copy_init(dfb::recv);
        for (uint32_t j = 0; j < m; j++) {
            copy_tile(dfb::recv, j0 + j, j);
        }
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::recv);
        for (uint32_t q = 1; q < CK; q++) {
            for (uint32_t j = 0; j < m; j++) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::recv, q * MAXOWN + j0 + j, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < m; j++) {
            pack_tile<true>(j, dfb::out, j0 + j);
        }
        tile_regs_release();
    }
    dfb_out.push_back(MAXOWN);
    dfb_recv.pop_front(CK * MAXOWN);
#endif
    dfb_x.pop_front(nblk * KB);
}
