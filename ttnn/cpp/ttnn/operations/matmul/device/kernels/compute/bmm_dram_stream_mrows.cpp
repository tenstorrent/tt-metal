// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for the multi-row DRAM-streamed matmul; work decode and block order match reader_dram_stream_mrows.cpp.
// A work core's columns are taken in P passes of at most (Dest tiles / MT) columns. A pass keeps its MT x ncq output
// tiles in Dest over the whole K (full sync, no partial sums in L1) and streams only its own weight columns; the
// activation (k-major: tile (m, k) at k * MT + m) is resident for every pass, so matmul_block reads row m of K tile k
// at in0 index k * MT + m with a row stride (kt_dim) of one tile.

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
#ifdef MR_SWIGLU
#include "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/swiglu_sfpu.hpp"
#endif

namespace {
constexpr uint32_t Nbt = get_arg(args::Nbt);
constexpr uint32_t KT = get_arg(args::KT);
constexpr uint32_t KB = get_arg(args::KB);
constexpr uint32_t CN = get_arg(args::CN);
constexpr uint32_t P = get_arg(args::P);
constexpr uint32_t NCP = get_arg(args::NCP);
constexpr uint32_t GX = get_arg(args::GX);
constexpr uint32_t NW = get_arg(args::NW);
constexpr uint32_t MT = get_arg(args::MT);
constexpr uint32_t OD = get_arg(args::OD);  // 2 with MR_SWIGLU: one output tile per gate / up pair
constexpr uint32_t BT = KB * NCP;
constexpr uint32_t NBLK = (KT + KB - 1) / KB;
#ifdef MR_L1ACC
constexpr uint32_t G = get_arg(args::G);            // streamed blocks per packer L1 accumulation step
constexpr uint32_t MAXACC = get_arg(args::MAXACC);  // Float32 accumulator tiles (MT x the widest core's columns)
#endif
constexpr uint8_t kRole[] = {DSMC_ROLE};

#ifdef MR_SWIGLU
constexpr bool kSwigluBlockFloat = get_arg(args::SWIGLU_BF) == 1;
// Dest tiles [0, n) hold gate / up pairs; leaves SiLU(gate) * up of pair i in Dest tile i.
FORCE_INLINE void swiglu_dest(uint32_t n) {
    MATH((ckernel::minimal_matmul_swiglu_tile_init()));
    for (uint32_t i = 0; i < n; i += 2) {
        MATH((ckernel::minimal_matmul_swiglu_tile<kSwigluBlockFloat>(i, i + 1, i / 2)));
    }
}
#endif
}  // namespace

void kernel_main() {
    const uint32_t role = kRole[get_absolute_logical_y() * GX + get_absolute_logical_x()];
    if (role == 0 || role > NW) {
        return;
    }
    const uint32_t wi = role - 1, cn = wi % CN;
    const uint32_t n0 = Nbt * cn / CN, n1 = Nbt * (cn + 1) / CN, nc = n1 - n0;

    DataflowBuffer dfb_x(dfb::x);
    DataflowBuffer dfb_w0(dfb::w0);
#ifdef MR_TWO_STREAMS
    DataflowBuffer dfb_w1(dfb::w1);
#endif
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::x, dfb::w0, dfb::out);
#ifdef MR_L1ACC
    // 16-bit Dest takes G blocks at a time; the packer adds each group into the Float32 accumulator (the first group
    // of a pass overwrites), and the accumulator is copied into the output after the last pass.
    DataflowBuffer dfb_acc(dfb::acc);
    dfb_acc.reserve_back(MAXACC);
    pack_reconfig_data_format(dfb::acc);
#else
    pack_reconfig_data_format(dfb::out);
#endif
    for (uint32_t q = 0; q < P; q++) {
        const uint32_t qa = nc * q / P, ncq = nc * (q + 1) / P - qa, nq = ncq / OD;
        matmul_block_init(dfb::x, dfb::w0, false, ncq, MT, 1);
#ifndef MR_L1ACC
        dfb_out.reserve_back(MT * nq);
        tile_regs_acquire();
#else
        (void)nq;
#endif
        for (uint32_t p = 0; p < NBLK; p++) {
#ifdef MR_L1ACC
            if (p % G == 0) {
                tile_regs_acquire();
            }
#endif
            const uint32_t r0 = p * KB, rb = (KT - r0) < KB ? (KT - r0) : KB;
            if (q == 0) {
                dfb_x.wait_front((r0 + rb) * MT);
            }
#ifdef MR_TWO_STREAMS
            const bool odd = (p & 1) != 0;
            DataflowBuffer& dfb_w = odd ? dfb_w1 : dfb_w0;
            const uint32_t w_id = odd ? static_cast<uint32_t>(dfb::w1) : static_cast<uint32_t>(dfb::w0);
#else
            DataflowBuffer& dfb_w = dfb_w0;
            const uint32_t w_id = dfb::w0;
#endif
            dfb_w.wait_front(BT);
#ifndef MR_ABL_NO_MM
            for (uint32_t r = 0; r < rb; r++) {
                matmul_block(dfb::x, w_id, (r0 + r) * MT, r * ncq, 0, false, ncq, MT, 1);
            }
#endif
            dfb_w.pop_front(BT);
#ifdef MR_L1ACC
            if ((p + 1) % G == 0 || p + 1 == NBLK) {
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_l1_acc(p >= G ? 1 : 0);
                for (uint32_t m = 0; m < MT; m++) {
                    for (uint32_t j = 0; j < ncq; j++) {
                        pack_tile<true>(m * ncq + j, dfb::acc, m * nc + qa + j);
                    }
                }
                tile_regs_release();
            }
#endif
        }
#ifndef MR_L1ACC
#ifdef MR_SWIGLU
        swiglu_dest(MT * ncq);
#endif
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < MT * nq; t++) {
            pack_tile<true>(t, dfb::out, t);
        }
        tile_regs_release();
        dfb_out.push_back(MT * nq);
#endif
    }
#ifdef MR_L1ACC
    pack_reconfig_l1_acc(0);
    dfb_acc.push_back(MAXACC);
    dfb_acc.wait_front(MAXACC);
    reconfig_data_format_srca(dfb::w0, dfb::acc);
    reconfig_data_format_srcb(dfb::x, dfb::acc);
    pack_reconfig_data_format(dfb::out);
    for (uint32_t q = 0; q < P; q++) {
        const uint32_t qa = nc * q / P, ncq = nc * (q + 1) / P - qa, nq = ncq / OD;
        dfb_out.reserve_back(MT * nq);
        tile_regs_acquire();
        copy_init(dfb::acc);
        for (uint32_t m = 0; m < MT; m++) {
            for (uint32_t j = 0; j < ncq; j++) {
                copy_tile(dfb::acc, m * nc + qa + j, m * ncq + j);
            }
        }
#ifdef MR_SWIGLU
        swiglu_dest(MT * ncq);
#endif
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < MT * nq; t++) {
            pack_tile<true>(t, dfb::out, t);
        }
        tile_regs_release();
        dfb_out.push_back(MT * nq);
    }
    dfb_acc.pop_front(MAXACC);
#endif
    dfb_x.pop_front(KT * MT);
}
