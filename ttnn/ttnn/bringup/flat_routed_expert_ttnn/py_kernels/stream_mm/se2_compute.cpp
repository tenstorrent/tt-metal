// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Big-M streamed-expert compute (TRISC) on one of the 64 compute cores. An expert's M rows run as S sub-blocks of
// MT tiles ("virtual experts" v = e * S + s) that all reuse the expert's weight slice, which stays resident in the in1
// ring (capacity: exactly one expert, BPE blocks):
//   gate/up(v): DST[m * 2 + {gate, up}] += x(v) [MT x KBLK per K-block, streamed] @ in1 gu block [KBLK x 2], over all
//               NK_GU K-blocks; h = silu(gate) * up on the PACK thread's SFPU, packed to H_LOCAL.
//   down(v):    DST[m * PCD + c] += h_all(v) [MT x KT_D] @ in1 down blocks [KBLK_D x PCD], in row passes of RT_D
//               tiles (RT_D * PCD <= 8); h_all is row-major within K-blocks of 8 tiles.
// A weight block is freed (popped) only after its last use, gu(e, S - 1) / d(e, S - 1); since the ring only pops from
// the front and gu(e + 1, 0) runs before d(e, S - 1) (pipelined order), fully used blocks are popped lazily in ring
// order, which lets the next expert's blocks land while the current one finishes.
//
// CT: 0 KBLK, 1 MT, 2 NK_GU, 3 KT_D, 4 KBLK_D, 5 NUM_EXPERTS, 6 S, 7 SLOT_TILES, 8 PCD, 9 RT_D
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#ifdef TRISC_PACK
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#endif

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t in1_cb = tt::CBIndex::c_1;
constexpr uint32_t h_all_cb = tt::CBIndex::c_2;
constexpr uint32_t h_local_cb = tt::CBIndex::c_3;
constexpr uint32_t out_cb = tt::CBIndex::c_16;

constexpr uint32_t kblk = get_compile_time_arg_val(0);
constexpr uint32_t mt = get_compile_time_arg_val(1);
constexpr uint32_t nk_gu = get_compile_time_arg_val(2);
constexpr uint32_t kt_d = get_compile_time_arg_val(3);
constexpr uint32_t kblk_d = get_compile_time_arg_val(4);
constexpr uint32_t num_experts = get_compile_time_arg_val(5);
constexpr uint32_t sub = get_compile_time_arg_val(6);
constexpr uint32_t slot = get_compile_time_arg_val(7);
constexpr uint32_t pcd = get_compile_time_arg_val(8);
constexpr uint32_t rt_d = get_compile_time_arg_val(9);
constexpr uint32_t nk_dd = kt_d / kblk_d;
constexpr uint32_t bpe = nk_gu + nk_dd;  // weight blocks per expert
constexpr uint32_t h_all_tiles = kt_d * mt;
constexpr uint32_t hk = 8;  // h_all K-block width (the gather layout)
static_assert(mt * 2 <= 8 && rt_d * pcd <= 8 && mt % rt_d == 0 && kt_d % kblk_d == 0 && kt_d % hk == 0);

uint32_t popped = 0;               // weight blocks popped from the in1 ring (absolute block index of its front)
uint32_t gu_done = 0, d_done = 0;  // experts whose gate/up (down) blocks have all had their last use
uint32_t gu_last = 0, d_last = 0;  // blocks of expert gu_done (d_done) already through their last use
uint32_t cur_ct = 2, cur_rt = mt;

FORCE_INLINE void set_mm(uint32_t ct, uint32_t rt) {
    if (ct != cur_ct || rt != cur_rt) {
        matmul_block_init(h_all_cb, in1_cb, false, ct, rt, kblk);
        cur_ct = ct;
        cur_rt = rt;
    }
}

// Tile index of absolute weight block a relative to the ring's read pointer, after waiting for it to land. Block a
// sits in slot a % BPE, and the unpacker does not wrap at the end of the ring, so a block past the wrap point gets a
// negative index (the unpacker's address arithmetic is modulo 2^32).
FORCE_INLINE uint32_t wblock(uint32_t a) {
    cb_wait_front(in1_cb, (a - popped + 1) * slot);
    return static_cast<uint32_t>(static_cast<int32_t>(a % bpe) - static_cast<int32_t>(popped % bpe)) * slot;
}

FORCE_INLINE void pop_used() {
    while (true) {
        const uint32_t e = popped / bpe, j = popped % bpe;
        const bool used = j < nk_gu ? (gu_done > e || (gu_done == e && j < gu_last))
                                    : (d_done > e || (d_done == e && j - nk_gu < d_last));
        if (!used) {
            break;
        }
        cb_pop_front(in1_cb, slot);
        ++popped;
    }
}

FORCE_INLINE void gate_up(uint32_t v) {
    const uint32_t e = v / sub;
    const bool last = v % sub == sub - 1;
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_GU_MM");
#endif
        set_mm(2, mt);
        tile_regs_acquire();
        for (uint32_t b = 0; b < nk_gu; ++b) {
            cb_wait_front(x_cb, mt * kblk);
            const uint32_t w = wblock(e * bpe + b);
            for (uint32_t k = 0; k < kblk; ++k) {
                matmul_block(x_cb, in1_cb, k, w + k * 2, 0, false, 2, mt, kblk);
            }
            cb_pop_front(x_cb, mt * kblk);
            if (last) {  // free the block for the next expert as soon as possible
                gu_last = b + 1;
                pop_used();
            }
        }
        tile_regs_commit();
    }
    if (last) {
        ++gu_done;
        gu_last = 0;
        pop_used();
    }
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_GU_ACT_PACK");
#endif
        cb_reserve_back(h_local_cb, mt);
        tile_regs_wait();
#ifndef SE_NO_ACT
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t m = 0; m < mt; ++m) {
            silu_tile_pack(m * 2);
        }
        for (uint32_t m = 0; m < mt; ++m) {
            PACK((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sfpu_binary_mul,
                (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
                m * 2,
                m * 2 + 1,
                m * 2,
                VectorMode::RC)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
        pack_reconfig_data_format(out_cb, h_local_cb);
        for (uint32_t m = 0; m < mt; ++m) {
            pack_tile(m * 2, h_local_cb);
        }
        pack_reconfig_data_format(h_local_cb, out_cb);
        tile_regs_release();
        cb_push_back(h_local_cb, mt);
    }
}

FORCE_INLINE void down(uint32_t v) {
    const uint32_t e = v / sub;
    const bool last = v % sub == sub - 1;
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_H_WAIT");
#endif
        cb_wait_front(h_all_cb, h_all_tiles);
    }
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_DOWN");
#endif
        set_mm(pcd, rt_d);
        cb_reserve_back(out_cb, mt * pcd);
        for (uint32_t r = 0; r < mt; r += rt_d) {
            tile_regs_acquire();
            uint32_t w = 0;
            for (uint32_t kk = 0; kk < kt_d; ++kk) {
                if (kk % kblk_d == 0) {
                    w = wblock(e * bpe + nk_gu + kk / kblk_d);
                }
                matmul_block(
                    h_all_cb,
                    in1_cb,
                    (kk / hk) * mt * hk + r * hk + kk % hk,
                    w + (kk % kblk_d) * pcd,
                    0,
                    false,
                    pcd,
                    rt_d,
                    hk);
                if (last && r + rt_d == mt && kk % kblk_d == kblk_d - 1) {
                    d_last = kk / kblk_d + 1;
                    pop_used();
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < rt_d * pcd; ++i) {
                pack_tile<true>(i, out_cb, r * pcd + i);  // row-major [MT x PCD]
            }
            tile_regs_release();
        }
        cb_pop_front(h_all_cb, h_all_tiles);
        cb_push_back(out_cb, mt * pcd);
    }
    if (last) {
        ++d_done;
        d_last = 0;
        pop_used();
    }
}

void kernel_main() {
    constexpr uint32_t num_v = num_experts * sub;
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, in1_cb, out_cb);
    matmul_block_init(x_cb, in1_cb, false, 2, mt, kblk);
#ifndef SE_NO_ACT
    silu_tile_init_pack();
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif
    // Pipelined order: gate/up of v + 1 before down of v hides the h exchange of v.
    gate_up(0);
    for (uint32_t v = 0; v < num_v; ++v) {
        if (v + 1 < num_v) {
            gate_up(v + 1);
        }
        down(v);
    }
}
