// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streamed-expert compute (TRISC) on one of the 64 compute cores.
// Per expert:
//   gate/up: DST[mt*2 + {0: gate, 1: up}] += x block [MT x KBLK] @ in1 block [KBLK x 2] over NK_GU K-blocks (the two
//            in1 columns are this core's gate and up tile columns), then h = silu(gate) * up in DST, packed to H_LOCAL.
//   down:    DST[mt*2 + n] += h_all block [MT x KBLK] @ in1 block [KBLK x 2] over NK_D K-blocks; h_all holds all 64
//            cores' h slices, row-major within each K-block (like x), packed to OUT.
// Accumulation stays in DST for the whole K walk (MT * 2 <= 8 tiles, half of DST), so there is no reload.
//
// CT: 0 KBLK, 1 MT, 2 NK_GU, 3 NK_D, 4 NUM_EXPERTS, 5 SLOT_TILES (landing slot; a block uses its first tiles),
//     6 PCD (down output tile columns of this core), 7 W (down pass width: MT * W <= 8 DST tiles; passes of W columns,
//     the last one PCD % W wide, each its own K walk over h_all and its own weight blocks [KBLK x width])
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

// The matmul configuration is set once: x and h_all are both bf8 with the same block shape, so gate/up and down share
// one matmul_block_init and no srcb reconfig. h = silu(gate) * up runs on the PACK thread's SFPU against the DEST half
// it has just been handed, so MATH moves straight on to the next matmul in the other half while the activation runs.
//
// Pipelined order: gate/up of expert e + 1 runs before down of expert e, so the h exchange of expert e (all 64 cores
// write their slice everywhere) overlaps a full gate/up instead of stalling compute. The weight stream is laid out in
// the same order: gu(0), gu(1), d(0), gu(2), d(1), ..., d(E - 1).
// The matmul MOP depends on ct_dim, so it is re-initialised only when the block width changes (gate/up is 2 wide).
uint32_t cur_ct = 2;
FORCE_INLINE void set_ct(uint32_t ct, uint32_t mt, uint32_t kblk) {
    if (ct != cur_ct) {
        matmul_block_init(tt::CBIndex::c_2, tt::CBIndex::c_1, false, ct, mt, kblk);
        cur_ct = ct;
    }
}

template <uint32_t kblk, uint32_t mt, uint32_t nk_gu, uint32_t slot_tiles>
FORCE_INLINE void gate_up() {
    constexpr uint32_t x_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t h_local_cb = tt::CBIndex::c_3;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_GU_MM");
#endif
        set_ct(2, mt, kblk);
        tile_regs_acquire();
        for (uint32_t b = 0; b < nk_gu; ++b) {
            cb_wait_front(x_cb, mt * kblk);
            cb_wait_front(in1_cb, slot_tiles);
            for (uint32_t k = 0; k < kblk; ++k) {
                matmul_block(x_cb, in1_cb, k, k * 2, 0, false, 2, mt, kblk);
            }
            cb_pop_front(x_cb, mt * kblk);
            cb_pop_front(in1_cb, slot_tiles);
        }
        tile_regs_commit();
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

template <
    uint32_t kblk,
    uint32_t mt,
    uint32_t nk_d,
    uint32_t h_all_tiles,
    uint32_t slot_tiles,
    uint32_t pcd,
    uint32_t w>
FORCE_INLINE void down() {
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t h_all_cb = tt::CBIndex::c_2;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_H_WAIT");
#endif
        cb_wait_front(h_all_cb, h_all_tiles);
    }
#ifdef SE_ZONES
    DeviceZoneScopedN("SE_DOWN");
#endif
    cb_reserve_back(out_cb, mt * pcd);
    for (uint32_t col = 0; col < pcd; col += w) {
        const uint32_t cw = pcd - col < w ? pcd - col : w;
        set_ct(cw, mt, kblk);
        tile_regs_acquire();
        for (uint32_t b = 0; b < nk_d; ++b) {
            cb_wait_front(in1_cb, slot_tiles);
            for (uint32_t k = 0; k < kblk; ++k) {
                matmul_block(h_all_cb, in1_cb, b * mt * kblk + k, k * cw, 0, false, cw, mt, kblk);
            }
            cb_pop_front(in1_cb, slot_tiles);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < mt * cw; ++i) {
            pack_tile<true>(i, out_cb, mt * col + i);  // pass-major: [pass][m][column of the pass]
        }
        tile_regs_release();
    }
    cb_pop_front(h_all_cb, h_all_tiles);
    cb_push_back(out_cb, mt * pcd);
}

void kernel_main() {
    constexpr uint32_t kblk = get_compile_time_arg_val(0);
    constexpr uint32_t mt = get_compile_time_arg_val(1);
    constexpr uint32_t nk_gu = get_compile_time_arg_val(2);
    constexpr uint32_t nk_d = get_compile_time_arg_val(3);
    constexpr uint32_t num_experts = get_compile_time_arg_val(4);
    constexpr uint32_t slot_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t pcd = get_compile_time_arg_val(6);
    constexpr uint32_t w = get_compile_time_arg_val(7);
    constexpr uint32_t x_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t h_all_tiles = nk_d * kblk * mt;
    static_assert(mt * 2 <= 8 && mt * w <= 8, "accumulators must fit in half of DST");

    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, in1_cb, out_cb);
    matmul_block_init(x_cb, in1_cb, false, 2, mt, kblk);
#ifndef SE_NO_ACT
    silu_tile_init_pack();
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif

    gate_up<kblk, mt, nk_gu, slot_tiles>();
    for (uint32_t e = 0; e < num_experts; ++e) {
        if (e + 1 < num_experts) {
            gate_up<kblk, mt, nk_gu, slot_tiles>();
        }
        down<kblk, mt, nk_d, h_all_tiles, slot_tiles, pcd, w>();
    }
}
