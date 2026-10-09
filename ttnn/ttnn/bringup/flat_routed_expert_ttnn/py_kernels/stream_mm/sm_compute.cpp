// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streamed-matmul compute (TRISC) on a compute core: out[Mt x PCN] = sum over K-blocks of in0 block [Mt x KBLK] x in1
// block [KBLK x PCN]. Partial sums accumulate in the output CB through the packer's L1 accumulation (block 0 packs,
// later blocks add in place), so no reload / format switch is needed. Output subblocks are OSH x PCN tiles.
// NUM_EXPERTS repeats reuse the same in0 and produce one output each.
//
// CT: 0 KBLK, 1 MT, 2 PCN, 3 NK (K-blocks), 4 OSH, 5 NUM_EXPERTS
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t kblk = get_compile_time_arg_val(0);
    constexpr uint32_t mt = get_compile_time_arg_val(1);
    constexpr uint32_t pcn = get_compile_time_arg_val(2);
    constexpr uint32_t nk = get_compile_time_arg_val(3);
    constexpr uint32_t osh = get_compile_time_arg_val(4);
    constexpr uint32_t num_experts = get_compile_time_arg_val(5);
    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t out_tiles = mt * pcn;
    static_assert(mt % osh == 0, "OSH must divide MT");

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, out_cb);
    matmul_block_init(in0_cb, in1_cb, false, pcn, osh, kblk);

    for (uint32_t e = 0; e < num_experts; ++e) {
        cb_reserve_back(out_cb, out_tiles);
        for (uint32_t b = 0; b < nk; ++b) {
            cb_wait_front(in0_cb, mt * kblk);
            cb_wait_front(in1_cb, kblk * pcn);
            pack_reconfig_l1_acc(b > 0 ? 1 : 0);
            for (uint32_t hs = 0; hs < mt / osh; ++hs) {
                tile_regs_acquire();
                uint32_t in0_index = hs * osh * kblk;
                uint32_t in1_index = 0;
                for (uint32_t k = 0; k < kblk; ++k) {
                    matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, false, pcn, osh, kblk);
                    ++in0_index;
                    in1_index += pcn;
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < osh * pcn; ++i) {
                    pack_tile<true>(i, out_cb, hs * osh * pcn + i);
                }
                tile_regs_release();
            }
            cb_pop_front(in0_cb, mt * kblk);
            cb_pop_front(in1_cb, kblk * pcn);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(out_cb, out_tiles);
    }
}
