// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// N-block row sum with a tiled output, compute: per block, the N inputs' 32 row segments summed by the packer in L1
// (c_24, 32 rows x 1024 row major; the first input overwrites), then tilized into c_16 (32 tiles).
// CT: 0 N   Common RT: 0 blocks, 1 P, 2 grid x (blocks me, me + P, ...)
#include <cstdint>
#include "api/compute/common.h"
#include "core_range.hpp"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    const uint32_t n = strided_count(
        core_index(get_common_arg_val<uint32_t>(2)), get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1));
    constexpr uint32_t DST = 4;
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_24);
    for (uint32_t blk = 0; blk < n; ++blk) {
        copy_tile_to_dst_init_short(tt::CBIndex::c_0);
        pack_reconfig_data_format(tt::CBIndex::c_24);
        cb_reserve_back(tt::CBIndex::c_24, 32);
        for (uint32_t i = 0; i < N; ++i) {
            cb_wait_front(tt::CBIndex::c_0, 32);
            pack_reconfig_l1_acc(i ? 1 : 0);
            for (uint32_t j0 = 0; j0 < 32; j0 += DST) {
                tile_regs_acquire();
                for (uint32_t j = 0; j < DST; ++j) {
                    copy_tile(tt::CBIndex::c_0, j0 + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < DST; ++j) {
                    pack_tile<true>(j, tt::CBIndex::c_24, j0 + j);
                }
                tile_regs_release();
            }
            cb_pop_front(tt::CBIndex::c_0, 32);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(tt::CBIndex::c_24, 32);
        compute_kernel_lib::tilize<32, tt::CBIndex::c_24, tt::CBIndex::c_16>(1);
    }
}
