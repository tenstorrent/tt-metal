// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t packed_scalar1 = get_arg(args::packed_scalar1);  // value1 (multiplier b)
    const uint32_t packed_scalar2 = get_arg(args::packed_scalar2);  // value2 (addend c)
    const auto multiplier = reinterpret_cast<const float*>(&packed_scalar1);
    const auto addend = reinterpret_cast<const float*>(&packed_scalar2);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        tile_regs_acquire();
        copy_init(dfb::in);
        copy_tile(dfb::in, 0, 0);  // a -> dst[0]

        fill_tile_init();
        fill_tile(1, *multiplier);  // b (scalar) -> dst[1]
        fill_tile(2, *addend);      // c (scalar) -> dst[2]

#ifndef SFPU_OP_CHAIN_0
#error "mac_tss_kernel requires SFPU_OP_CHAIN_0 to be defined via get_block_defines"
#endif
        // expands to mac_tile_init<DataFormat>(); mac_tile<DataFormat>(0, 1, 2, 0);
        SFPU_OP_CHAIN_0
        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
}
