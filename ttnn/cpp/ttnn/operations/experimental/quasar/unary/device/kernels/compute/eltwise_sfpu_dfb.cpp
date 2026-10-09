// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar unary compute: DFB port of eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp.
// Per tile: unpack in -> FPU datacopy to DEST[0] -> SFPU_OP_CHAIN_0 (the factory's get_block_defines, e.g.
// "cos_tile_init(); cos_tile(0);") -> pack DEST[0] to out. The include set is trimmed to what compiles on
// every arch: the upstream kernel's unconditional rpow/rdiv/fill/mul_int includes are not needed by the
// chain, which pulls its own op header through sfpu_split_includes.h (SFPU_OP_*_INCLUDE defines).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t in_id = dfb_in.get_id();
    const uint32_t out_id = dfb_out.get_id();

    compute_kernel_hw_startup(in_id, out_id);
    copy_init(in_id);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        // copy_init before every copy, as the Quasar SFPU test kernel (test_kernels/compute/eltwise_sfpu_2_0.cpp)
        // does: the chain's SFPU init runs between two copies.
        copy_init(in_id);
        copy_tile(in_id, /*tile_index=*/0, /*dst_index=*/0);
#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(/*dst_index=*/0, out_id);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_in.pop_front(1);
    }
}
