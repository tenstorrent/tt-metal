// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <
    int BATCH_SIZE,
    dfb::AccessPattern InPap,
    dfb::AccessPattern InCap,
    dfb::AccessPattern OutPap,
    dfb::AccessPattern OutCap>
FORCE_INLINE void transpose(DFBBindingToken<InPap, InCap> dfb_in, DFBBindingToken<OutPap, OutCap> dfb_out) {
    DataflowBuffer dfb_in_dfb(dfb_in);
    DataflowBuffer dfb_out_dfb(dfb_out);
    dfb_in_dfb.wait_front(BATCH_SIZE);

    tile_regs_acquire();
    for (std::uint32_t i = 0; i < BATCH_SIZE; i++) {
        transpose_tile(dfb_in, i, i);
    }
    tile_regs_commit();
    dfb_in_dfb.pop_front(BATCH_SIZE);

    dfb_out_dfb.reserve_back(BATCH_SIZE);
    tile_regs_wait();
    pack_untilize_dest<1>(dfb_out, BATCH_SIZE);
    tile_regs_release();

    dfb_out_dfb.push_back(BATCH_SIZE);
}
void kernel_main() {
    constexpr int BATCH_SIZE = 8;
    const std::uint32_t total_tiles = get_arg(args::total_tiles);
    const std::uint32_t num_batches = total_tiles / BATCH_SIZE;
    const std::uint32_t leftover = total_tiles % BATCH_SIZE;

    compute_kernel_hw_startup(dfb::in, dfb::transpose_in);
    pack_untilize_init(dfb::in, dfb::transpose_in);
    transpose_init(dfb::in);

    pack_untilize_dest_init<1>(dfb::transpose_in);

    for (std::uint32_t i = 0; i < num_batches; i++) {
        transpose<BATCH_SIZE>(dfb::in, dfb::transpose_in);
    }

    for (std::uint32_t idx = 0; idx < leftover; idx++) {
        transpose<1>(dfb::in, dfb::transpose_in);
    }
    pack_untilize_uninit(dfb::transpose_in);
}
