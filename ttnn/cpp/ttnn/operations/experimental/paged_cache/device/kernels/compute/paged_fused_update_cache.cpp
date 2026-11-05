// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }

    constexpr auto Wt = get_arg(args::Wt);
#ifdef ROW_PATCH
    // bfp8_b cache: pack the input (one tile row: a head per row) to bfp8_b once. Every row of a bfp8_b tile has its
    // own shared exponent per 16 values, so row h of these tiles is exactly the encoding head h's row gets in a
    // re-packed cache block; the writer copies those bytes into the cache in place.
    compute_kernel_hw_startup(dfb::in, dfb::patch_src);
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_patch(dfb::patch_src);
    dfb_in.wait_front(Wt);
    dfb_patch.reserve_back(Wt);
    copy_init(dfb::in);
    for (uint32_t t = 0; t < Wt; ++t) {
        tile_regs_acquire();
        copy_tile(dfb::in, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, dfb::patch_src, t);
        tile_regs_release();
    }
    dfb_patch.push_back(Wt);
    dfb_in.pop_front(Wt);
    return;
#else
    const auto num_local_heads = get_arg(args::num_local_heads);

    // Which of the two inputs this instance serves is structural: the factory places one compute
    // kernel per input shard grid, and each binds only its own input buffer.
    compute_kernel_hw_startup(dfb::in, dfb::untilized_in);

    // Untilize input (single block, init only - no uninit needed)
    compute_kernel_lib::untilize<
        Wt,
        dfb::in,
        dfb::untilized_in,
        compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

    for (uint32_t cur_head = 0; cur_head < num_local_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration from previous iteration
        compute_kernel_lib::untilize<Wt, dfb::cache, dfb::untilized_cache>(1);

        // Wait on writer to update block. Tilize with reconfiguration
#ifdef ARCH_QUASAR
        // Quasar: tilize_init programs unpack+math only and pack_reconfig_data_format is gasket-only, so
        // the packer's L1 destination (BFD) still points at dfb::untilized_cache from the untilize above
        // -> the re-tilized block would land in the untilize ring and dfb::out would never be written
        // (all-zero cache block). Retarget the packer BFD before packing. Same fix as update_cache.cpp.
        pack_init(dfb::out);
#endif
        compute_kernel_lib::tilize<Wt, dfb::untilized_cache2, dfb::out>(1);
    }
#endif
}
