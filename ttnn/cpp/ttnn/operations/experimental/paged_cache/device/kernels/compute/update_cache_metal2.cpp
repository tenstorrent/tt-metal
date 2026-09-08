// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"
#include "api/compute/pack.h"

// Helper constexpr function to compute num_blocks_per_col
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;

    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }

    return 1;
}

void kernel_main() {
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t num_heads = get_arg(args::num_heads);

    // dfb::cache holds the cache tiles the reader pulled in; dfb::in the resident input shard.
    // dfb::untilized_cache and dfb::untilized_cache2 are aliased — the writer patches the new row
    // into the region published through the first and republishes it through the second, which is
    // what this kernel re-tilizes into dfb::out.
    compute_kernel_hw_startup(dfb::in, dfb::untilized_in);

    // Untilize input (standalone operation)
    compute_kernel_lib::untilize<
        Wt,
        dfb::in,
        dfb::untilized_in,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        compute_kernel_lib::untilize<Wt, dfb::cache, dfb::untilized_cache>(1);

        // Wait on writer to update block, then tilize back
#ifdef ARCH_QUASAR
        // Quasar: tilize_init programs unpack+math only and pack_reconfig_data_format is gasket-only, so
        // the packer's L1 destination (BFD) still points at dfb::untilized_cache from pack_untilize_init
        // above -> the re-tilized block would land in the untilize ring and dfb::out would never be
        // written (all-zero cache block). Retarget the packer (quasar_porting.md s7; same idiom as
        // experimental/quasar/pool_generic compute_pool_2d.cpp and conv2d conv_bmm_tilize_metal2.cpp).
        pack_init(dfb::out);
#endif
        compute_kernel_lib::tilize<Wt, dfb::untilized_cache2, dfb::out>(1);
    }
}
