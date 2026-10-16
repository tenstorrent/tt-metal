// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }

    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto num_heads = get_arg(args::num_heads);

    // Row-major input needs no untilize step, so this kernel never touches an input buffer; the
    // writer drains the resident shard directly.
    compute_kernel_hw_startup(dfb::cache, dfb::untilized_cache);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration
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
}
