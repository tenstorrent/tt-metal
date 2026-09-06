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
    const bool has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }
    // Read but unused, as in the legacy kernel. A row-major input arrives already untilized, so this
    // kernel never touches either input buffer and has nothing to select between -- unlike the tiled
    // sibling, where is_input1 chooses which input to untilize. Kept so the host's per-node runtime
    // argument emission is unchanged. The two dead input buffer-index compile-time args the legacy
    // kernel read alongside it are gone: a buffer index is expressed as a dataflow-buffer binding in
    // this API and nothing else, and there is no endpoint here to declare.
    [[maybe_unused]] const bool is_input1 = get_arg(args::is_input1);

    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t num_heads = get_arg(args::num_heads);

    // dfb::cache holds the cache tiles the reader pulled in. dfb::untilized_cache and
    // dfb::untilized_cache2 are aliased -- the writer patches the new row into the region published
    // through the first and republishes it through the second, which is what this kernel re-tilizes
    // into dfb::out.
    compute_kernel_hw_startup(dfb::cache, dfb::untilized_cache);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration
        compute_kernel_lib::untilize<Wt, dfb::cache, dfb::untilized_cache>(1);

        // Wait on writer to update block. Tilize with reconfiguration
        compute_kernel_lib::tilize<Wt, dfb::untilized_cache2, dfb::out>(1);
    }
}
