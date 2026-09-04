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
    // Selects which of the two fused inputs this core untilizes. Both input DFBs are bound to this
    // kernel -- a kernel cannot touch a DFB it has not bound -- and only one is touched per
    // invocation. The ternary yields a DFBBindingToken because all DFB tokens share one type
    // (identity lives in a runtime member), which is what lets a runtime value pick a handle here.
    const bool is_input1 = get_arg(args::is_input1);

    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t num_heads = get_arg(args::num_heads);

    // dfb::cache holds the cache tiles the reader pulled in; dfb::src1 / dfb::src2 the two resident
    // input shards. dfb::untilized_cache and dfb::untilized_cache2 are aliased -- the writer patches
    // the new row into the region published through the first and republishes it through the second,
    // which is what this kernel re-tilizes into dfb::out.
    compute_kernel_hw_startup(is_input1 ? dfb::src1 : dfb::src2, dfb::untilized_in);

    // Untilize input (single block, init only - no uninit needed).
    // Two instantiations because the source buffer is a template parameter: the branch is on the
    // runtime `is_input1`, and both are compiled into every node's binary. This is the legacy shape
    // unchanged -- only the buffer-index constants became binding tokens.
    if (!is_input1) {
        compute_kernel_lib::untilize<
            Wt,
            dfb::src2,
            dfb::untilized_in,
            compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    } else {
        compute_kernel_lib::untilize<
            Wt,
            dfb::src1,
            dfb::untilized_in,
            compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    }

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration from previous iteration
        compute_kernel_lib::untilize<Wt, dfb::cache, dfb::untilized_cache>(1);

        // Wait on writer to update block. Tilize with reconfiguration
        compute_kernel_lib::tilize<Wt, dfb::untilized_cache2, dfb::out>(1);
    }
}
