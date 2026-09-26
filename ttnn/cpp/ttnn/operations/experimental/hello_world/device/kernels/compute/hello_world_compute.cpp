// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // ckl::copy (the pure-copy one-liner)

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr auto dfb_input_id = tt::CBIndex::c_0;
    constexpr auto dfb_output_id = tt::CBIndex::c_2;

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    // The MATH (compute) trisc has no built-in core-coordinate getter; the factory
    // passes this core's coordinates as runtime args so every placed core can DPRINT
    // its own identity.
    const uint32_t core_x = get_arg_val<uint32_t>(1);
    const uint32_t core_y = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(dfb_input_id, dfb_output_id);

    // DPRINT_MATH (not the generic DPRINT) so it fires only on the MATH trisc: a compute
    // kernel's kernel_main runs on all three tensix triscs, and the generic DPRINT would
    // emit three identical lines per core. One line per core is the cleaner onboarding signal.
    DPRINT_MATH("Hello, world! I am core ({}, {}) and I process {} tile(s).\n", core_x, core_y, num_tiles);

    // Pure copy: CopyTile(D0) -> PackTile(D0). The data passes through the FPU untouched.
    // The input/output specs are template arguments; only the iteration shape is a function argument.
    ckl::copy<
        ckl::input(dfb_input_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::output(
            dfb_output_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::tiles(num_tiles));
}
