// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "experimental/kernel_args.h"
#include "dram_read.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in_obj(dfb::in);
    DataflowBuffer dfb_out_obj(dfb::out);
    compute_kernel_hw_startup(dfb::in, dfb::out);

    dram_read::for_each_chunk(num_tiles, [&](uint32_t count) {
        compute_kernel_lib::copy<
            compute_kernel_lib::input(
                dfb::in,
                compute_kernel_lib::WaitPolicy::PerTile,
                compute_kernel_lib::PopPolicy::PerTile,
                compute_kernel_lib::DataFormatReconfig::Disabled),
            compute_kernel_lib::output(
                dfb::out,
                compute_kernel_lib::ReservePolicy::PerTile,
                compute_kernel_lib::PushPolicy::PerTile,
                compute_kernel_lib::DataFormatReconfig::Disabled)>(
            compute_kernel_lib::IterationShape::tiles(count), dfb_in_obj, dfb_out_obj);
    });
}
