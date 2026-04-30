// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// V2 eltwise helper validation kernel — single-input streaming Exp via
// eltwise_pipeline + eltwise_chain(CopyTile, Exp).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Hardware init must come first — must be called exactly once.
    ckernel::compute_kernel_hw_startup(cb_in, cb_out);
    // SFPU init: programs unpacker for cb_in → DEST and packer for cb_out.
    ckernel::init_sfpu(cb_in, cb_out);

    compute_kernel_lib::eltwise_pipeline<cb_out>(
        num_tiles,
        compute_kernel_lib::eltwise_chain(compute_kernel_lib::CopyTile<cb_in>{}, compute_kernel_lib::Exp<>{}));
}
