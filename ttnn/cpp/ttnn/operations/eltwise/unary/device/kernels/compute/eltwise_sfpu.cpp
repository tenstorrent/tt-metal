// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

// Kernel-local struct that wraps the program-factory-injected SFPU_OP_CHAIN_0
// macro. The macro expands to init+exec statements for the selected unary SFPU
// op (e.g. `exp_tile_init(); exp_tile(0);`). We put the entire macro in
// `call()` and keep `init()` empty — the chain combinator runs init() then
// call() per tile, and the macro carries both init+exec in one block.
namespace {
struct SfpuChainMacroOp : compute_kernel_lib::UnaryOp<SfpuChainMacroOp, compute_kernel_lib::Dst::D0> {
    // Conservative: the chained op may program SFPU LUT (exp/log/tanh/etc.).
    // This disables hoisting if combined in a multi-LUT chain, which is the
    // safe default for an unknown op selection.
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() {}
    ALWI static void call(uint32_t /*dst*/) {
#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif
    }
};
}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    ckernel::compute_kernel_hw_startup(cb_input, cb_output);
    ckernel::init_sfpu(cb_input, cb_output);

    compute_kernel_lib::eltwise_pipeline<cb_output>(
        num_tiles, compute_kernel_lib::eltwise_chain(compute_kernel_lib::CopyTile<cb_input>{}, SfpuChainMacroOp{}));
}
