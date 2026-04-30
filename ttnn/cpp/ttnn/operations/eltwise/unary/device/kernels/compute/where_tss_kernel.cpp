// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
template <compute_kernel_lib::Dst Slot>
struct FillScalarOp : compute_kernel_lib::UnaryOp<FillScalarOp<Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;
    uint32_t value;

    ALWI static void init() { ckernel::fill_tile_init(); }
    ALWI void call(uint32_t dst) const {
#if defined(INP_INT32) || defined(INP_UINT32)
        ckernel::fill_tile_int<DataFormat::Int32>(dst, value);
#endif
#if defined(INP_FLOAT) || defined(INP_FLOAT32)
        const auto fval = reinterpret_cast<const float*>(&value);
        ckernel::fill_tile(dst, *fval);
#endif
    }
};

struct WhereChainMacroOp : compute_kernel_lib::UnaryOp<WhereChainMacroOp, compute_kernel_lib::Dst::D0> {
    // SFPU_OP_CHAIN_0 macro emits init+exec for where; conservatively LUT clobber.
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
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    ckernel::compute_kernel_hw_startup(cb_input, cb_output);
    ckernel::init_sfpu(cb_input, cb_output);

#ifndef SFPU_OP_CHAIN_0
#error "where_tss_kernel requires SFPU_OP_CHAIN_0 to be defined via get_block_defines"
#endif

    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

    eltwise_pipeline<cb_output>(
        num_tiles,
        eltwise_chain(
            CopyTile<cb_input, Dst::D0>{},
            FillScalarOp<Dst::D1>{{}, packed_scalar1},
            FillScalarOp<Dst::D2>{{}, packed_scalar2},
            WhereChainMacroOp{}));
}
