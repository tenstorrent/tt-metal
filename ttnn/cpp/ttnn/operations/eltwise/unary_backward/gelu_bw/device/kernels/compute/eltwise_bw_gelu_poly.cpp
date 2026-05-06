// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/gelu.h"  // gelu_derivative_tile{,_init}
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace {

// Local SFPU op-struct for `gelu_derivative_tile<false>`. Not yet covered by
// `eltwise_special.hpp`; inline here to keep the migration self-contained.
template <compute_kernel_lib::Dst Slot = compute_kernel_lib::Dst::D0>
struct GeluDerivative
    : compute_kernel_lib::UnaryOp<GeluDerivative<Slot>, Slot> {
    static ALWI void init() { gelu_derivative_tile_init<false>(); }
    static ALWI void call(uint32_t idst) { gelu_derivative_tile<false>(idst); }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = per_core_block_cnt * per_core_block_size;

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    // grad_in[i] = grad_out[i] * GELU'(input[i])
    using Chain = EltwiseChain<
        CopyTile<cb_grad_out, Dst::D0, CopyTilePolicy::WaitAndPop>,
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::WaitAndPop>,
        GeluDerivative<Dst::D1>,
        MulBinary<Dst::D0, Dst::D1, Dst::D0>,
        PackTile<cb_grad_in, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_grad_out, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::WaitAndPop>{},
        GeluDerivative<Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<cb_grad_in, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
