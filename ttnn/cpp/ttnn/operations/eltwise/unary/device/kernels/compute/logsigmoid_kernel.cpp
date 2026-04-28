// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_output = tt::CBIndex::c_2;

    using namespace compute_kernel_lib::eltwise;

    init_sfpu(cb_input, cb_output);

    // logsigmoid(x): D0 holds x, D1 holds exp(-x).
    // Logsigmoid<D0, D1, D0> writes -log(1 + exp(-x)) into D0 in a single
    // numerically-stable LLK call.
    auto chain = eltwise_chain(
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>{},
        Neg<Dst::D1>{},
        Exp<Approx::Fast, Approx::Fast, Dst::D1>{},
        Logsigmoid<Dst::D0, Dst::D1, Dst::D0>{});
    eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
}
