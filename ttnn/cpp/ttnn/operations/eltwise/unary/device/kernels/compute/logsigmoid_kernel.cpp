// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/logsigmoid.h"  // logsigmoid_tile{,_init}
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"   // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"   // Negative

namespace {

// DEST-DEST binary SFPU op for `logsigmoid_tile(in0, in1, out)`.
// logsigmoid(x) = -log(1 + exp(-x)); the LLK kernel takes in0=x, in1=exp(-x).
template <
    compute_kernel_lib::Dst In0 = compute_kernel_lib::Dst::D0,
    compute_kernel_lib::Dst In1 = compute_kernel_lib::Dst::D1,
    compute_kernel_lib::Dst Out = compute_kernel_lib::Dst::D0>
struct LogSigmoidBinary
    : compute_kernel_lib::BinaryOp<LogSigmoidBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { logsigmoid_tile_init(); }
    static ALWI void call(uint32_t i0, uint32_t i1, uint32_t o) { logsigmoid_tile(i0, i1, o); }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_input, cb_input, cb_output);

    // logsigmoid(x):
    //   D0 = x; D1 = x; D1 = -D1; D1 = exp(D1) (fast); logsigmoid(D0, D1) -> D0
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>{},
        Negative<Dst::D1>{},
        Exp<Approx::Fast, Approx::Exact, Dst::D1>{},
        LogSigmoidBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
