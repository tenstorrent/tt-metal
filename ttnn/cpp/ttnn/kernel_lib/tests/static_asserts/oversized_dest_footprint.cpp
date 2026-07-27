// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: the chain must reject an element whose declared DEST
// footprint cannot fit even one lane. Without the central guard max_block is zero,
// and the runtime walk cannot make forward progress.
// MUST fail to compile with "DEST footprint exceeds the available DEST capacity".

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

struct OversizedDestElement : DestOnlyTag {
    static constexpr uint32_t lane_width = DEST_AUTO_LIMIT + 1;

    static ALWI void init() {}
    ALWI void exec(uint32_t, uint32_t) const {}
};

}  // namespace compute_kernel_lib

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(EltwiseShape::single(), OversizedDestElement{});
}
