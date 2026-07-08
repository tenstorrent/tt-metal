// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative compile test (G2 / SA-03): Block index mode with a Streaming lifecycle is illegal.
//
// A Block walker reads the absolute CB-front index `base + i`; Streaming pops per tile, so the
// front shifts every iter and absolute indexing reads the wrong tile. is_legal_kind_lifecycle
// (eltwise_chain.hpp:266-289) rejects (Block, Streaming) and CopyTile static_asserts on it.
// This kernel MUST fail to compile with "CopyTile: (IndexMode, Policy) is illegal for Block".

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input, OperandKind::Block>{},
        PackTile<cb_out>{});
}
