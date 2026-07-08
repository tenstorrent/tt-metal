// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative compile test (G2 / SA-04): a Row/Col broadcast index requires a non-streaming policy
// (the caller must stage all broadcast tiles upfront; a streaming front-advance breaks the re-read).
// MUST fail to compile with "non-streaming policy".

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
        CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input, OperandKind::Row>{},
        PackTile<cb_out>{});
}
