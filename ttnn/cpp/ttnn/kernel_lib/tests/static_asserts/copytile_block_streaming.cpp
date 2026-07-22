// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: Block index mode with a Streaming lifecycle is illegal — a Block walker
// reads an absolute CB-front index, but Streaming pops per tile so the front shifts and the index
// reads the wrong tile. is_legal_kind_lifecycle rejects (Block, Streaming).
// MUST fail to compile with "CopyTile: (IndexMode, Policy) is illegal for Block".

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
        CopyTile<input(cb_in, InputLifecycle::Streaming, OperandKind::Block), Dst::D0>{},
        PackTile<output(cb_out)>{});
}
