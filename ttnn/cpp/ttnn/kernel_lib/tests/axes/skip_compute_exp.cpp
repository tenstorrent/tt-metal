// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Skip-compute performance-analysis knob.
//
// Identical to a streaming exp(x) chain, except the chain runs in skip-compute mode: it emits ONLY
// the CB lifecycle (wait/pop on the input, reserve/push on the output) plus the tile_regs dst-sync
// window, and skips ALL init + reconfig + compute (unpack, exp_tile, pack_tile). The CB counts are
// byte-for-byte identical to a normal run, so the reader/writer handshake is intact and the kernel
// does NOT hang — it just pushes uninitialized tiles. Used to measure pure CB/sync overhead.
//
// Skip is a build macro, not part of the eltwise_chain API: this kernel opts the
// CKL_ELTWISE_CHAIN_SKIP_COMPUTE macro in to 1 before including the header, and the chain reads it
// internally (constexpr in eltwise_chain_impl). The call site is unchanged from the Run twin
// (hoist_single_call.cpp). A build `-DCKL_ELTWISE_CHAIN_SKIP_COMPUTE=0` would flip it back to Run.

// Skip profiling twin: default the build macro to 1 (overridable by a -D on the build).
#ifndef CKL_ELTWISE_CHAIN_SKIP_COMPUTE
#define CKL_ELTWISE_CHAIN_SKIP_COMPUTE 1
#endif

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    // Skip-compute is a build macro, not a call-site arg: CKL_ELTWISE_CHAIN_SKIP_COMPUTE (=1 above)
    // makes this chain emit only the CB lifecycle + tile_regs window and elide init + compute. The
    // call site is byte-identical to the Run twin (hoist_single_call.cpp).
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming>{},
        Exp<>{},
        PackTile<cb_out, OutputLifecycle::Streaming>{});
}
