// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Out-of-bounds DST probe: identity copy through a single DEST slot (slot index is a CT arg).
//
// Every chain element static_asserts `to_u32(slot) < DEST_AUTO_LIMIT`, which depends on the compute
// config (half/full sync × fp32_dest_acc): half-sync bf16=8/fp32=4, full-sync bf16=16/fp32=8. The
// driving pytest picks (slot, sync, fp32_acc) so a legal slot compiles and reproduces the input,
// and an over-limit slot FAILS to compile with "DEST slot exceeds DEST_AUTO_LIMIT".

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t slot = get_compile_time_arg_val(1);
    constexpr auto Slot = static_cast<compute_kernel_lib::Dst>(slot);

    compute_kernel_hw_startup(cb_in, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(total_tiles),
        compute_kernel_lib::CopyTile<cb_in, Slot>{},
        compute_kernel_lib::PackTile<
            cb_out,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::Output,
            Slot>{});
}
