// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Output-lifecycle / CB-synchronization suite (G1 output side).
//
// Identity copy out[i] = A[i] over n tiles. Input streams (chain owns). The OUTPUT PackTile uses a
// selectable OutputLifecycle; the chain emits the reserve/push edges its lifecycle declares and the
// compute kernel supplies whatever the chain does NOT. A reserve/push miscount hangs the writer
// (BRISC) or overwrites an unpushed tile — so the test asserts no-hang AND correct values.
//
//   life  OutputLifecycle       chain emits                         caller supplies
//   0     Streaming             reserve 1 + push 1 / iter           nothing
//   1     Bulk                  reserve n upfront + push n at end    nothing
//   2     BulkReservePerTile    reserve n upfront, push 1 / iter     nothing   (SDPA reduce_c)
//   3     CallerManaged         pack only (no reserve / no push)     reserve n before, push n after (tt-train L1 accum)

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t life = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    CircularBuffer cb_out_obj(cb_out);
    auto in = CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{};

    if constexpr (life == 0) {
        eltwise_chain(
            EltwiseShape::tiles(n), in, PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
    } else if constexpr (life == 1) {
        eltwise_chain(EltwiseShape::tiles(n), in, PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::None>{});
    } else if constexpr (life == 2) {
        eltwise_chain(
            EltwiseShape::tiles(n),
            in,
            PackTile<cb_out, OutputLifecycle::BulkReservePerTile, PackTileReconfig::None>{});
    } else {  // life == 3: CallerManaged — chain packs only, caller brackets reserve+push
        cb_out_obj.reserve_back(n);
        eltwise_chain(
            EltwiseShape::tiles(n), in, PackTile<cb_out, OutputLifecycle::CallerManaged, PackTileReconfig::None>{});
        cb_out_obj.push_back(n);
    }
}
