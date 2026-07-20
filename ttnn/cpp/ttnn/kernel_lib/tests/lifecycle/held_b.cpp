// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Lifecycle / CB-synchronization suite.
//
// out[i] = A[i] + B[0] over n tiles: A streams (chain owns wait+pop); B is a single held tile reused
// every iter on a selectable InputLifecycle. The chain emits exactly the CB edges its lifecycle
// declares; this kernel supplies the rest. A miscount hangs (wait/pop never satisfied) or reads a
// stale tile, so each case asserts BOTH no-hang and correct values.
//
//   life  lifecycle        chain emits (for B)         caller supplies
//   0     Bulk             wait 1 upfront + pop 1 end   nothing
//   1     HeldBulk         wait 1 upfront, no pop       pop 1 after
//   2     HeldStream       wait 1 / iter, no pop        pop 1 after
//   3     CallerManaged    nothing                      wait 1 before, pop 1 after
//   4     DeferredPop      no wait, pop 1 at end        wait 1 before
//
// B uses OperandKind::Scalar (re-read at relative tile 0 each iter) — the held-operand pattern.

#include <cstdint>
// eltwise_chain.hpp first: it pulls in the compute API (PACK/UNPACK macros + llk decls) that
// circular_buffer.h depends on. Reversing the order fails to compile.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t life = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    CircularBuffer cb_b_obj(cb_b);

    // A: streaming. B: held single tile (Scalar index, relative tile 0). Output: streaming.
    auto pack = PackTile<cb_out, output(OutputLifecycle::Streaming, DataFormatReconfig::Disabled)>{};

    if constexpr (life == 0) {  // Bulk — chain owns both edges
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::None,
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(InputLifecycle::Bulk, DataFormatReconfig::Disabled)>{},
            pack);
    } else if constexpr (life == 1) {  // HeldBulk — chain waits upfront, caller pops after
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::None,
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(InputLifecycle::HeldBulk, DataFormatReconfig::Disabled)>{},
            pack);
        cb_b_obj.pop_front(1);
    } else if constexpr (life == 2) {  // HeldStream — chain waits per-iter, caller pops after
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::None,
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(InputLifecycle::HeldStream, DataFormatReconfig::Disabled)>{},
            pack);
        cb_b_obj.pop_front(1);
    } else if constexpr (life == 3) {  // CallerManaged — chain emits nothing for B
        cb_b_obj.wait_front(1);
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::None,
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(InputLifecycle::CallerManaged, DataFormatReconfig::Disabled)>{},
            pack);
        cb_b_obj.pop_front(1);
    } else {  // life == 4: DeferredPop — caller waits before, chain pops at end
        cb_b_obj.wait_front(1);
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::None,
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(InputLifecycle::DeferredPop, DataFormatReconfig::Disabled)>{},
            pack);
    }
}
