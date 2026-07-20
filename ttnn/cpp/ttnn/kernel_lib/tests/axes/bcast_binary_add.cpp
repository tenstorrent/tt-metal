// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Operand broadcast suite: A + B with a hardware broadcast on B. BroadcastDim controls the
// INTRA-tile replication (mirrors ckernel::BroadcastType: COL=1, ROW=2, SCALAR=3):
//   ROW    -> B row 0 down all rows   -> out[r][c] = A[r][c] + B[0][c]
//   COL    -> B col 0 across all cols -> out[r][c] = A[r][c] + B[r][0]
//   SCALAR -> B element [0][0]        -> out[r][c] = A[r][c] + B[0][0]
// The caller does only the dim-agnostic hw init (compute_kernel_hw_startup); the chain emits the
// broadcast init AND compute itself from BroadcastDim. A random B makes ROW vs COL differ, so an
// axis swap fails PCC.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t dim = get_compile_time_arg_val(1);  // 1=Col, 2=Row, 3=Scalar (ckernel values)

    using namespace compute_kernel_lib;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    if constexpr (dim == 2) {
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::Row,
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled)>{},
            PackTile<cb_out, output(OutputLifecycle::Streaming, DataFormatReconfig::Disabled)>{});
    } else if constexpr (dim == 1) {
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::Col,
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled)>{},
            PackTile<cb_out, output(OutputLifecycle::Streaming, DataFormatReconfig::Disabled)>{});
    } else {  // dim == 3 -> Scalar
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::Scalar,
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled),
                input(InputLifecycle::Streaming, DataFormatReconfig::Disabled)>{},
            PackTile<cb_out, output(OutputLifecycle::Streaming, DataFormatReconfig::Disabled)>{});
    }
}
