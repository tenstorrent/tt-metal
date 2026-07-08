// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Operand broadcast suite (G3 / OK-03, OK-04, OK-05).
//
// A + B with a hardware broadcast on the B operand. BroadcastDim controls the INTRA-tile
// replication (mirrors ckernel::BroadcastType: COL=1, ROW=2, SCALAR=3):
//   ROW    -> replicate B's row 0 down all rows   -> out[r][c] = A[r][c] + B[0][c]
//   COL    -> replicate B's col 0 across all cols -> out[r][c] = A[r][c] + B[r][0]
//   SCALAR -> replicate B's element [0][0]        -> out[r][c] = A[r][c] + B[0][0]
//
// The caller owns the big init (init_bcast), exactly like the migrated bcast_h / bcast_w
// kernels — the chain owns only per-element work. dim is a compile-time arg so init_bcast
// and BinaryFpu stay in lockstep. A random B makes ROW vs COL produce different results, so
// a broadcast-axis swap fails PCC instead of accidentally matching.

#include <cstdint>
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t dim = get_compile_time_arg_val(1);  // 1=Col, 2=Row, 3=Scalar (ckernel values)

    using namespace compute_kernel_lib;

    if constexpr (dim == 2) {
        init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(cb_a, cb_b, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::Row,
                InputLifecycle::Streaming,
                InputLifecycle::Streaming,
                BinaryDataFormatReconfig::None>{},
            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
    } else if constexpr (dim == 1) {
        init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::COL>(cb_a, cb_b, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::Streaming,
                BinaryDataFormatReconfig::None>{},
            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
    } else {  // dim == 3 -> Scalar
        init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::SCALAR>(cb_a, cb_b, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(n),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::Scalar,
                InputLifecycle::Streaming,
                InputLifecycle::Streaming,
                BinaryDataFormatReconfig::None>{},
            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
    }
}
