// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cpp/ttnn/operations/ccl/shared_with_host/snake_ring.hpp"

namespace ttnn::ring_attention_all_gather {

// 22 from main's CompileTimeArg enum, +1 for the trace-safe has_slot_metadata flag this branch appends
// as the last reader arg. Must stay equal to the reader kernel's kNumFixedCompileTimeArgs sentinel.
constexpr uint32_t kReaderFixedCompileTimeArgCount = 23;
constexpr uint32_t kWriterFixedCompileTimeArgCount = 24;

// Rank order of the tensor shards: 0 = axis ring (transport rank is the tensor rank), 1 = full-mesh
// snake with row-major tensor ranks, 2 = full-mesh snake with column-major tensor ranks (mesh axis 1
// outermost, e.g. SP-major shards with SP on axis 1). A bool argument maps onto 0 / 1.
template <uint32_t FullMeshOrder>
constexpr uint32_t tensor_rank_from_transport_rank(
    uint32_t transport_rank, uint32_t mesh_rows, uint32_t mesh_cols, ttnn::ccl::snake_ring::Orientation orientation) {
    if constexpr (FullMeshOrder == 2) {
        return ttnn::ccl::snake_ring::column_major_index(transport_rank, mesh_rows, mesh_cols, orientation);
    } else if constexpr (FullMeshOrder == 1) {
        return ttnn::ccl::snake_ring::row_major_index(transport_rank, mesh_rows, mesh_cols, orientation);
    } else {
        return transport_rank;
    }
}

}  // namespace ttnn::ring_attention_all_gather
