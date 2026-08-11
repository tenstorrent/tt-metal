// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cpp/ttnn/operations/ccl/shared_with_host/snake_ring.hpp"

namespace ttnn::ring_attention_all_gather {

constexpr uint32_t kReaderFixedCompileTimeArgCount = 18;
constexpr uint32_t kWriterFixedCompileTimeArgCount = 23;

template <bool FullMesh>
constexpr uint32_t tensor_rank_from_transport_rank(
    uint32_t transport_rank, uint32_t mesh_rows, uint32_t mesh_cols, ttnn::ccl::snake_ring::Orientation orientation) {
    if constexpr (FullMesh) {
        return ttnn::ccl::snake_ring::row_major_index(transport_rank, mesh_rows, mesh_cols, orientation);
    } else {
        return transport_rank;
    }
}

}  // namespace ttnn::ring_attention_all_gather
