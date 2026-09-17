// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_topology.hpp"
#include "device/chronological_topology_device_operation.hpp"
#include "ttnn/device_operation.hpp"
namespace ttnn::experimental::kda {
Tensor chronological_topology(
    const Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t batch_heads,
    uint32_t key_dim,
    uint32_t value_dim) {
    return ttnn::device_operation::launch<prim::ChronologyOperation>(
        prim::ChronologyParams{sequence_parallel_axis, local_rows, batch_heads, key_dim, value_dim},
        prim::ChronologyInputs{actual_start})[0];
}
}  // namespace ttnn::experimental::kda
