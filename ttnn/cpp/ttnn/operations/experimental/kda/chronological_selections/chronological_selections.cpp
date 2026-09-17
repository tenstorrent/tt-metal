// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_selections.hpp"
#include "device/chronological_selections_device_operation.hpp"
#include "ttnn/device_operation.hpp"
namespace ttnn::experimental::kda {
Tensor chronological_selections(
    const Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t batch_heads,
    uint32_t key_dim,
    uint32_t value_dim, const std::optional<Tensor>& actual_end) {
    return ttnn::device_operation::launch<prim::ChronologicalSelectionsOperation>(
        prim::ChronologicalSelectionsParams{sequence_parallel_axis, local_rows, batch_heads, key_dim, value_dim},
        prim::ChronologicalSelectionsInputs{actual_start, actual_end})[0];
}
}  // namespace ttnn::experimental::kda
