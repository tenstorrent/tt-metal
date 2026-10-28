// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum.hpp"
#include "device/offset_cumsum_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum {

std::array<ttnn::Tensor, 4> offset_cumsum(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t experts_per_chip,
    const ttnn::MemoryConfig& memory_config,
    bool use_l1_small_for_semaphores) {
    const auto& shape = input_tensor.logical_shape();
    uint32_t n_routed_experts = shape[-1];

    auto reshaped = ttnn::reshape(input_tensor, ttnn::Shape({1, n_routed_experts}));

    // TODO(nuked-op all_gather): restore real call
    (void)num_links;
    (void)memory_config;
    (void)use_l1_small_for_semaphores;
    auto gathered = reshaped;

    auto row_major = ttnn::to_layout(gathered, tt::tt_metal::Layout::ROW_MAJOR, std::nullopt, std::nullopt);

    return ttnn::prim::offset_cumsum(row_major, cluster_axis, experts_per_chip);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum
