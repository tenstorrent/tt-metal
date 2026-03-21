// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum.hpp"
#include "device/offset_cumsum_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum {

std::array<ttnn::Tensor, 2> offset_cumsum(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_links,
    const ttnn::MemoryConfig& memory_config) {
    const auto& shape = input_tensor.logical_shape();
    uint32_t n_routed_experts = shape[-1];

    auto reshaped = ttnn::reshape(input_tensor, ttnn::Shape({1, n_routed_experts}));

    auto gathered = ttnn::all_gather(
        reshaped,
        /*dim=*/0,
        /*cluster_axis=*/cluster_axis,
        /*subdevice_id=*/std::nullopt,
        /*memory_config=*/memory_config,
        /*optional_output_tensor=*/std::nullopt,
        /*num_links=*/num_links);

    auto row_major = ttnn::to_layout(gathered, tt::tt_metal::Layout::ROW_MAJOR, std::nullopt, std::nullopt);

    return ttnn::prim::offset_cumsum(row_major);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum
