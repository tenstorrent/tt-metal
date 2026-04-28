// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum.hpp"
#include "device/offset_cumsum_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/subgroup_gather_histograms/subgroup_gather_histograms.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum {

std::array<ttnn::Tensor, 3> offset_cumsum(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t experts_per_chip,
    const ttnn::MemoryConfig& memory_config,
    uint32_t num_dispatch_subgroups) {
    TT_FATAL(num_dispatch_subgroups >= 1, "num_dispatch_subgroups must be >= 1 (got {})", num_dispatch_subgroups);

    const auto& shape = input_tensor.logical_shape();
    uint32_t n_routed_experts = shape[-1];

    log_info(
        tt::LogOp,
        "offset_cumsum composite: input_shape={} cluster_axis={} num_links={} num_dispatch_subgroups={}",
        shape,
        cluster_axis,
        num_links,
        num_dispatch_subgroups);

    auto reshaped = ttnn::reshape(input_tensor, ttnn::Shape({1, n_routed_experts}));
    log_info(tt::LogOp, "offset_cumsum composite: reshaped_shape={}", reshaped.logical_shape());

    ttnn::Tensor gathered;
    if (num_dispatch_subgroups > 1) {
        // Subgroup-scoped gather — fabric traffic stays within each dispatch subgroup.
        // Output is [dispatch_group_size, n_routed_experts] holding only this subgroup's rows.
        gathered = ttnn::subgroup_gather_histograms(
            reshaped,
            /*cluster_axis=*/cluster_axis,
            /*num_dispatch_subgroups=*/num_dispatch_subgroups,
            /*num_links=*/num_links,
            /*memory_config=*/memory_config);
        log_info(
            tt::LogOp, "offset_cumsum composite: subgroup_gather_histograms output shape={}", gathered.logical_shape());
    } else {
        gathered = ttnn::all_gather(
            reshaped,
            /*dim=*/0,
            /*cluster_axis=*/cluster_axis,
            /*subdevice_id=*/std::nullopt,
            /*memory_config=*/memory_config,
            /*optional_output_tensor=*/std::nullopt,
            /*num_links=*/num_links);
        log_info(tt::LogOp, "offset_cumsum composite: all_gather output shape={}", gathered.logical_shape());
    }

    auto row_major = ttnn::to_layout(gathered, tt::tt_metal::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
    log_info(tt::LogOp, "offset_cumsum composite: row_major shape={}", row_major.logical_shape());

    // Both gather paths produce an [H, n_routed_experts] tensor where H == dispatch_group_size.
    // With that uniform contract, the prim cumsums over all H rows and writes the result at
    // row_idx = coord[cluster_axis] % H. We pass num_dispatch_subgroups so the program-cache
    // key differs for subgroup vs. non-subgroup runs (different input shapes).
    return ttnn::prim::offset_cumsum(row_major, cluster_axis, experts_per_chip, num_dispatch_subgroups);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum
