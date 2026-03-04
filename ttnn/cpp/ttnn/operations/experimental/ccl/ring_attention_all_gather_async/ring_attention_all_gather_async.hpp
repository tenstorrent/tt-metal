// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteRingAttentionAllGatherAsync {
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        std::vector<ttnn::Tensor>& persistent_output_buffer,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::ccl::Topology topology,
        uint32_t num_links = 1,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        CoreCoord core_grid_offset = CoreCoord(0, 0),
        ttnn::ccl::CoreAllocationStrategy core_allocation_strategy =
            ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto ring_attention_all_gather_async = ttnn::register_operation<
    "ttnn::experimental::ring_attention_all_gather_async",
    ttnn::operations::experimental::ccl::ExecuteRingAttentionAllGatherAsync>();

}  // namespace experimental
}  // namespace ttnn
