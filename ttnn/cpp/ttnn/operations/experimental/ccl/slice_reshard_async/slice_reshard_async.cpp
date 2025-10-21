// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_reshard_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteSliceReshardAsync::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    uint32_t output_dim_offset,
    uint32_t output_dim_shape,
    uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "slice_reshard_async op is only supported for Fast Dispatch");

    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    uint32_t num_devices;
    auto mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    // Use the mesh dimensions to determine the ring size
    num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    TT_FATAL(num_devices > 1, "slice_reshard_async op will only work for num_devices > 1, but has {}", num_devices);

    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    return tt::tt_metal::operation::run(
               ttnn::SliceReshardAsync(
                   devices,
                   dim,
                   output_dim_offset,
                   output_dim_shape,
                   cluster_axis,
                   final_semaphore,
                   barrier_semaphore,
                   num_preferred_links.value_or(1),
                   memory_config.value_or(input_tensor.memory_config()),
                   topology_,
                   num_devices),
               {input_tensor},
               {},
               {})
        .at(0);
}

}  // namespace ttnn::operations::experimental::ccl
