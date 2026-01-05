// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_reshard_async.hpp"

#include <utility>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_device_operation.hpp"

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
    tt::tt_fabric::Topology usable_topology = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    return ttnn::prim::slice_reshard_async(
        input_tensor,
        dim,
        output_dim_offset,
        output_dim_shape,
        cluster_axis,
        final_semaphore,
        barrier_semaphore,
        num_preferred_links.value_or(1),
        memory_config.value_or(input_tensor.memory_config()),
        usable_topology);
}

}  // namespace ttnn::operations::experimental::ccl
