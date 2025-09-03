// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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
    const ttnn::Tensor& input_tensors,
    int32_t dim,
    uint32_t output_dim_offset,
    uint32_t output_dim_shape,
    uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const MeshDevice& mesh_device,
    std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology) {
    return ttnn::operations::experimental::ccl::slice_reshard_async(
        input_tensors,
        dim,
        output_dim_offset,
        output_dim_shape,
        cluster_axis,
        final_semaphore,
        barrier_semaphore,
        mesh_device,
        num_preferred_links,
        memory_config,
        topology);
}

}  // namespace ttnn::operations::experimental::ccl
