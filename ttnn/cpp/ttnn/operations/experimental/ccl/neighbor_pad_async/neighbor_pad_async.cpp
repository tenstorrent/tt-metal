// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/device/neighbor_pad_async_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteNeighborPadAsync::invoke(
    const ttnn::Tensor& input_tensors,
    int32_t dim,
    uint32_t padding,
    const std::string& padding_mode,
    bool direction,
    uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const MeshDevice& mesh_device,
    std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology) {
    return ttnn::operations::experimental::ccl::neighbor_pad_async(
        input_tensors,
        dim,
        padding,
        padding_mode,
        direction,
        cluster_axis,
        final_semaphore,
        barrier_semaphore,
        mesh_device,
        num_preferred_links,
        memory_config,
        topology);
}

}  // namespace ttnn::operations::experimental::ccl
