// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async.hpp"
#include <utility>
#include "ttnn/operations/ccl/mesh_partition/device/mesh_partition_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteReduceScatterMinimalAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    log_debug(tt::LogOp, "DEBUG: using reduce_scatter_minimal_async");
    if (composite_common::use_composite_reduce_scatter(input_tensor, dim, cluster_axis)) {
        log_debug(tt::LogOp, "DEBUG: using composite_reduce_scatter");
        return composite_common::composite_reduce_scatter(
            input_tensor, dim, num_links, memory_config, subdevice_id, cluster_axis);
    } else {
        log_debug(tt::LogOp, "DEBUG: using reduce_scatter_minimal_async");
        return ttnn::operations::experimental::ccl::reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore,
            num_links,
            memory_config,
            intermediate_memory_config,
            topology_,
            subdevice_id,
            cluster_axis,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel);
    }
}
}  // namespace ttnn::operations::experimental::ccl
