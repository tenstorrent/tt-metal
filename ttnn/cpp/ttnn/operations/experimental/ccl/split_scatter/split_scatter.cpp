// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_scatter.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/split_scatter/device/split_scatter_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteSplitScatter::invoke(
    ttnn::Tensor& input_tensor,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode) {
    return ttnn::operations::experimental::ccl::split_scatter(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        enable_persistent_fabric_mode);
}

}  // namespace ttnn::operations::experimental::ccl
