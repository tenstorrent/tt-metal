// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swap_tensor_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/swap_tensor_async/device/swap_tensor_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteSwapTensorAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::swap_tensor_async(
        input_tensor, multi_device_global_semaphore, num_links, memory_config, topology, subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
