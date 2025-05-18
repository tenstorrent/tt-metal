// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteSwapTensorAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& priority_tensor_a,
        const ttnn::Tensor& priority_tensor_b,
        const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto swap_tensor_async = ttnn::register_operation<
    "ttnn::experimental::swap_tensor_async",
    ttnn::operations::experimental::ccl::ExecuteSwapTensorAsync>();

}  // namespace experimental
}  // namespace ttnn
