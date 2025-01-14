// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::speculative_execution {

struct ExecuteConsolidateCache {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& priority_tensor,
        const ttnn::Tensor& other_priority_tensor,
        const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
        uint32_t num_links);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& priority_tensor,
        const ttnn::Tensor& other_priority_tensor,
        const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
        uint32_t num_links);
};

}  // namespace operations::experimental::speculative_execution

namespace experimental {

constexpr auto consolidate_cache = ttnn::register_operation<
    "ttnn::experimental::consolidate_cache",
    ttnn::operations::experimental::speculative_execution::ExecuteConsolidateCache>();

}  // namespace experimental

}  // namespace ttnn
