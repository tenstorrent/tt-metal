// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteSplitScatter {
    static ttnn::Tensor invoke(
        ttnn::Tensor& input_tensor,
        const int32_t dim,
        const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<SubDeviceId> subdevice_id = std::nullopt,
        bool enable_persistent_fabric_mode = false);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto split_scatter = ttnn::
    register_operation<"ttnn::experimental::split_scatter", ttnn::operations::experimental::ccl::ExecuteSplitScatter>();

}  // namespace experimental
}  // namespace ttnn
