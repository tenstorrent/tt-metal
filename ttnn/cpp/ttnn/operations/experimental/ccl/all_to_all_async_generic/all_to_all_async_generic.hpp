// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllToAllAsyncGeneric {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<Tensor>& persistent_output_buffer,
        int32_t in_dim,
        int32_t out_dim,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_to_all_async_generic = ttnn::register_operation<
    "ttnn::experimental::all_to_all_async_generic",
    ttnn::operations::experimental::ccl::ExecuteAllToAllAsyncGeneric>();

}  // namespace experimental
}  // namespace ttnn
