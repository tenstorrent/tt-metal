// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace ccl {
namespace transformer {

struct ExecuteAllReduceAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& buffer_tensor,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        const std::optional<size_t> num_preferred_links,
        std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt);
};

}  // namespace transformer
}  // namespace ccl
}  // namespace experimental
}  // namespace operations

namespace experimental {

constexpr auto all_reduce_create_qkv_heads = ttnn::register_operation<
    "ttnn::experimental::all_reduce_create_qkv_heads",
    ttnn::operations::experimental::ccl::transformer::ExecuteAllReduceAsync>();

}  // namespace experimental
}  // namespace ttnn
