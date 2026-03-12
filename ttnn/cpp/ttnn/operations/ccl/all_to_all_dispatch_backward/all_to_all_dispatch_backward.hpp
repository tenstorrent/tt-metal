// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::ccl {

struct ExecuteAllToAllDispatchBackward {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& grad_output,
        const ttnn::Tensor& expert_mapping_tensor,
        const ttnn::Tensor& expert_metadata_tensor,
        std::optional<uint32_t> num_links = std::nullopt,
        std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<uint32_t>& axis = std::nullopt,
        const std::optional<uint32_t>& output_shard_dim = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace operations::ccl

constexpr auto all_to_all_dispatch_backward =
    ttnn::register_operation<"ttnn::all_to_all_dispatch_backward", ttnn::operations::ccl::ExecuteAllToAllDispatchBackward>();

}  // namespace ttnn
