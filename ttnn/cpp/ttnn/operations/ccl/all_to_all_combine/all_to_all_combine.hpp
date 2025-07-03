// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::ccl {

struct ExecuteAllToAllCombine {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_mapping_tensor,
        const ttnn::Tensor& expert_metadata_tensor,
        const GlobalSemaphore& global_semaphore,
        uint32_t num_links = 1,
        tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<uint32_t>& axis = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace operations::ccl

constexpr auto all_to_all_combine =
    ttnn::register_operation<"ttnn::all_to_all_combine", ttnn::operations::ccl::ExecuteAllToAllCombine>();

}  // namespace ttnn
