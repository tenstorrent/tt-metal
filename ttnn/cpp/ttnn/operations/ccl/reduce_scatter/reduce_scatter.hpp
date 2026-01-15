// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::ccl {

struct ExecuteReduceScatter {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& intermediate_memory_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
        std::optional<uint32_t> num_links = std::nullopt,
        std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt);
};

}  // namespace operations::ccl

constexpr auto reduce_scatter =
    ttnn::register_operation<"ttnn::reduce_scatter", ttnn::operations::ccl::ExecuteReduceScatter>();

}  // namespace ttnn
