// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::ccl {

struct ExecuteAllReduce {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<uint32_t> num_links = std::nullopt,
        std::optional<tt::tt_fabric::Topology> topology = std::nullopt);
};

}  // namespace operations::ccl

constexpr auto all_reduce = ttnn::register_operation<"ttnn::all_reduce", ttnn::operations::ccl::ExecuteAllReduce>();

}  // namespace ttnn
