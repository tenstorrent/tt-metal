// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::ccl {

struct ExecuteAllBroadcast {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<uint32_t> num_links = 1,  // change to std::nullopt when we add support/test for links > 1 #30798
        std::optional<ttnn::ccl::Topology> topology =
            ttnn::ccl::Topology::Linear);  // change to std::nullopt when we add support/test for non-linear topologies
                                           // #30798
};

}  // namespace operations::ccl

constexpr auto all_broadcast =
    ttnn::register_operation<"ttnn::all_broadcast", ttnn::operations::ccl::ExecuteAllBroadcast>();

}  // namespace ttnn
