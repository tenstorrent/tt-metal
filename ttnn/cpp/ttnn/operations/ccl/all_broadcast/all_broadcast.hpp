// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

std::vector<ttnn::Tensor> all_broadcast(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<uint32_t> num_links = 1,  // change to std::nullopt when we add support/test for links > 1 #30798
    std::optional<ttnn::ccl::Topology> topology =
        ttnn::ccl::Topology::Linear);  // change to std::nullopt when we add support/test for non-linear topologies
                                       // #30798

}  // namespace ttnn
