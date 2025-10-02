// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file all_gather.cpp
 * @brief This file contains the C++ entry-points for all_gather operation.
 */

#include "all_gather.hpp"

namespace ttnn::operations::ccl2 {

ttnn::Tensor ExecuteAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const ttnn::ccl2::Topology topology,
    const std::optional<ttnn::MemoryConfig>& output_memory_config,
    const std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    // Invoke the host program
    return tt::tt_metal::operation::run(
               ttnn::AllGather{dim, topology, output_memory_config, subdevice_id}, {input_tensor})
        .at(0);
}

}  // namespace ttnn::operations::ccl2
