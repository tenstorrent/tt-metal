// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

ttnn::Tensor all_to_all_async_generic(
    const ttnn::Tensor& input_tensor,
    int32_t in_dim,
    int32_t out_dim,
    const std::optional<Tensor>& persistent_output_buffer = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::ccl::Topology> topology = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental
