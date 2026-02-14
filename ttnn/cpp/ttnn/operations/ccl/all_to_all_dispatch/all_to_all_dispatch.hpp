// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::ccl {

std::array<ttnn::Tensor, 2> all_to_all_dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis = std::nullopt,
    const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<uint32_t>& output_concat_dim = std::nullopt);

}  // namespace operations::ccl

// Export to ttnn namespace
using operations::ccl::all_to_all_dispatch;

}  // namespace ttnn
