// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

ttnn::Tensor all_gather(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn::experimental
