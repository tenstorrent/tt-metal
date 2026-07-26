// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& output_tensor,
    uint32_t cluster_axis,
    const GlobalSemaphore& data_valid_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather

namespace ttnn::experimental::deepseek_prefill {
using operations::experimental::deepseek_prefill::high_bw_all_gather::high_bw_all_gather;
}  // namespace ttnn::experimental::deepseek_prefill
