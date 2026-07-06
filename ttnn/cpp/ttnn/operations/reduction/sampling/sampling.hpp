// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn {

Tensor sampling(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& k,
    const Tensor& p,
    const Tensor& temp,
    const std::optional<uint32_t>& seed = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<tt::tt_metal::GlobalSemaphore>& war_semaphore = std::nullopt,
    const std::optional<tt::tt_metal::CoreCoord>& war_sem_drain_core = std::nullopt);

}  // namespace ttnn
