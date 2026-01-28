// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

std::pair<std::optional<tt::tt_metal::Tensor>, uint32_t> create_reciprocal_tensor_if_needed(
    tt::tt_metal::IDevice* device, uint32_t W, const tt::tt_metal::CoreRangeSet& cores, bool use_welford);

}  // namespace ttnn::prim
