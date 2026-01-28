// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::prim {
struct ManualSeedParams {
    tt::tt_metal::distributed::MeshDevice* device = nullptr;
    std::optional<uint32_t> seeds = std::nullopt;
    std::optional<uint32_t> user_ids = std::nullopt;
    std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

struct ManualSeedInputs {
    std::optional<Tensor> seeds = std::nullopt;
    std::optional<Tensor> user_ids = std::nullopt;
};

}  // namespace ttnn::prim
