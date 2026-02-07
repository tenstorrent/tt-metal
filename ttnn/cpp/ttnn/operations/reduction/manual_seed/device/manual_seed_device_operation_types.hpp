// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <tuple>

namespace ttnn::prim {
struct ManualSeedParams {
    tt::tt_metal::distributed::MeshDevice* device = nullptr;
    std::optional<uint32_t> seeds = std::nullopt;
    std::optional<uint32_t> user_ids = std::nullopt;
    std::optional<CoreRangeSet> sub_core_grids = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("device", "seeds", "user_ids", "sub_core_grids");
    auto attribute_values() const { return std::forward_as_tuple(device, seeds, user_ids, sub_core_grids); }
};

struct ManualSeedInputs {
    std::optional<Tensor> seeds = std::nullopt;
    std::optional<Tensor> user_ids = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("seeds", "user_ids");
    auto attribute_values() const { return std::forward_as_tuple(seeds, user_ids); }
};

}  // namespace ttnn::prim
