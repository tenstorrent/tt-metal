// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>
#include <tuple>

namespace ttnn::prim {
struct TopkParams {
    uint32_t k{};
    int8_t dim{};
    bool largest{};
    bool sorted{};
    // When true, the bitonic sort/merge/rebuild stages keep the lowest index among equal values, so
    // ties are broken deterministically instead of by array position. Off by default: it changes
    // which index is returned for tied values, so callers must opt in.
    bool stable{};
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::CoreRangeSet sub_core_grids;
};

struct TopkInputs {
    Tensor input;
    std::optional<Tensor> indices;
    std::optional<std::tuple<Tensor, Tensor>> preallocated_outputs;
};
}  // namespace ttnn::prim
