// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct FillPadParams {
    ttnn::PadValue fill_value;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
};

struct FillPadInputs {
    Tensor input;
};

}  // namespace ttnn::prim
