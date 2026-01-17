// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct PlusoneParams {
    const std::optional<CoreRangeSet> sub_core_grids;
    const bool skip_negative_entries;
};

}  // namespace ttnn::experimental::prim
