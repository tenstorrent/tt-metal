// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct OffsetCumsumParams {
    uint32_t cluster_axis;
    uint32_t num_dispatch_subgroups = 1;
};

}  // namespace ttnn::experimental::prim
