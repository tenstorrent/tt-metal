// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ZeroCacheRangeParams {
    uint32_t start_page = 0;  // first page to zero (inclusive)
    uint32_t end_page = 0;    // last page to zero (exclusive)
};

struct ZeroCacheRangeInputs {
    Tensor cache;
};

}  // namespace ttnn::prim
