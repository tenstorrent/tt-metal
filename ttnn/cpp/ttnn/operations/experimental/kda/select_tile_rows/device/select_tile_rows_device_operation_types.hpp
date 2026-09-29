// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SelectTileRowsParams {
    uint32_t width;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct SelectTileRowsInputs {
    Tensor input;
    Tensor indices;
};

}  // namespace ttnn::experimental::prim
