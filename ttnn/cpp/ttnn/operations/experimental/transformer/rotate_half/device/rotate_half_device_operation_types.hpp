// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RotateHalfParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

}  // namespace ttnn::experimental::prim
