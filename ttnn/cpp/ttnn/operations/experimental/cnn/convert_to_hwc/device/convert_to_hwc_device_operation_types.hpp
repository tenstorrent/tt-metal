// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct ConvertToHwcParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
};

struct ConvertToHwcInputs {
    const Tensor& input;
};

}  // namespace ttnn::experimental::prim
