// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ProdAllParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ProdAllInputs {
    Tensor input;
};

}  // namespace ttnn::prim
