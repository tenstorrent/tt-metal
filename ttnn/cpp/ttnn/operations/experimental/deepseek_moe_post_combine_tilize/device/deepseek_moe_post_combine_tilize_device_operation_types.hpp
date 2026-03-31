// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEPostCombineTilizeParams {
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DeepseekMoEPostCombineTilizeInputs {
    ttnn::Tensor input_tensor;
};

}  // namespace ttnn::experimental::prim
