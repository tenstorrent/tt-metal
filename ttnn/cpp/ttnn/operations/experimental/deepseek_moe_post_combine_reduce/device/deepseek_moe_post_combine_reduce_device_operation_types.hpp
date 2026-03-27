// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

struct DeepseekMoEPostCombineReduceParams {
    uint32_t expert_dim;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DeepseekMoEPostCombineReduceInputs {
    ttnn::Tensor combine_output;
    ttnn::Tensor weights;
};

}  // namespace ttnn::experimental::prim
