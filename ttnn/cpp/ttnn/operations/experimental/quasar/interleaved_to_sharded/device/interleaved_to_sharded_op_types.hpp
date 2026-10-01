// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim::qsr {

struct InterleavedToShardedParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    bool keep_l1_aligned{};
};

struct InterleavedToShardedInputs {
    ttnn::Tensor input_tensor;
    std::optional<ttnn::Tensor> output_tensor;
};

}  // namespace ttnn::prim::qsr
