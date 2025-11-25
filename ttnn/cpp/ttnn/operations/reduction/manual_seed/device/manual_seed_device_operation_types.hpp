// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::operations::reduction::manual_seed {
struct operation_attributes_t {
    tt::tt_metal::distributed::MeshDevice* device = nullptr;
    std::optional<uint32_t> seeds = std::nullopt;
    std::optional<uint32_t> user_ids = std::nullopt;
};

struct tensor_args_t {
    Tensor
        output;  // TODO: To be removed when API will be fixed with https://github.com/tenstorrent/tt-metal/pull/32260
    std::optional<Tensor> seeds = std::nullopt;
    std::optional<Tensor> user_ids = std::nullopt;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::reduction::manual_seed
