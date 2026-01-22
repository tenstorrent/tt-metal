// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::standardize_w_rm {

struct StandardizeWRmParams {
    const float epsilon;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct StandardizeWRmInputs {
    const Tensor& input;
};

}  // namespace ttnn::operations::reduction::standardize_w_rm
