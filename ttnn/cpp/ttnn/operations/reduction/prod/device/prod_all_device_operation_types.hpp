// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::prod_all {

struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    Tensor input;
};

}  // namespace ttnn::operations::reduction::prod_all
