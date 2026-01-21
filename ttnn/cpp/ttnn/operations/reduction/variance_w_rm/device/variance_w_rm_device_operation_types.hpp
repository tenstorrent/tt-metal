// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::variance_w_rm {

struct VarianceWRmParams {
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct VarianceWRmInputs {
    const Tensor& input;
};

}  // namespace ttnn::operations::reduction::variance_w_rm
