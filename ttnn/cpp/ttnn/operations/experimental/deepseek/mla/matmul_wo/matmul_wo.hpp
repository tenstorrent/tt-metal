// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/base_types.hpp>
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek::mla {

ttnn::Tensor matmul_wo(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w_tensor,
    const ttnn::Tensor& output_tensor,
    uint32_t layer_id);

}  // namespace ttnn::experimental::deepseek::mla
