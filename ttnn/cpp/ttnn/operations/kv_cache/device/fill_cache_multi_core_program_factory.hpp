// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::kv_cache {

tt::tt_metal::operation::ProgramWithCallbacks fill_cache_multi_core(
    const Tensor& cache_tensor, const Tensor& input_tensor, const uint32_t batch_idx, const uint32_t update_idx);

}  // namespace ttnn::operations::kv_cache
