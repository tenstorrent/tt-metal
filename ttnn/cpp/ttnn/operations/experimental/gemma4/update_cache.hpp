// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::gemma4 {

Tensor update_cache(
    const Tensor& cache,
    const Tensor& input,
    const Tensor& request_descriptors,
    uint32_t request_index,
    uint32_t request_local_tokens,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t cluster_axis);

}  // namespace ttnn::experimental::gemma4
