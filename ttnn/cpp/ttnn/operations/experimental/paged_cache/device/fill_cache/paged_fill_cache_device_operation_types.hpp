// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct PagedFillCacheParams {
    const uint32_t batch_idx_fallback;
    const std::optional<std::set<ttnn::MeshCoordinate>> mesh_coords;
    const bool noop = false;  // When true, kernels early exit
    // Optional per-call block_size override; see PagedUpdateCacheParams::block_size_override.
    const std::optional<uint32_t> block_size_override = std::nullopt;
};

struct PagedFillCacheInputs {
    Tensor cache_tensor;  // also output tensor
    Tensor input_tensor;
    Tensor page_table;
    std::optional<Tensor> batch_idx_tensor_opt;
};

}  // namespace ttnn::experimental::prim
