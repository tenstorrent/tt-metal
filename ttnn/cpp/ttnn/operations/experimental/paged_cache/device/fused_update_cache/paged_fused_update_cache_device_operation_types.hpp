// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct PagedFusedUpdateCacheParams {
    const std::vector<uint32_t> update_idxs;
    const uint32_t batch_offset;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    const bool share_cache;
    const std::optional<std::set<ttnn::MeshCoordinate>> mesh_coords;
    // > 1: each input shard carries one user's rows_per_user consecutive 32-row tiles (row j in tile j)
    // and update_idxs_tensor carries one position per row (user-major: user u, row j at u * rows_per_user + j).
    const uint32_t rows_per_user = 1;
};

struct PagedFusedUpdateCacheInputs {
    Tensor cache_tensor1;
    Tensor input_tensor1;
    Tensor cache_tensor2;
    Tensor input_tensor2;
    std::optional<Tensor> update_idxs_tensor;
    std::optional<Tensor> page_table;
};

using PagedFusedUpdateCacheResultSpec = std::vector<tt::tt_metal::TensorSpec>;
using PagedFusedUpdateCacheResult = std::tuple<Tensor, Tensor>;

}  // namespace ttnn::experimental::prim
