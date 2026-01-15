// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <set>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::paged_cache::update {

struct UpdateParams {
    const std::vector<uint32_t> update_idxs;
    const uint32_t batch_offset;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    const bool share_cache;
    const std::optional<std::set<ttnn::MeshCoordinate>> mesh_coords;
};

struct UpdateInputs {
    Tensor cache_tensor;
    Tensor input_tensor;
    std::optional<Tensor> update_idxs_tensor;
    std::optional<Tensor> page_table;
};

}  // namespace ttnn::operations::experimental::paged_cache::update
