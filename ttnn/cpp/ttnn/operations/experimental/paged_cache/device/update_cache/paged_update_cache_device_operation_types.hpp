// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <set>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct PagedUpdateCacheParams {
    const std::vector<uint32_t> update_idxs;
    const uint32_t batch_offset;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    const bool share_cache;
    const std::optional<std::set<ttnn::MeshCoordinate>> mesh_coords;
    // Optional override of the per-block token capacity. When unset, the
    // kernel derives ``block_size`` from ``cache_tensor.padded_shape()[2]``
    // (the legacy CUDA-style "block_size from cache" path). When set,
    // every call may interpret the same physical buffer as having a
    // different ``(block_size, head_dim)`` tile arrangement, as long as
    // the total bytes per cache block are preserved — this is what
    // upstream vLLM's hybrid kv-cache-groups manager assumes after its
    // page-size unifier doubles smaller-page-size groups' ``block_size``
    // to equalise per-block memory. See validate_on_program_cache_miss
    // for the byte-count consistency check.
    const std::optional<uint32_t> block_size_override;
};

struct PagedUpdateCacheInputs {
    Tensor cache_tensor;
    Tensor input_tensor;
    std::optional<Tensor> update_idxs_tensor;
    std::optional<Tensor> page_table;
};

}  // namespace ttnn::experimental::prim
