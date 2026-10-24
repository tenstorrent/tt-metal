// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

// SDPAProgramConfig / PagedCacheGeometryOverride are shared with the source op (sdpa_config.hpp is not forked).
namespace ttnn::operations::bringup {
using ttnn::operations::transformer::PagedCacheGeometryOverride;
using ttnn::operations::transformer::SDPAProgramConfig;
}  // namespace ttnn::operations::bringup

namespace ttnn::prim::bringup {

struct RingDistributedSDPAParams {
    uint32_t ring_size = 0;
    std::optional<uint32_t> ring_id;
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::bringup::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<int64_t> chunk_start_idx;
};

struct RingDistributedSDPAInputs {
    ttnn::Tensor q;
    ttnn::Tensor k;
    ttnn::Tensor v;
    std::optional<ttnn::Tensor> page_table;
};

}  // namespace ttnn::prim::bringup
