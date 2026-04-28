// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/distributed/types.hpp"
#include <optional>
#include <set>

namespace ttnn::prim {

struct SDPAParams {
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    bool is_causal = false;
    std::optional<int64_t> chunk_start_idx;        // Chunked legacy: scalar offset, part of program cache key
    std::optional<Tensor> chunk_start_idx_tensor;  // Chunked flexible: device tensor [1] int32, read at runtime
    DeviceComputeKernelConfig compute_kernel_config;
    bool use_mla = false;
    std::optional<uint32_t> head_dim_v;
    std::optional<uint32_t> sliding_window_size;
    // When set, the op only runs on the listed mesh coordinates. Other coords
    // hosting the input tensor stay idle for this op (no program is dispatched
    // there). Selects a mesh-workload program factory over the single-program one.
    std::optional<std::set<ttnn::MeshCoordinate>> mesh_coords;
};

struct SDPAInputs {
    Tensor q;
    Tensor k;
    std::optional<Tensor> v;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> page_table;
    std::optional<Tensor> attention_sink;
};

}  // namespace ttnn::prim
