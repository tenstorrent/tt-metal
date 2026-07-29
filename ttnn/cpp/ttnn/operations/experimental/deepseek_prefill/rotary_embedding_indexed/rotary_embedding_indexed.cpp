// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_indexed.hpp"

#include "device/rotary_embedding_indexed_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed {

ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t kv_actual_global,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::rotary_embedding_indexed(
        input, cos, sin, trans_mat, kv_actual_global, cluster_axis, memory_config, compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed
