// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_fused_qkv.hpp"

#include "device/deepseek_fused_qkv_device_operation.hpp"

namespace ttnn::operations::experimental::transformer::deepseek_fused_qkv {

std::vector<ttnn::Tensor> deepseek_fused_qkv(
    const ttnn::Tensor& hidden,
    const ttnn::Tensor& wqa,
    const ttnn::Tensor& wqb,
    const ttnn::Tensor& wkv,
    const ttnn::Tensor& qa_norm_w,
    const ttnn::Tensor& kv_norm_w,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    float eps,
    uint32_t rope_dim,
    uint32_t num_heads,
    const std::optional<tt::tt_metal::MemoryConfig>& q_mem_config,
    const std::optional<tt::tt_metal::MemoryConfig>& kv_mem_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::deepseek_fused_qkv(
        hidden,
        wqa,
        wqb,
        wkv,
        qa_norm_w,
        kv_norm_w,
        cos,
        sin,
        trans_mat,
        eps,
        rope_dim,
        num_heads,
        q_mem_config,
        kv_mem_config,
        compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::transformer::deepseek_fused_qkv
