// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_sdpa.hpp"

#include <cmath>
#include <cstring>

#include "device/kv_sdpa_device_operation.hpp"

namespace ttnn {

Tensor kv_sdpa(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const std::optional<Tensor>& attn_mask,
    std::optional<float> scale,
    const std::optional<Tensor>& past_k,
    const std::optional<Tensor>& past_v,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    const uint32_t DH = input_tensor_q.logical_shape()[-1];
    const float s = scale.value_or(1.0f / std::sqrt(static_cast<float>(DH)));
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &s, sizeof(float));
    return ttnn::prim::kv_sdpa(
        input_tensor_q, input_tensor_k, input_tensor_v, attn_mask, scale_bits, past_k, past_v, compute_kernel_config);
}

}  // namespace ttnn
