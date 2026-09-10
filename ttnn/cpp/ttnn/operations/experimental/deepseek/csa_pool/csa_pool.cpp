// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "csa_pool.hpp"

#include "device/csa_pool_device_operation.hpp"

namespace ttnn::experimental::deepseek {

Tensor csa_pool_window(
    const Tensor& prev_kv,
    const Tensor& prev_gate,
    const Tensor& win_kv,
    const Tensor& win_gate,
    const Tensor& position_bias,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::csa_pool_window(
        prev_kv, prev_gate, win_kv, win_gate, position_bias, memory_config, compute_kernel_config);
}

}  // namespace ttnn::experimental::deepseek
