// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "engram_gate_prefill.hpp"

#include "device/engram_gate_prefill_device_operation.hpp"

namespace ttnn::experimental::deepseek {

Tensor engram_gate_prefill(
    const Tensor& streams,
    const Tensor& kv,
    const Tensor& gate_weight,
    float eps,
    float clamp_value,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::engram_gate_prefill(
        streams, kv, gate_weight, eps, clamp_value, memory_config, compute_kernel_config);
}

}  // namespace ttnn::experimental::deepseek
