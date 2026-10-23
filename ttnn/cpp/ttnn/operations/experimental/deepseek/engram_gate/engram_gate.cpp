// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "engram_gate.hpp"

#include "device/engram_gate_device_operation.hpp"

namespace ttnn::experimental::deepseek {

Tensor engram_gate(
    const Tensor& streams,
    const Tensor& kv,
    const Tensor& gate_weight,
    float eps,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::engram_gate(streams, kv, gate_weight, eps, memory_config, compute_kernel_config);
}

}  // namespace ttnn::experimental::deepseek
