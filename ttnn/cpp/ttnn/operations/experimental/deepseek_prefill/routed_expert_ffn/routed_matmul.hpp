// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

// Internal (not python-bound) forked matmul with a runtime-gated early-return.
// Each kernel reads max_expert_iter[0,0] from DRAM at entry; if curr_expert_iter > max_expert_iter,
// the kernel returns without touching CBs or semaphores. Output on skip is
// whatever was in the optional_output_tensor before — caller must avoid reading
// skipped slices.
//
// Intended BH-only. Factory accepts only MatmulMultiCoreReuseMultiCastProgramConfig.
ttnn::Tensor routed_matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& max_expert_iter,
    uint32_t curr_expert_iter,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    // Output memory config. Optional: when an optional_output_tensor is also
    // provided the device op ignores output_memory_config for allocation; when
    // omitted we fall back to the optional tensor's config, or to DRAM
    // interleaved if there's no optional tensor either.
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
