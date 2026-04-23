// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

// Internal (not python-bound) forked matmul with a runtime-gated early-return.
// Each kernel reads max_expert_iter[0,0] from DRAM at entry; if curr_expert_iter > max_expert_iter,
// the kernel returns without touching CBs or semaphores. Output on skip is
// whatever was in the optional_output_tensor before — caller must avoid reading
// skipped slices.
//
// Argument order mirrors ttnn::matmul. program_config and compute_kernel_config are
// optional in the signature for API parity but are required in practice: the forked
// factory only supports MatmulMultiCoreReuseMultiCastProgramConfig and has no
// auto-select path, so passing nullopt TT_FATALs. max_expert_iter and curr_expert_iter
// are the routed-specific parameters and are required.
//
// Intended BH-only.
ttnn::Tensor routed_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    // memory_config: when an optional_output_tensor is also provided the device op
    // ignores it for allocation; when both are omitted we default to DRAM interleaved.
    const std::optional<const tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    std::optional<const tt::tt_metal::DataType> dtype = std::nullopt,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const ttnn::Activation>& activation = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& max_expert_iter = std::nullopt,
    uint32_t curr_expert_iter = 0);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
