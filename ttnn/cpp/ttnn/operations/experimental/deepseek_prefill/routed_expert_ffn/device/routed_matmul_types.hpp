// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device {

// Attributes that affect the compiled program (part of the program cache key).
// expert_iter lives here so the framework's tensor-visitor on tensor_args_t isn't
// asked to walk a scalar — but the custom compute_program_hash on
// RoutedMatmulDeviceOperation deliberately excludes expert_iter from the hash, so
// the same program is reused across iterations (only runtime args change).
// fused_activation lives inside program_config (on the matmul variant structs).
struct RoutedMatmulParams {
    ttnn::operations::matmul::MatmulProgramConfig program_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::DataType output_dtype;
    uint32_t expert_iter;
};

// Tensor inputs. max_iter is a small DRAM tile-layout tensor whose [0,0] scalar
// each kernel reads to decide skip-vs-execute against the runtime expert_iter.
struct RoutedMatmulInputs {
    ttnn::Tensor a;
    ttnn::Tensor b;
    ttnn::Tensor max_iter;
    std::optional<ttnn::Tensor> optional_output_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device
