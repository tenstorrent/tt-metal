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
// The three runtime scalars (local_expert_idx, curr_expert_iter, expert_iter_length)
// live here so the framework's tensor-visitor on tensor_args_t isn't asked to walk
// them — and the custom compute_program_hash on RoutedMatmulDeviceOperation hashes
// only tensor_args + tensor specs, so every scalar here is excluded by construction.
// That lets the same program be reused across experts and chunk iterations; only
// runtime args change between dispatches. fused_activation lives inside program_config
// (on the matmul variant structs).
struct RoutedMatmulParams {
    ttnn::operations::matmul::MatmulProgramConfig program_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::DataType output_dtype;
    uint32_t local_expert_idx;
    uint32_t curr_expert_iter;
    uint32_t expert_iter_length;
};

// Tensor inputs for the per-chunk guard predicate:
//   skip iff expert_token_counts[global_expert_idx_table[local_expert_idx]]
//           <= curr_expert_iter * expert_iter_length
// Both tables are small DRAM row-major uint32 tensors. Row-major layout means
// torch[0, 0, i] lands at DRAM byte i*4 so the kernel can index directly into
// its L1 scratch with no face/row packing math. Each kernel reads them straight
// from DRAM — no host involvement.
struct RoutedMatmulInputs {
    ttnn::Tensor a;
    ttnn::Tensor b;
    ttnn::Tensor global_expert_idx_table;  // (1, 1, experts_per_chip), uint32, DRAM, ROW_MAJOR
    ttnn::Tensor expert_token_counts;      // (1, 1, num_global_experts), uint32, DRAM, ROW_MAJOR
    std::optional<ttnn::Tensor> optional_output_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device
