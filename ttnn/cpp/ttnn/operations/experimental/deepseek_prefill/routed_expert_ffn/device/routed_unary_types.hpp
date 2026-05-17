// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device {

// Single-op, sharded-only routed unary. Mirrors RoutedMatmulParams: only the
// per-iteration scalars (local_expert_idx, curr_expert_iter, expert_iter_length)
// change between dispatches — they live here so they can be excluded from the
// program hash, letting the same cached program be reused across all chunk
// iterations.
struct RoutedUnaryParams {
    // Single-op chain (size must be 1). Stored as a vector so RoutedUnaryParams
    // is default-constructible — the reflect library used by device_operation
    // launch/log paths requires that.
    std::vector<ttnn::operations::unary::EltwiseUnaryWithParam> op_chain;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::DataType output_dtype;
    bool fp32_dest_acc_en;
    bool preserve_fp32_precision;
    bool bfp8_pack_precise;
    uint32_t local_expert_idx;
    uint32_t curr_expert_iter;
    uint32_t expert_iter_length;
};

// Tensor inputs mirror RoutedMatmulInputs. Both guard tables are small DRAM
// row-major uint32 tensors — same format the matmul consumes so we can share
// guard.h without any translation layer.
struct RoutedUnaryInputs {
    ttnn::Tensor input;
    ttnn::Tensor global_expert_idx_table;  // (1, 1, experts_per_chip), uint32, DRAM, ROW_MAJOR
    ttnn::Tensor expert_token_counts;      // (1, 1, num_global_experts), uint32, DRAM, ROW_MAJOR
    std::optional<ttnn::Tensor> optional_output_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device
