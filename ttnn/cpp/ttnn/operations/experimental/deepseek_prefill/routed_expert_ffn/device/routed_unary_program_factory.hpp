// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "routed_unary_types.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device {

// Forked minimal unary program factory. Scope is intentionally narrow:
//   * Sharded input + sharded output (same memory config, in-place unary style).
//   * TILE layout only.
//   * Single unary op in the chain (no LOGIT / WHERE_TSS / typecast scalars).
// The reader/writer/compute kernels are forked copies of unary_ng's that read
// global_expert_idx_table + expert_token_counts from DRAM and early-return when
// expert_token_counts[global_expert_idx_table[local_expert_idx]]
// <= curr_expert_iter * expert_iter_length.
struct RoutedUnaryProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::KernelHandle compute_kernel_id{};
        tt::tt_metal::CBHandle cb_src{};
        tt::tt_metal::CBHandle cb_out{};
        tt::tt_metal::CBHandle cb_guard{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RoutedUnaryParams& operation_attributes,
        const RoutedUnaryInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RoutedUnaryParams& operation_attributes,
        const RoutedUnaryInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device
