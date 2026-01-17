// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"

namespace ttnn::prim {

struct LayerNormShardedSharedVariables {
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    tt::tt_metal::KernelHandle writer_mcast_sender_kernels_id = {};
    tt::tt_metal::KernelHandle writer_mcast_receiver_kernels_id = {};
    uint32_t num_none_all_to_all_workers = 0;
    bool is_pre_all_gather = false;
    tt::tt_metal::CBHandle cb_in0{};
    tt::tt_metal::CBHandle cb_in1{};
    tt::tt_metal::CBHandle cb_stats{};
    tt::tt_metal::CBHandle cb_add_out{};
    tt::tt_metal::CBHandle cb_output{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    CoreRangeSet all_cores;  // Actual cores used (needed for parallel composition)
};
struct LayerNormShardedProgramFactory {
    using shared_variables_t = LayerNormShardedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    // Original create() - creates a new program
    static cached_program_t create(
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        Tensor& tensor_return_value);

    // Direct contribution: adds kernels/CBs to an existing program
    // Used by ttnn::parallel for fusing multiple operations
    // Note: Sharded layernorm uses multicasting which requires careful handling
    // when used with core_range_override in parallel context
    static shared_variables_t add_to(
        tt::tt_metal::Program& program,
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_override = std::nullopt);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        Tensor& tensor_return_value);

    // Override runtime args using shared_variables directly (for parallel composition)
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        shared_variables_t& shared_variables,
        const LayerNormParams& operation_attributes,
        const LayerNormInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
