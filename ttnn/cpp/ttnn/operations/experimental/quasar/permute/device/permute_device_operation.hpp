// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include <tt_stl/small_vector.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/quasar/permute/device/permute_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

// Quasar leaves six DM harts of a cluster to kernels (DM0 serves the DFB ISR, DM1 the remapper).
constexpr uint32_t kPermuteMaxThreads = 6;
// Cap of one L1 window (read or write) of one DM thread; a thread owns one of each.
constexpr uint32_t kPermuteMaxWindowBytes = 128 * 1024;

struct PermuteParams {
    ttsl::SmallVector<uint32_t> dims;
    tt::tt_metal::MemoryConfig output_mem_config;
    float pad_value = 0.0f;
    uint32_t num_threads = kPermuteMaxThreads;
};

struct PermuteInputs {
    Tensor input;
};

// The block plan of one permute: how the tensor is cut into blocks and how a DM thread moves one block.
permute_plan::BlockPlan permute_block_plan(const PermuteParams& params, const Tensor& input, uint32_t target_blocks);

struct PermuteProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PermuteParams& operation_attributes, const PermuteInputs& tensor_args, Tensor& tensor_return_value);
};

struct PermuteDeviceOperation {
    using operation_attributes_t = PermuteParams;
    using tensor_args_t = PermuteInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PermuteProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor permute(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    float pad_value = 0.0f,
    uint32_t num_threads = kPermuteMaxThreads);

}  // namespace ttnn::prim::qsr
