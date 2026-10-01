// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gather_codegen_program_factory.hpp"

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct GatherCodegenParams {
    const int8_t dim;
    const bool sparse_grad;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<CoreRangeSet> sub_core_grids;
    // Pure functions of the input/index tensor shapes (compute_gather_geometry), hoisted here so
    // they participate in this struct's hash/equality instead of being recomputed inside the
    // program factory.
    const uint32_t Ht;
    const uint32_t Wt_input;
    const uint32_t Wt_index;
    const uint32_t index_valid_h_last;
    const uint32_t index_valid_w_last;
    const uint32_t index_ht_per_batch;
};

struct GatherCodegenInputs {
    const Tensor& input_tensor;
    const Tensor& input_index_tensor;
    std::optional<Tensor> output_tensor;
};

struct GatherCodegenDeviceOperation {
    using operation_attributes_t = GatherCodegenParams;
    using tensor_args_t = GatherCodegenInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        GatherCodegenProgramFactoryInterleaved,
        GatherCodegenProgramFactoryTiled,
        GatherCodegenProgramFactoryStreaming>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

Tensor gather_codegen(
    const Tensor& input_tensor,
    int8_t dim,
    const Tensor& input_index_tensor,
    bool sparse_grad,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::prim
