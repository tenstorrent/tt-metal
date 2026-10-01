// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scatter_codegen_program_factory.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ScatterCodegenParams {
    // TILE-only geometry (interleaved/streaming factories).
    const uint32_t Ht;
    const uint32_t Wt_output;
    const uint32_t Wt_index;
    const uint32_t output_logical_w;
    const uint32_t idx_valid_h_last;
    const uint32_t idx_valid_w_last;
    const uint32_t Ht_per_batch_input;
    const uint32_t Ht_per_batch_src;
    // ROW_MAJOR-only geometry (rm/bf16_reduce_rm factories).
    const uint32_t num_sticks;
    const uint32_t input_stick_elems;
    const uint32_t index_stick_elems;
    // Shared across every factory.
    const std::array<uint32_t, 17> page_map;
    const uint32_t reduction_mode;
    const uint32_t value_kind;
    const tt::tt_metal::MemoryConfig output_mem_config;
    // Execution control: every program factory threads it into its own work split, so it is honoured
    // rather than dropped. Still participates in the default whole-struct hash like every other field
    // above.
    const std::optional<CoreRangeSet> sub_core_grids;
};

struct ScatterCodegenInputs {
    const Tensor& input_tensor;
    const Tensor& index_tensor;
    const Tensor& src_tensor;
    std::optional<Tensor> output_tensor;
};

struct ScatterCodegenDeviceOperation {
    using operation_attributes_t = ScatterCodegenParams;
    using tensor_args_t = ScatterCodegenInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        ScatterCodegenProgramFactoryInterleaved,
        ScatterCodegenProgramFactoryStreaming,
        ScatterCodegenProgramFactoryRowMajor,
        ScatterCodegenProgramFactoryBf16ReduceRowMajor>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

Tensor scatter_codegen(
    const ScatterCodegenParams& params,
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    const std::optional<Tensor>& output_tensor);

}  // namespace ttnn::prim
