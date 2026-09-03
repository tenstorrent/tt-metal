// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/operation.hpp"

#include "prepare_chunk_recurrence_device_operation_types.hpp"
#include "prepare_chunk_recurrence_program_factory.hpp"

namespace ttnn::experimental::prim {

struct PrepareChunkRecurrenceOperation {
    using operation_attributes_t = PrepareChunkRecurrenceParams;
    using tensor_args_t = PrepareChunkRecurrenceInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<PrepareChunkRecurrenceProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> prepare_chunk_recurrence(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    uint32_t num_heads,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t output_bf16_mask);

}  // namespace ttnn::experimental::prim
