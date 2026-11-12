// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operation.hpp"
#include "chain_affine_transforms_program_factory.hpp"

namespace ttnn::experimental::prim {

struct ChainAffineTransformsOperation {
    using operation_attributes_t = ChainAffineTransformsParams;
    using tensor_args_t = ChainAffineTransformsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChainAffineTransformsProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::pair<Tensor, Tensor> chain_affine_transforms(
    const Tensor& transforms,
    const Tensor& initial_state,
    const tt::tt_metal::MemoryConfig&,
    const DeviceComputeKernelConfig&,
    const Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows);

}  // namespace ttnn::experimental::prim
