// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operation.hpp"
#include "reduce_affine_transforms_program_factory.hpp"

namespace ttnn::experimental::prim {

struct ReduceAffineTransformsOperation {
    using operation_attributes_t = ReduceAffineTransformsParams;
    using tensor_args_t = ReduceAffineTransformsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ReduceAffineTransformsProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::pair<Tensor, Tensor> reduce_affine_transforms(
    const Tensor&, const Tensor&, uint32_t, const tt::tt_metal::MemoryConfig&, const DeviceComputeKernelConfig&);

}  // namespace ttnn::experimental::prim
