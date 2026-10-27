// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::clone {

// Metal 2.0 program factory (defined below CloneOperation, forward-declared here so
// it can appear in CloneOperation::program_factory_t).
struct CloneProgramFactory;

struct CloneOperation {
    struct operation_attributes_t {
        const tt::tt_metal::DataType dtype;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<CloneProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

struct CloneProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const CloneOperation::operation_attributes_t& operation_attributes,
        const CloneOperation::tensor_args_t& tensor_args,
        CloneOperation::tensor_return_value_t& output);
};

}  // namespace ttnn::operations::data_movement::clone

namespace ttnn::prim {
ttnn::Tensor clone(
    const Tensor& input,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
}  // namespace ttnn::prim
