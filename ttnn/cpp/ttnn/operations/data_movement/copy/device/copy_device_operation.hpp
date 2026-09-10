// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "copy_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct CopyDeviceOperation {
    using operation_attributes_t = ttnn::prim::CopyParams;
    using tensor_args_t = ttnn::prim::CopyInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // NOTE: SameMemoryConfig is NOT ported to Metal 2.0. The peer op `data_movement/move`
    // (MoveProgramFactory) reuses this factory's create_descriptor() and depends on its
    // ProgramDescriptor return type and positional runtime-arg layout, so porting it would break an
    // op outside this port's scope. Left on the legacy descriptor concept; the framework dispatches
    // the ported factories (DefaultRowMajor, DefaultTilized) and this one per-factory at runtime.
    struct SameMemoryConfig {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    struct DefaultRowMajor {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    struct DefaultTilized {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<SameMemoryConfig, DefaultRowMajor, DefaultTilized>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor copy(
    const Tensor& input,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output,
    bool backwards = false);

}  // namespace ttnn::prim
