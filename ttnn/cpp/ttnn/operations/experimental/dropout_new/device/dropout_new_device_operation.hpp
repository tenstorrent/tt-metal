// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "dropout_new_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DropoutNewProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DropoutNewParams& args, const DropoutNewInputs& tensor_args, Tensor& output);
};

struct DropoutNewMeshWorkloadFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DropoutNewParams& args,
        const DropoutNewInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

struct DropoutNewDeviceOperation {
    using operation_attributes_t = DropoutNewParams;
    using tensor_args_t = DropoutNewInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DropoutNewProgramFactory, DropoutNewMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::DropoutNewDeviceOperation::tensor_return_value_t dropout_new(
    const Tensor& input,
    float prob,
    float scale,
    uint32_t seed,
    bool use_per_device_seed,
    DataType output_dtype,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
