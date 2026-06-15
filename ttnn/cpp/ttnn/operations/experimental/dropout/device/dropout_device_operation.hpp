// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "dropout_program_factory.hpp"

#include "dropout_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct DropoutDeviceOperation {
    using operation_attributes_t = DropoutParams;
    using tensor_args_t = DropoutInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DropoutProgramFactory, DropoutMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Kept (not attribute_names): the hash coarsens the input to its VOLUME, since dropout is
    // elementwise (program depends on tile count, not shape). attribute_names can't express that —
    // it only controls the attrs struct, while the input shape is hashed from tensor_args. The seed
    // it excludes is re-applied via get_dynamic_runtime_args below.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // seed is excluded from the program hash (so calls differing only in seed cache-hit); it is
    // DYNAMIC and re-applied to the cached program on every dispatch (per-device offset applied when
    // use_per_device_seed). Must mirror the compute-kernel seed runtime arg in the factory.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& args,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::DropoutDeviceOperation::tensor_return_value_t dropout(
    const Tensor& input,
    float prob,
    float scale,
    uint32_t seed,
    bool use_per_device_seed,
    DataType output_dtype,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
