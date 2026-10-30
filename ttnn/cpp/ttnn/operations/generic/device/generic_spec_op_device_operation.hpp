// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/reflection.hpp>  // ttsl::hash::hash_t

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::generic {

// Metal 2.0 counterpart of GenericOpDeviceOperation: runs a caller-built ProgramSpec instead of a
// ProgramDescriptor. This is the route for Python-built programs on Quasar, where the legacy
// DataMovementKernel that ProgramDescriptor produces is not supported.
struct GenericSpecOpDeviceOperation {
    struct operation_attributes_t {
        tt::tt_metal::experimental::ProgramSpec spec;
        // Per-node runtime arg values; reapplied on every dispatch, so they are not part of the hash.
        std::vector<tt::tt_metal::experimental::KernelRunArgs> kernel_run_args;
        // Each TensorParameter of `spec` bound to an index into io_tensors.
        std::vector<std::pair<tt::tt_metal::experimental::TensorParamName, std::size_t>> tensor_bindings;
        std::optional<ttsl::hash::hash_t> custom_program_hash;

        // ProgramSpec is not reflectable, so logging and profiling see only its name.
        static constexpr auto attribute_names = std::forward_as_tuple("program_spec_name", "custom_program_hash");
        auto attribute_values() const { return std::forward_as_tuple(spec.name, custom_program_hash); }
    };

    struct tensor_args_t {
        std::vector<Tensor> io_tensors;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct GenericSpecProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<GenericSpecProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::generic

namespace ttnn::prim {
// The last element of io_tensors is the (preallocated) output, as for generic_op.
Tensor generic_spec_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::GenericSpecOpDeviceOperation::operation_attributes_t& operation_attributes);
}  // namespace ttnn::prim
