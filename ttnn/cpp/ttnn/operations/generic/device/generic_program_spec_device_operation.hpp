// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/reflection.hpp>

#include "generic_op_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::generic {

// generic_op on a Metal 2.0 ProgramSpec and ProgramRunArgs. Like the ProgramDescriptor form, it hands the caller's
// programs to Metal unchanged and adds program caching, mesh dispatch and per-call argument checks.
struct GenericProgramSpecDeviceOperation {
    struct operation_attributes_t {
        // A program and its run args. They point into the caller's objects, which outlive the call.
        struct Program {
            // Unset for the ProgramSpec form: the program runs on every device that holds the tensors.
            std::optional<ttnn::MeshCoordinateRange> range;
            const tt::tt_metal::experimental::ProgramSpec* spec = nullptr;
            const tt::tt_metal::experimental::ProgramRunArgs* run_args = nullptr;
        };
        std::vector<Program> programs;
        std::optional<uint64_t> custom_program_hash;

        // Op logging and the profiler show the number of programs, not whole specs.
        static constexpr auto attribute_names = std::forward_as_tuple("num_programs");
        auto attribute_values() const { return std::make_tuple(programs.size()); }
    };
    using tensor_args_t = generic::tensor_args_t;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Returns the caller's programs on the first call and the caller's run args on every later call.
    struct ProgramFactory {
        static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);

        static const tt::tt_metal::experimental::ProgramRunArgs& override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRange& range);
    };
    using program_factory_t = std::variant<ProgramFactory>;

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Hash of everything in a ProgramSpec that changes the compiled program. generic_op keys its program cache on it,
// combined with each program's mesh range, unless the caller gives a custom_program_hash.
ttsl::hash::hash_t compute_program_spec_hash(const tt::tt_metal::experimental::ProgramSpec& spec);

}  // namespace ttnn::operations::generic

namespace ttnn::prim {

Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::GenericProgramSpecDeviceOperation::operation_attributes_t& operation_attributes);

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::GenericProgramSpecDeviceOperation::operation_attributes_t& operation_attributes);

}  // namespace ttnn::prim
