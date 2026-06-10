// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::operations::rand {

struct RandDeviceOperation {
    struct operation_attributes_t {
        const ttnn::Shape shape;
        DataType dtype;
        Layout layout;
        const MemoryConfig memory_config;
        MeshDevice* device;
        const float from;
        const float to;
        uint32_t seed;
        ttsl::SmallVector<bool> mesh_dim_is_sharded;

        // seed/from/to are PER-ENQUEUE values (Metal 2.0): they are carried by
        // create_per_enqueue_run_args and re-applied on every dispatch via UpdateProgramRunArgs. They
        // are deliberately EXCLUDED from attribute_values() so the framework's program hash treats two
        // calls that differ only in seed/from/to as a CACHE HIT (a single cached program), rather than
        // recompiling per seed. `device` is FIRST so the framework can discover the mesh device via
        // get_first_object_of_type over attribute_values() — rand has no input tensor.
        static constexpr auto attribute_names =
            std::forward_as_tuple("device", "shape", "dtype", "layout", "memory_config");
        auto attribute_values() const { return std::forward_as_tuple(device, shape, dtype, layout, memory_config); }
    };

    struct tensor_args_t {};

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // Metal 2.0 factory (ProgramSpecFactoryWithPerEnqueueArgsConcept).
    //
    // create_program_spec builds the immutable ProgramSpec and the ENQUEUE-INVARIANT run-args (the
    // per-core work split: start_id / num_tiles), declaring those args invariant in the spec. They are
    // applied once on cache miss and retained across enqueues.
    //
    // create_per_enqueue_run_args carries the PER-ENQUEUE values: the per-core RNG seed and the
    // from/to range, plus the output tensor binding. The framework merges these with the invariant set
    // for the cache-miss SetProgramRunArgs, and on every cache hit re-applies ONLY this set via
    // UpdateProgramRunArgs — so a different seed produces different output while the program is reused.
    //
    // This replaces the legacy descriptor + get_dynamic_runtime_args + descriptor-patching machinery
    // that rand previously used to refresh the seed on a cache hit.
    struct RandProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_spec(
            const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output);
        static tt::tt_metal::experimental::ProgramRunArgs create_per_enqueue_run_args(
            const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<RandProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::rand

namespace ttnn::prim {
ttnn::operations::rand::RandDeviceOperation::tensor_return_value_t uniform(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    float from,
    float to,
    uint32_t seed,
    ttsl::SmallVector<bool> mesh_dim_is_sharded = {});
}  // namespace ttnn::prim
