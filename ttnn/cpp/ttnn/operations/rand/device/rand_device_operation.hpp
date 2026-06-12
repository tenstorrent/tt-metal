// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

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
    };

    struct tensor_args_t {};

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // MetalV2 factory — AdvancedProgramSpecFactoryConcept with the per-enqueue split (Option 3++).
    //
    // The method surface IS the documentation of what's what:
    //   - extract_immutable_info → the cache key AND the sole input to create_program_spec. It is
    //     the structural projection of the request (output layout + grid). It deliberately EXCLUDES
    //     seed/from/to, so two calls that differ only in those values map to the same cache entry — and
    //     a mutable value cannot leak into the spec, because the builder never sees anything but the
    //     ImmutableInfo.
    //   - create_program_spec → the immutable blueprint only (DFBs, kernels, work-units, schemas).
    //   - create_invariant_run_args → the ENQUEUE-INVARIANT run-args (the per-core work split
    //     start_id / num_tiles, declared invariant in the spec). Set once on cache miss and retained.
    //   - create_per_enqueue_args → the PER-ENQUEUE run-args: the per-core RNG seed + from/to range and
    //     the output tensor. Re-applied on every dispatch via UpdateProgramRunArgs, so a new seed
    //     produces new output while the cached program is reused.
    //
    // This replaces the legacy descriptor + get_dynamic_runtime_args + descriptor-patching machinery.
    struct RandProgramFactory {
        struct immutable_info_t {
            tt::tt_metal::TensorSpec output_spec;
            CoreCoord grid;

            static constexpr auto attribute_names = std::forward_as_tuple("output_spec", "grid");
            auto attribute_values() const { return std::forward_as_tuple(output_spec, grid); }
        };

        static immutable_info_t extract_immutable_info(
            const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
        static tt::tt_metal::experimental::ProgramSpec create_program_spec(const immutable_info_t& info);
        static tt::tt_metal::experimental::ProgramRunArgs create_invariant_run_args(const immutable_info_t& info);
        static tt::tt_metal::experimental::ProgramRunArgs create_per_enqueue_args(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    using program_factory_t = std::variant<RandProgramFactory>;
    // No select_program_factory: program_factory_t is a single alternative, so the framework selects it
    // automatically. No compute_program_hash: the cache key is the factory's ImmutableInfo.

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
