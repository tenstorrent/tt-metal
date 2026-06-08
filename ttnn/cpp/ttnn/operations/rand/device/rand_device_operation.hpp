// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::rand {

// ============================================================================
// rand — Metal 2.0 tiered factory
// ============================================================================
//
// rand is migrated to the Metal 2.0 host API. The factory is expressed as two
// orthogonal concepts:
//
//   COMPOSITION — the program is built in three tiers by lifecycle:
//     1. create_program_spec  → the IMMUTABLE blueprint (kernels, DFB, work-units,
//                               tensor parameters, RTA schema). Built on a cache MISS.
//     2. create_static_args   → run-arg VALUES fixed for a cache entry (the work-split
//                               scalars start_id / num_tiles). Set on a cache MISS only;
//                               left baked on a hit.
//     3. create_dynamic_args  → run-arg VALUES that vary per dispatch (the RNG seed,
//                               the [from,to) range, the output tensor address). Applied
//                               on the miss AND re-applied on every cache HIT.
//
//   IDENTITY — the cache key. rand takes the framework DEFAULT (hash the ProgramSpec):
//     two calls that differ only in seed/from/to produce the same spec, so they hit;
//     the new seed is re-applied via create_dynamic_args. No compute_program_hash is
//     needed — correctness is structural. (An op may later add compute_program_hash
//     purely as a host-dispatch perf opt-in; rand does not, by default.)
//
// This replaces the descriptor-era pair (create_descriptor + get_dynamic_runtime_args):
// the static/dynamic partition that get_dynamic_runtime_args expressed by hand is now a
// first-class property of the factory surface, validated against the spec schema.
// ============================================================================

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

        // `device` is FIRST: rand has no input tensor, so the framework discovers the mesh
        // device via get_first_object_of_type over attribute_values(), inspecting element 0.
        static constexpr auto attribute_names =
            std::forward_as_tuple("device", "shape", "dtype", "layout", "memory_config");
        auto attribute_values() const { return std::forward_as_tuple(device, shape, dtype, layout, memory_config); }
    };

    struct tensor_args_t {};

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // ImmutableInfo (adopted from Audrey's PR #45961): the projection of the op args that the
    // immutable program structure depends on — and nothing else. The cache key is hash(ImmutableInfo)
    // (see the adapter), and create_program_spec / create_static_args derive ONLY from it. Because
    // seed/from/to are not in here, they cannot leak into the spec or the key: cache-key correctness
    // ("right program") and seed-staleness avoidance are structural, not by-convention.
    struct immutable_info_t {
        TensorSpec output_spec;  // shape/dtype/layout/memory_config — everything the blueprint needs
        CoreCoord grid;          // compute grid the work split is computed against

        static constexpr auto attribute_names = std::forward_as_tuple("output_spec", "grid");
        auto attribute_values() const { return std::forward_as_tuple(output_spec, grid); }
    };

    // Extract the immutable projection from the op args. Derivable from attributes alone (rand has no
    // input tensor); the framework hashes the result to form the cache key.
    static immutable_info_t extract_immutable_info(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // ---- Composition: the three tiers (Metal 2.0) -------------------------------------

    // Tier 1 — immutable blueprint. Built once on a cache miss. Derives ONLY from ImmutableInfo.
    static tt::tt_metal::experimental::ProgramSpec create_program_spec(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    // Tier 2 — static run-args (work-split scalars). Set once on a cache miss; left baked on a hit.
    // Derives ONLY from ImmutableInfo (pure function of the cache key).
    static tt::tt_metal::experimental::ProgramRunArgs create_static_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    // Tier 3 — dynamic run-args (seed/from/to + output address). Re-applied on EVERY dispatch.
    // The optional coordinate carries the per-device seed offset for sharded meshes; the adapter
    // passes the dispatch coordinate per range.
    static tt::tt_metal::experimental::ProgramRunArgs create_dynamic_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);

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
