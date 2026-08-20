// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed {

struct RotaryEmbeddingIndexedDeviceOperation {
    struct operation_attributes_t {
        uint32_t cluster_axis;  // mesh axis the cos/sin caches are SP-sharded along.
        // Prior valid global KV length in tokens. Used only on the SCALAR path (when no `metadata`
        // tensor is supplied): a per-call scalar intentionally NOT hashed — it lives in a common
        // runtime arg patched on cache hits by MeshWorkloadFactory::override_runtime_arguments, so one
        // cached program is reused across chunks while the value stays current. On the METADATA path it
        // is unused (0); the reader reads kv_actual_global on-device from element [0] of the 1-element
        // `metadata` tensor.
        uint32_t kv_actual_global;  // scalar path only
        MemoryConfig output_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& cos;
        const Tensor& sin;
        const Tensor& trans_mat;
        // Optional. When set: a dedicated 1-element uint32 DRAM tensor, replicated across the mesh,
        // holding kv_actual_global (tokens, tile-aligned) directly at element [0]. The reader NoC-reads
        // that one element on-device (traceable path), so a captured ttnn trace advances the value per
        // chunk via an in-place host update of this tensor. When empty, the op uses the scalar
        // `kv_actual_global` attribute.
        std::optional<Tensor> metadata;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Per-device sharding means each mesh coordinate gets its own program (my_sp_coord is a per-device
    // compile-time arg), so the op builds the mesh workload itself rather than stamping one coord-blind
    // ProgramSpec. No per-coordinate state is needed on cache hits (override re-derives everything from
    // attributes/tensor_args), but the mesh-workload adapter requires a shared-variables type.
    struct SharedVariables {};

    struct MeshWorkloadFactory {
        using shared_variables_t = SharedVariables;
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& args,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& args,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

    private:
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        // Build this device's named-arg ProgramSpec (my_sp_coord baked from `coord`), compile it via
        // MakeProgramFromSpec, and set its initial run args.
        static cached_program_t create_at(
            const operation_attributes_t& args,
            const ttnn::MeshCoordinate& coord,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<MeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed

namespace ttnn::prim {

// Unified primitive. `metadata` selects the path: set -> traceable on-device read of kv_actual_global
// (from element [0] of the 1-element metadata tensor; the scalar arg is ignored, pass 0); empty ->
// scalar path using the kv_actual_global attribute.
ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    const std::optional<ttnn::Tensor>& metadata,
    uint32_t kv_actual_global,
    uint32_t cluster_axis,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim
