// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

namespace ttnn::operations::ccl {

struct MeshPartitionDeviceOperation {
    struct operation_attributes_t {
        uint32_t dim;
        std::optional<uint32_t> cluster_axis;
        const MemoryConfig output_mem_config;
    };
    struct tensor_args_t {
        const ttnn::Tensor input_tensor;
        const std::optional<ttnn::Tensor> optional_output_tensor;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;

    using tensor_return_value_t = ttnn::Tensor;

    struct MeshPartition {
        using OverrideRuntimeArgsCallback = std::function<void(
            const void*,
            tt::tt_metal::Program&,  // ‼  no const, exact type
            const std::vector<ttnn::Tensor>&,
            const std::vector<std::optional<const ttnn::Tensor>>&,
            const std::vector<ttnn::Tensor>&)>;

        // Each coordinate owns a program with fixed slice geometry; cache hits refresh only tensors.
        struct shared_variables_t {};
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MeshPartition>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Label the output Shard{dim} along the partitioned mesh axis (the union default would keep the input's label);
    // the rules are detail::compute_mesh_partition_topology below.
    static std::vector<tt::tt_metal::TensorTopology> compute_output_topologies(
        const operation_attributes_t&, const tensor_args_t&);
};

namespace detail {
uint32_t get_cluster_axis_size(const ttnn::Tensor& input_tensor, const std::optional<uint32_t>& cluster_axis);

// Output TensorTopology of a mesh_partition as a pure function of the input's label (no device access; unit-tested
// in tests/ttnn/unit_tests/gtests/ccl/test_mesh_partition_topology_rules.cpp). `topology` is nullopt when no
// TensorTopology can describe the partitioned result; the op then returns {} and launch() keeps the input's label
// (the framework's union default). `fallback_reason` is set for every fallback, so the op can warn (once per
// distinct case), and null only for inputs validation rejects right after the hook or that carry no placements.
//
// `dim` is normalised to [0, rank). Shard dims inside the label may be negative or out of range (left behind by
// rank-changing ops, #52331) and are compared normalised; an out-of-range one counts as not sharding `dim`.
// N = devices in the label, M = devices in the mesh. Rules, in the order they apply:
//
//   #  input label                       cluster_axis  output label
//   0  a label that does not describe   any           fallback (warn), except as noted. A collapsed label has no axes
//      the devices the partition acts on               and must cover the mesh. An N-D sub-mesh label (fits per axis:
//                                                      mesh_shape_override's SUBMESH mode, coords = the block's) is
//                                                      kept when the cluster axis spans the mesh along that axis, so
//                                                      every partition group lies inside the block: rule 1 then applies
//                                                      within the block. A whole-mesh partition of a sub-mesh label, a
//                                                      sub-mesh label cut along a shorter axis, and a cluster axis on a
//                                                      row-major reshape ({1,8} over 2x4) fall back; the reshape with
//                                                      cluster_axis=None goes to rule 2 (its coords are row-major).
//   1  N-D [p0, p1, ...]                 a             p[a] = Shard{dim}, other axes unchanged. If another axis i also
//                                                      shards dim: (i) size-1 axis i -> Replicate (exact); (ii) i < a,
//                                                      p[a] was Replicate or Shard{dim}, every other axis size 1 ->
//                                                      collapsed {N},[Shard{dim}] over the input's row-major coords
//                                                      (row-major hierarchical sharding: exact when p[a] was Replicate,
//                                                      the output-bytes stance of rule 2 when it was already
//                                                      Shard{dim}, a label the mappers themselves refuse to build);
//                                                      (iii) otherwise fallback (warn): dim ends up sharded on two axes
//                                                      in an order no label states.
//   2  any                               None          collapsed {N},[Shard{dim}] over the input's row-major coords
//                                                      when every non-trivial axis is Replicate or Shard{dim}; a Shard
//                                                      of another dim on a non-trivial axis -> fallback (warn). An
//                                                      existing Shard{dim} is overwritten: device k then holds chunk k
//                                                      of ITS OWN input shard, so the label describes the output bytes,
//                                                      not a re-slicing of the input's global view.
//   3  collapsed {N},[p], N == size of a a             {N},[Shard{dim}] whatever p was (overwritten, as in rule 2).
//   4  collapsed {N},[Replicate],        a             N-D over the mesh: Shard{dim} on a, Replicate elsewhere.
//      multi-axis mesh
//   5  collapsed {N},[Shard{k}],         a             fallback (warn): neither form can hold both the old Shard{k} and
//      multi-axis mesh                                 the partition's Shard{dim}.
//
// Overwriting what the partitioned axis held (rules 1, 2, 3) follows reduce_scatter_minimal_async: after the op the
// devices along that axis hold distinct slices of dim, which is what the label must say for serialisation to keep
// every slice, and what the axis held before is not recoverable from the output in either label form. The fallbacks
// are the cases where the output's own layout has no spelling, so the stale input label plus a warning is the honest
// option left; every fallback warns, once per distinct case, because the hook runs on every launch.
struct MeshPartitionTopology {
    std::optional<tt::tt_metal::TensorTopology> topology;
    const char* fallback_reason = nullptr;
};
MeshPartitionTopology compute_mesh_partition_topology(
    const tt::tt_metal::TensorTopology& input_topology,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t dim,
    std::optional<uint32_t> cluster_axis,
    uint32_t rank);
}  // namespace detail

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor mesh_partition(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt);
}  // namespace ttnn::prim
