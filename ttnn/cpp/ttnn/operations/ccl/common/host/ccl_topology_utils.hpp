// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <tt_stl/small_vector.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/distributed_tensor_configs.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>

#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

// Output TensorTopology labels for the collective ops (all_gather, all_broadcast, all_reduce, reduce_scatter and the
// mesh_partition / all_to_all family that scatters like reduce_scatter).
//
// The label contract: a tensor's TensorTopology describes the bytes each device holds. The mesh composer and the
// serialiser trust it -- a Replicate axis is deduplicated to one shard -- so an over-claimed Replicate is data loss
// and an under-claimed one is a spurious concat. Every op in the family used to edit the input's placements by
// mesh-axis index, which is only meaningful for an N-D label (one placement per mesh axis). The collapsed 1-D label
// `{N},[placement]` that ShardTensorToMesh / ReplicateTensorToMesh produce has a single placement for the whole mesh,
// so the per-axis edit landed on index 0 whatever `cluster_axis` was. These helpers expand a collapsed label to one
// placement per mesh axis first, apply the op's edit on `cluster_axis`, then decide whether the result is still
// expressible per axis or has to collapse back to a 1-D label.
//
// Failure handling: a label the helper cannot honour (a collapsed label that does not cover the mesh in row-major
// order, a gather that would interleave shards, a result no label can express) is a TT_FATAL when
// `ttnn::CONFIG.strict_ccl_topology` is set, and otherwise a log line (once per distinct message per process)
// followed by `std::nullopt`. A device op returns `{}` from compute_output_topologies on nullopt so launch() keeps
// the union default (the input's label, as mesh_partition does today); a host relabel site skips the relabel. Strict
// is off by default (existing models keep running) and is intended to be enabled in CI as a follow-up.
//
// The fallback is not equally safe for the two families. For all_gather / all_broadcast / all_reduce the input's
// label never adds a Replicate the output does not have, so a refusal is logged at warning level. For the
// reduce_scatter family the input's label can keep a Replicate on an axis whose devices now hold different pieces
// (N-D [Replicate, Shard{d}] scattered on d along axis 0), which the serialiser deduplicates: refusals are logged at
// error level and strict mode is the only safe mode for that family.
//
// Known strict-mode limitation: all_reduce_async's reduce_scatter + all_gather path (and its whole-mesh overload) runs
// reduce_scatter_minimal_async and all_gather_async on intermediates and relabels the final tensor honestly from the
// original input, but those device ops have no caller-relabels switch, so a refusal on an intermediate (a collapsed
// Shard{d} reduced along the outer axis when the scatter dim is d; an N-D input sharded on another dim through the
// whole-mesh overload) TT_FATALs under strict mode and logs under warn-only although the final label would be right.
namespace ttnn::operations::ccl::common {

using TopologyPlacement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using TopologyPlacements = ttsl::SmallVector<TopologyPlacement>;

// Expands `in` to one placement per axis of `mesh_shape`. An N-D label (distribution dims == placements == mesh dims)
// is returned verbatim. A collapsed 1-D label must cover the whole mesh in row-major order; Replicate expands to
// Replicate on every axis, Shard{d} to Shard{d} on every axis of size > 1 (row-major hierarchical sharding) and
// Replicate on size-1 axes. Anything else is not expandable: nullopt is returned and `reason` (if given) says why.
std::optional<TopologyPlacements> uncollapse_placements(
    const tt::tt_metal::TensorTopology& in,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    std::string* reason = nullptr);

// Label of an all_gather output. `cluster_axis` becomes Replicate; with no cluster_axis every placement becomes
// Replicate and the input's distribution shape and coordinates are kept. A collapsed input on a mesh with a single
// axis of size > 1 keeps its collapsed spelling (`{N},[Replicate]`); on a multi-axis mesh it is expanded to one
// placement per axis. `gathered_dim` may be negative or out of range (out of range never matches a Shard). With
// `require_contiguous_gather`, a collapsed Shard{gathered_dim} input gathered along an axis that is not the innermost
// still-sharded one is a failure: the ring order is the tensor coordinate order, so the pieces would interleave and
// no label describes the result. Gathering a different dim than the one the input is sharded on is honest along any
// axis. Ops that do not concatenate (all_reduce, all_broadcast) pass false. An out-of-range `cluster_axis` returns
// nullopt without a warning: the op's own validation rejects it right after the hook.
std::optional<tt::tt_metal::TensorTopology> all_gather_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t gathered_dim,
    bool require_contiguous_gather = true);

// Mesh shape from `input.device()->shape()`, rank from `input.logical_shape()`.
std::optional<tt::tt_metal::TensorTopology> all_gather_output_topology(
    const Tensor& input,
    std::optional<uint32_t> cluster_axis,
    int32_t gathered_dim,
    bool require_contiguous_gather = true);

// all_reduce and all_broadcast leave every device on `cluster_axis` with the same bytes: the all_gather label with no
// contiguity requirement.
std::optional<tt::tt_metal::TensorTopology> all_reduce_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank);
std::optional<tt::tt_metal::TensorTopology> all_reduce_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis);

std::optional<tt::tt_metal::TensorTopology> all_broadcast_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank);
std::optional<tt::tt_metal::TensorTopology> all_broadcast_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis);

// Label of a reduce_scatter output: `cluster_axis` becomes Shard{scatter_dim} (normalised), whatever placement it
// held (an existing Shard of another dim on that axis is overwritten, as reduce_scatter_minimal_async always did).
// Without a cluster_axis every device holds a distinct piece in the label's coordinate order, so the result is the
// collapsed `{N},[Shard{scatter_dim}]` over the input's coordinates -- unless another dim stays sharded on a
// non-trivial axis, which no label can state next to the new piece. A size-1 axis that sharded `scatter_dim` becomes
// Replicate (one chunk is the whole extent). When another non-trivial axis shards `scatter_dim` the result is only
// expressible when `cluster_axis` is inner (higher index) to every such axis, every other axis is trivial, nothing else
// is sharded and `cluster_axis` held Replicate or Shard{scatter_dim}: outer coarse pieces then inner fine pieces is
// exactly row-major hierarchical sharding, returned as the collapsed label carrying the input's coordinates. Scattering
// along an outer axis lays the pieces out column-major, which no label describes: a failure. These are the
// mesh_partition rules (MeshPartitionDeviceOperation::compute_output_topologies), which 1b-A moves onto this helper.
std::optional<tt::tt_metal::TensorTopology> reduce_scatter_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t scatter_dim);
std::optional<tt::tt_metal::TensorTopology> reduce_scatter_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t scatter_dim);

// mesh_partition and all_to_all scatter `out_dim` along `cluster_axis` exactly like reduce_scatter.
std::optional<tt::tt_metal::TensorTopology> mesh_partition_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim);
std::optional<tt::tt_metal::TensorTopology> mesh_partition_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim);

std::optional<tt::tt_metal::TensorTopology> all_to_all_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim);
std::optional<tt::tt_metal::TensorTopology> all_to_all_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim);

}  // namespace ttnn::operations::ccl::common
