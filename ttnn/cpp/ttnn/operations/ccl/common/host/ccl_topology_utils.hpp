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
// order, an N-D label whose axes do not align with the mesh axes, a gather that would interleave shards, a result no
// label can express) is a TT_FATAL when `ttnn::CONFIG.strict_ccl_topology` is set, and otherwise a log line (once per
// distinct message per process) followed by `std::nullopt`. A device op then returns the input's label from
// compute_output_topologies (explicitly: returning `{}` would hand the framework the union over every tensor
// argument, persistent output buffers included) and a host relabel site skips the relabel, leaving the tensor with the
// label its last op gave it (composite_all_gather: the tail's label). Strict is off by default (existing models keep
// running) and is intended to be enabled in CI as a follow-up.
//
// The fallback is not equally safe for the two families. For all_gather / all_broadcast / all_reduce the input's
// label never adds a Replicate the output does not have, so a refusal is logged at warning level. For the
// reduce_scatter family the input's label can keep a Replicate on an axis whose devices now hold different pieces
// (N-D [Replicate, Shard{d}] scattered on d along axis 0), which the serialiser deduplicates: refusals are logged at
// error level and strict mode is the only safe mode for that family.
//
// Intermediates: a host op that labels its result from its own input (all_reduce_async: a reduce_scatter + all_gather,
// or an all_gather of an unsqueezed tensor + a local sum) runs device ops on tensors whose labels nobody reads. It
// wraps them in a CallerRelabelsScope, inside which a refusal returns nullopt silently -- no log, no TT_FATAL -- so a
// legitimate all_reduce is not refused for an intermediate no label describes (a collapsed Shard{d} reduce-scattered
// on d along the outer axis is column-major). The op's own label is computed outside the scope, so a bad input still
// fails loudly.
namespace ttnn::operations::ccl::common {

using TopologyPlacement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using TopologyPlacements = ttsl::SmallVector<TopologyPlacement>;

// While an object of this class is alive on the current thread, every refusal in this file returns nullopt without
// logging or throwing (see "Intermediates" above). Scopes nest. Not copyable.
class CallerRelabelsScope {
public:
    CallerRelabelsScope();
    ~CallerRelabelsScope();
    CallerRelabelsScope(const CallerRelabelsScope&) = delete;
    CallerRelabelsScope& operator=(const CallerRelabelsScope&) = delete;
};

// Expands `in` to one placement per axis of `mesh_shape`. An N-D label (distribution dims == placements == mesh dims)
// is returned verbatim, provided its axes align with the mesh axes: its distribution shape fits the mesh per axis and
// its coordinates walk a distribution-shaped block of the mesh in row-major order (an N-D mapper whose shape fits, with
// or without an offset). A mapper whose mesh_shape_override does not fit distributes in row-major order over the whole
// mesh, so axis i of its label is not axis i of the mesh and no per-axis edit describes it: not expandable. A
// collapsed 1-D label must cover the whole mesh in row-major order; Replicate expands to Replicate on every axis,
// Shard{d} to Shard{d} on every axis of size > 1 (row-major hierarchical sharding) and Replicate on size-1 axes.
// Anything else is not expandable: nullopt is returned and `reason` (if given) says why.
std::optional<TopologyPlacements> uncollapse_placements(
    const tt::tt_metal::TensorTopology& in,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    std::string* reason = nullptr);

// Label of an all_gather output. `cluster_axis` becomes Replicate; with no cluster_axis every placement becomes
// Replicate and the input's distribution shape and coordinates are kept. A collapsed input on a mesh with a single
// axis of size > 1 keeps its collapsed spelling (`{N},[Replicate]`); on a multi-axis mesh it is expanded to one
// placement per axis. `gathered_dim` may be negative or out of range (out of range never matches a Shard). With
// `require_contiguous_gather`, gathering a dim that is sharded on more than one mesh axis (a collapsed Shard, or an
// N-D label that spells the same dim on two axes, both read as row-major hierarchical sharding) along an axis that is
// not the innermost still-sharded one is a failure: the ring order is the row-major device order, so the pieces would
// interleave and no label describes the result. Gathering a different dim than the one the input is sharded on is
// honest along any axis. Ops that do not concatenate (all_reduce, all_broadcast) pass false. An out-of-range
// `cluster_axis` returns nullopt without a warning: the op's own validation rejects it right after the hook.
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

// Label of a reduce_scatter output. A reduction sums over the devices it spans, so whatever those devices held is
// consumed: `cluster_axis` becomes Shard{scatter_dim} (normalised) whatever placement it held, and the other axes
// keep theirs. Without a cluster_axis the sum spans the whole ring and every device holds a distinct piece in ring
// order, so the result is the collapsed `{N},[Shard{scatter_dim}]` over the input's coordinates (the ring order the
// pure overload has to assume) whatever the input placements were. A size-1 axis that sharded `scatter_dim` becomes
// Replicate (one chunk is the whole extent). When another non-trivial axis shards `scatter_dim` the result is only
// expressible when `cluster_axis` is inner (higher index) to every such axis, every other axis is trivial and nothing
// else is sharded: outer coarse pieces then inner fine pieces is exactly row-major hierarchical sharding, returned as
// the collapsed label carrying the input's coordinates. Scattering along an outer axis lays the pieces out
// column-major (device (r, c) holds piece c * R + r), which no label describes: a failure.
std::optional<tt::tt_metal::TensorTopology> reduce_scatter_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t scatter_dim);

// reduce_scatter's whole-mesh ring rank is the index into the tensor's device-storage coordinates (ccl_common.cpp,
// get_linearized_index_from_physical_coord without a cluster_axis), so this overload spells a whole-mesh label over
// those coordinates rather than the input label's -- for a tensor stored on every device of the mesh. On a multi-host
// mesh the storage holds this host's shard while the label is global, so the label's coordinates are kept (as they
// are for a single-host sub-mesh tensor, whose label already lists its own devices). With a cluster_axis the ring rank
// is the coordinate value itself and the label's coordinates are kept. This spelling is reduce_scatter's: the
// mesh_partition / all_to_all Tensor overloads below return the pure result, since those ops rank differently.
std::optional<tt::tt_metal::TensorTopology> reduce_scatter_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t scatter_dim);

// mesh_partition and all_to_all scatter `out_dim` along `cluster_axis` like reduce_scatter, but nothing is summed, so
// a Shard of another dim is not consumed: a whole-mesh scatter of a tensor still sharded on another dim along a
// non-trivial axis, and a scatter along an axis that itself holds a Shard of another dim (each device would hold a
// piece of two dims on one axis), are refusals (reduce_scatter consumes both). A stale Shard dim left by a
// rank-changing op (#52331) matches nothing, which the gather and reduce rules treat as harmless, but counts as a
// Shard of another dim here, so a partition refuses it. No op in this change calls these two; the stacked changes
// that move mesh_partition, all_to_all_async and their fused variants onto the helper do, which is why they live here.
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
