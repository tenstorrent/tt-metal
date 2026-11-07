// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

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
// and an under-claimed one is a spurious concat. Editing the input's placements by mesh-axis index is only
// meaningful for an N-D label (one placement per mesh axis): the collapsed 1-D label `{N},[placement]` that
// ShardTensorToMesh / ReplicateTensorToMesh produce has a single placement for the whole mesh, so a per-axis edit of
// it would land on index 0 whatever `cluster_axis` was. These helpers expand a collapsed label to one placement per
// mesh axis first, apply the op's edit on `cluster_axis`, then decide whether the result is still expressible per
// axis or has to collapse back to a 1-D label.
//
// Coverage: the devices an op reaches are the mesh's, not the label's. A cluster-axis op spans the mesh's full extent
// along that axis and a whole-mesh op every device, so a label that covers fewer devices along the axis addressed (a
// (1, 4) row block on a 2x4 along axis 0, or over the whole mesh) is refused: the devices the label does not describe
// take part and no label over the block can say what they hold. A block is gathered or scattered within itself only
// along an axis it spans fully (the row along axis 1).
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
// wraps them in a SilenceIntermediateRefusals scope, inside which a refusal returns nullopt silently -- no log, no
// TT_FATAL -- so a legitimate all_reduce is not refused for an intermediate no label describes (a collapsed Shard{d}
// reduce-scattered on d along the outer axis is column-major). The op's own label is computed outside the scope, so a
// bad input still fails loudly.
namespace ttnn::operations::ccl::common {

using TopologyPlacement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using TopologyPlacements = ttsl::SmallVector<TopologyPlacement>;

// While an object of this class is alive on the current thread, every refusal in this file returns nullopt without
// logging or throwing (see "Intermediates" above). Scopes nest. Not copyable.
class SilenceIntermediateRefusals {
public:
    SilenceIntermediateRefusals();
    ~SilenceIntermediateRefusals();
    SilenceIntermediateRefusals(const SilenceIntermediateRefusals&) = delete;
    SilenceIntermediateRefusals& operator=(const SilenceIntermediateRefusals&) = delete;
};

// Which family a refusal belongs to (see "The fallback is not equally safe" above): a gather-family refusal is logged
// at warning level, a scatter-family one at error level.
enum class RefusalFamily { Gather, Scatter };

// A refusal raised by a device op's hook rather than by a helper above: a label the op cannot renumber into its
// output layout, a ring order no label can carry. Handled exactly like the helpers' own refusals -- silent nullopt
// inside a SilenceIntermediateRefusals scope, TT_FATAL under strict mode, otherwise one log line per distinct message
// -- so a hook's "no honest label" is as visible as the helper's and strict mode covers it. Always returns nullopt (or
// throws), so a hook writes `return {refuse(what, family).value_or(input_topology)};` or calls it and returns the
// input's label.
std::optional<tt::tt_metal::TensorTopology> refuse(const std::string& what, RefusalFamily family);

// Expands `in` to one placement per axis of `mesh_shape`. An N-D label (distribution dims == placements == mesh dims)
// is returned verbatim, provided its axes align with the mesh axes: its distribution shape fits the mesh per axis and
// its coordinates walk a distribution-shaped block of the mesh in row-major order (an N-D mapper whose shape fits, with
// or without an offset). A mapper whose mesh_shape_override does not fit distributes in row-major order over the whole
// mesh, so axis i of its label is not axis i of the mesh and no per-axis edit describes it: not expandable. The
// coordinates are only consulted when there are as many as the distribution has devices: a device tensor's label
// lists its storage's coordinates (every device of the mesh, however small the block its input occupied), so a
// sub-range label on a device tensor is judged by its shape alone -- it fits, so it was mapped as a block. A
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

// The respelling the Tensor overload above applies, for a caller whose reduce_scatter input is a label rather than a
// tensor (a fused op reduce-scatters its own matmul result, which no tensor carries when its hook runs) or whose op
// ranks a whole-mesh ring the same way (all_to_all_async_generic): a whole-mesh `label` (cluster_axis nullopt) is
// spelled over `ring_tensor`'s device-storage coordinates -- the tensor the op's program factory passes to
// get_linearized_index_from_physical_coord -- when that tensor is stored on every device of the mesh. With a
// cluster_axis, a tensor that is not stored on every device, or no label, `label` is returned as is. A label whose
// coordinate count differs from the storage's is a scatter-family refusal naming `op`.
//
// What it guards against, on a 1x4 line reduce-scattering dim 3 over the whole mesh (piece i goes to ring rank i,
// the index into the storage coordinates):
//   storage coordinates (ring order):   (0,0) (0,1) (0,2) (0,3)   -> the devices hold S[0] S[1] S[2] S[3]
//   label coordinates of a tensor built from an explicit list (from_host_shards, combine_device_tensors):
//                                       (0,2) (0,3) (0,0) (0,1)
//   pure result `{4},[Shard{3}]` over the label's coordinates claims piece 0 is on (0,2), which holds S[2]: composing
//   by it reads the pieces scrambled. Over the storage coordinates it names the device that holds each piece.
std::optional<tt::tt_metal::TensorTopology> over_storage_ring_order(
    std::optional<tt::tt_metal::TensorTopology> label,
    const Tensor& ring_tensor,
    std::optional<uint32_t> cluster_axis,
    std::string_view op = "reduce_scatter");

// mesh_partition and all_to_all scatter `out_dim` along `cluster_axis` like reduce_scatter, but nothing is summed, so
// a Shard of another dim is not consumed: a whole-mesh scatter of a tensor still sharded on another dim along a
// non-trivial axis, and a scatter along an axis that itself holds a Shard of another dim (each device would hold a
// piece of two dims on one axis), are refusals (reduce_scatter consumes both). A stale Shard dim left by a
// rank-changing op (#52331) matches nothing, which the gather and reduce rules treat as harmless, but counts as a
// Shard of another dim here, so a partition refuses it. mesh_partition's hook calls mesh_partition_output_topology;
// all_to_all_output_topology is what the stacked experimental-CCL change calls for all_to_all_async and its
// fused variants, which is why both live here. The mesh_partition Tensor overload spells a whole-mesh label over the
// storage's coordinates like reduce_scatter's (the program factory numbers devices row-major over the mesh, the
// storage order of a tensor stored on every device); the all_to_all one returns the pure result, since those ops'
// hooks handle their own ring order.
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

/**
 * The label of an output computed from several labelled operands (a fused matmul of a collective's result by a
 * weight, an elementwise op on a reduce_scatter result), as a TensorTopology over `mesh_coords`: the union launch()
 * applies to plain ops (ttnn::device_operation::detail::compute_output_placements_and_shape), with
 * `operands.front()` the label the operation computed itself, plus one rule: a sharded first operand fixes the
 * result's distribution rank and sharded operands of another rank are left out, so a collapsed {N},[Shard{d}]
 * collective result is not out-voted by N-D operands into Replicate on an axis whose devices hold distinct pieces.
 */
tt::tt_metal::TensorTopology union_topology(
    const std::vector<std::reference_wrapper<const tt::tt_metal::TensorTopology>>& operands,
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords);

}  // namespace ttnn::operations::ccl::common
