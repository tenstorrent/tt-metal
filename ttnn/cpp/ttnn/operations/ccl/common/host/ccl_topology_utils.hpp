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

#include "ttnn/tensor/tensor.hpp"

// Output TensorTopology labels for the collective ops (all_gather, all_broadcast, all_reduce, reduce_scatter and the
// mesh_partition / all_to_all family that scatters like reduce_scatter).
//
// The label contract: a tensor's TensorTopology describes the bytes each device holds. The mesh composer and the
// serialiser trust it -- a Replicate axis is deduplicated to one shard -- so an over-claimed Replicate is data loss
// and an under-claimed one is a spurious concat. Every op in the family used to edit the input's placements by
// mesh-axis index, which is only meaningful for an N-D label (one placement per mesh axis). The collapsed 1-D label
// `{N},[placement]` that ShardTensorToMesh / ReplicateTensorToMesh produce has a single placement for the whole mesh,
// so the per-axis edit landed on index 0 whatever `cluster_axis` was. These helpers expand a collapsed label to per
// mesh axis first, apply the op's edit on `cluster_axis`, then decide whether the result is still expressible per axis
// or has to collapse back to a 1-D label.
//
// Failure handling: a label the helper cannot honour (a collapsed label that does not cover the mesh in row-major
// order, a gather that would interleave shards, a result no label can express) is a TT_FATAL when
// `ttnn::CONFIG.strict_ccl_topology` is set, and otherwise a warning followed by the legacy per-axis edit. The strict
// mode is what CI runs; the default is warn-only for one release so existing models keep running.
namespace ttnn::operations::ccl::common {

using TopologyPlacement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using TopologyPlacements = ttsl::SmallVector<TopologyPlacement>;

// Normalises a possibly-negative tensor dim against `rank`; nullopt when out of range in either direction.
std::optional<uint32_t> normalize_tensor_dim(int dim, uint32_t rank);

// Expands `in` to one placement per axis of `mesh_shape`. An N-D label (distribution dims == placements == mesh dims)
// is returned verbatim. A collapsed 1-D label must cover the whole mesh in row-major order; Replicate expands to
// Replicate on every axis, Shard{d} to Shard{d} on every axis of size > 1 (row-major hierarchical sharding) and
// Replicate on size-1 axes. Anything else is not expandable: nullopt is returned and `reason` (if given) says why.
std::optional<TopologyPlacements> uncollapse_placements(
    const tt::tt_metal::TensorTopology& in,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    std::string* reason = nullptr);

// Label of an all_gather output. `cluster_axis` becomes Replicate; with no cluster_axis every placement becomes
// Replicate and the input's distribution shape and coordinates are kept. `gathered_dim` may be negative or out of
// range (out of range never matches a Shard). With `require_contiguous_gather`, a collapsed Shard{gathered_dim} input
// gathered along an axis that is not the innermost still-sharded one is a failure: the ring order is the tensor
// coordinate order, so the pieces would interleave and no label describes the result. Ops that do not concatenate
// (all_reduce, all_broadcast) pass false.
tt::tt_metal::TensorTopology all_gather_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t gathered_dim,
    bool require_contiguous_gather = true);

tt::tt_metal::TensorTopology all_gather_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t gathered_dim, bool require_contiguous_gather = true);

// all_reduce and all_broadcast leave every device on `cluster_axis` with the same bytes: the all_gather label with no
// contiguity requirement.
tt::tt_metal::TensorTopology all_reduce_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank);
tt::tt_metal::TensorTopology all_reduce_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis);

tt::tt_metal::TensorTopology all_broadcast_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank);
tt::tt_metal::TensorTopology all_broadcast_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis);

// Label of a reduce_scatter output: `cluster_axis` becomes Shard{scatter_dim} (normalised). Without a cluster_axis a
// collapsed input keeps its distribution shape and coordinates with the single placement Shard{scatter_dim}; an N-D
// input gets Shard{scatter_dim} on every axis of size > 1. When the same tensor dim ends up sharded on more than one
// axis the result is only expressible when every non-trivial axis shards that dim and nothing else is sharded -- that
// is exactly row-major hierarchical sharding, returned as the collapsed label `{N},[Shard{dim}]` carrying the input's
// coordinates. Any other multi-axis sharding of one dim is a failure.
tt::tt_metal::TensorTopology reduce_scatter_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t scatter_dim);
tt::tt_metal::TensorTopology reduce_scatter_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t scatter_dim);

// mesh_partition and all_to_all scatter `out_dim` along `cluster_axis` exactly like reduce_scatter.
tt::tt_metal::TensorTopology mesh_partition_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim);
tt::tt_metal::TensorTopology mesh_partition_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim);

tt::tt_metal::TensorTopology all_to_all_output_topology(
    const tt::tt_metal::TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim);
tt::tt_metal::TensorTopology all_to_all_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim);

}  // namespace ttnn::operations::ccl::common
