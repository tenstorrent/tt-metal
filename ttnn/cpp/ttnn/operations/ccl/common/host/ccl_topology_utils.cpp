// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"

#include <sstream>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/config.hpp"

namespace ttnn::operations::ccl::common {

using tt::tt_metal::TensorTopology;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshShape;
using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;

std::optional<uint32_t> normalize_tensor_dim(int dim, uint32_t rank) {
    const int normalized_dim = dim < 0 ? static_cast<int>(rank) + dim : dim;
    if (normalized_dim < 0 || normalized_dim >= static_cast<int>(rank)) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(normalized_dim);
}

namespace {

std::string describe(const TensorTopology& topology) {
    std::ostringstream ss;
    ss << topology;
    return ss.str();
}

// A 1-D label with a single placement for the whole mesh. On a 1-D mesh this is also the N-D form, so callers that
// need the distinction also check the mesh rank.
bool is_collapsed(const TensorTopology& in) {
    return in.distribution_shape().dims() == 1 && in.placements().size() == 1;
}

// Shard dims are stored as the caller spelled them (possibly negative, possibly stale after a rank change), so they
// are compared normalised; an out-of-range dim never matches anything.
bool is_shard_of(const TopologyPlacement& placement, uint32_t tensor_dim, uint32_t rank) {
    const auto* shard = std::get_if<Shard>(&placement);
    return shard != nullptr && normalize_tensor_dim(shard->dim, rank) == tensor_dim;
}

// The label's coordinates are exactly MeshCoordinateRange(mesh_shape), in order. That is what makes the collapsed
// ring index equal the row-major device index, which the expansion below relies on.
bool covers_mesh_row_major(const std::vector<MeshCoordinate>& coords, const MeshShape& mesh_shape) {
    if (coords.size() != mesh_shape.mesh_size()) {
        return false;
    }
    // The host-side factories spell a 1-D mesh's coordinates as (0, i); accept that spelling too.
    const MeshShape range_shape = (mesh_shape.dims() == 1 && !coords.empty() && coords.front().dims() == 2)
                                      ? MeshShape(1, mesh_shape[0])
                                      : mesh_shape;
    size_t index = 0;
    for (const auto& coord : MeshCoordinateRange(range_shape)) {
        if (coords[index++] != coord) {
            return false;
        }
    }
    return true;
}

bool strict_ccl_topology() { return ttnn::CONFIG.get<"strict_ccl_topology">(); }

// The per-axis edit every op in the family applied before this helper existed: `cluster_axis` (or every entry when
// there is none) takes `placement`, whatever the label's rank. Kept only as the warn-only fallback.
TensorTopology legacy_relabel(
    const TensorTopology& in, std::optional<uint32_t> cluster_axis, const TopologyPlacement& placement) {
    auto out = in.placements();
    if (cluster_axis.has_value()) {
        if (*cluster_axis < out.size()) {
            out[*cluster_axis] = placement;
        }
    } else {
        for (auto& entry : out) {
            entry = placement;
        }
    }
    return TensorTopology(in.distribution_shape(), std::move(out), in.mesh_coords());
}

TensorTopology fail(
    const std::string& what,
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const TopologyPlacement& placement) {
    if (strict_ccl_topology()) {
        TT_THROW("{}", what);
    }
    log_warning(
        tt::LogOp,
        "{}; keeping the legacy per-axis label. This becomes an error when ttnn.CONFIG.strict_ccl_topology is set "
        "(the default in a future release).",
        what);
    return legacy_relabel(in, cluster_axis, placement);
}

// Decides whether `out` is expressible with one placement per axis. A tensor dim sharded on more than one axis is
// only expressible as the collapsed row-major label, and only when every non-trivial axis shards that dim and
// nothing else is sharded: that is exactly what a 1-D mapper produces, so the 1-D concat composer reassembles it.
std::optional<TensorTopology> finalise(
    const TensorTopology& in,
    const MeshShape& mesh_shape,
    TopologyPlacements out,
    uint32_t rank,
    bool collapsed_input,
    std::string* reason) {
    const MeshShape& axis_shape = collapsed_input ? mesh_shape : in.distribution_shape();

    std::optional<uint32_t> duplicated_dim;
    for (size_t i = 0; i < out.size() && !duplicated_dim.has_value(); ++i) {
        const auto* shard = std::get_if<Shard>(&out[i]);
        if (shard == nullptr) {
            continue;
        }
        const auto dim = normalize_tensor_dim(shard->dim, rank);
        if (!dim.has_value()) {
            continue;
        }
        for (size_t j = i + 1; j < out.size(); ++j) {
            if (is_shard_of(out[j], *dim, rank)) {
                duplicated_dim = dim;
                break;
            }
        }
    }

    if (!duplicated_dim.has_value()) {
        return TensorTopology(axis_shape, std::move(out), in.mesh_coords());
    }

    bool row_major_hierarchical = true;
    for (size_t axis = 0; axis < out.size(); ++axis) {
        const bool shards_duplicated_dim = is_shard_of(out[axis], *duplicated_dim, rank);
        if (axis_shape[static_cast<int32_t>(axis)] > 1) {
            row_major_hierarchical &= shards_duplicated_dim;
        } else {
            row_major_hierarchical &= shards_duplicated_dim || !std::holds_alternative<Shard>(out[axis]);
        }
    }
    if (!row_major_hierarchical) {
        if (reason != nullptr) {
            *reason = fmt::format(
                "the output would shard tensor dim {} on more than one mesh axis in a layout no TensorTopology can "
                "express (per-axis result {} for input {})",
                *duplicated_dim,
                describe(TensorTopology(axis_shape, out, in.mesh_coords())),
                describe(in));
        }
        return std::nullopt;
    }
    return TensorTopology(
        MeshShape(static_cast<uint32_t>(axis_shape.mesh_size())),
        TopologyPlacements{Shard{static_cast<int>(*duplicated_dim)}},
        in.mesh_coords());
}

MeshShape mesh_shape_of(const Tensor& input, const char* op) {
    TT_FATAL(input.device() != nullptr, "{} output topology requires a mesh-device tensor", op);
    return input.device()->shape();
}

uint32_t rank_of(const Tensor& input) { return static_cast<uint32_t>(input.logical_shape().rank()); }

}  // namespace

std::optional<TopologyPlacements> uncollapse_placements(
    const TensorTopology& in, const MeshShape& mesh_shape, std::string* reason) {
    const auto& distribution_shape = in.distribution_shape();
    const auto& placements = in.placements();
    const size_t mesh_dims = mesh_shape.dims();
    const auto set_reason = [reason](std::string text) {
        if (reason != nullptr) {
            *reason = std::move(text);
        }
    };

    if (distribution_shape.dims() == mesh_dims && placements.size() == mesh_dims) {
        return placements;
    }
    if (!is_collapsed(in)) {
        set_reason(fmt::format(
            "tensor topology {} is neither a collapsed 1-D label nor one placement per axis of mesh {}",
            describe(in),
            mesh_shape));
        return std::nullopt;
    }
    if (distribution_shape.mesh_size() != mesh_shape.mesh_size() ||
        !covers_mesh_row_major(in.mesh_coords(), mesh_shape)) {
        set_reason(fmt::format(
            "collapsed tensor topology {} does not cover mesh {} in row-major order (fewer shards than devices, or a "
            "sub-mesh); create the tensor with an N-D mesh mapper",
            describe(in),
            mesh_shape));
        return std::nullopt;
    }

    TopologyPlacements out(mesh_dims, TopologyPlacement{Replicate{}});
    if (const auto* shard = std::get_if<Shard>(&placements.front())) {
        for (size_t axis = 0; axis < mesh_dims; ++axis) {
            if (mesh_shape[static_cast<int32_t>(axis)] > 1) {
                out[axis] = *shard;
            }
        }
    }
    return out;
}

TensorTopology all_gather_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t gathered_dim,
    bool require_contiguous_gather) {
    const TopologyPlacement replicate = Replicate{};
    if (!cluster_axis.has_value()) {
        // Whole-mesh gather: every device ends up with every piece, whatever the label's rank.
        TopologyPlacements out(in.placements().size(), replicate);
        return TensorTopology(in.distribution_shape(), std::move(out), in.mesh_coords());
    }

    const uint32_t axis = *cluster_axis;
    if (axis >= mesh_shape.dims()) {
        return fail(
            fmt::format("all_gather cluster_axis {} is out of range for mesh {}", axis, mesh_shape),
            in,
            cluster_axis,
            replicate);
    }
    std::string reason;
    auto expanded = uncollapse_placements(in, mesh_shape, &reason);
    if (!expanded.has_value()) {
        return fail(reason, in, cluster_axis, replicate);
    }

    // A 1-D mapper shards in row-major device order, so after expanding, a gather along an outer axis would
    // interleave pieces that an inner axis still keeps apart. Only the gathered dim can interleave; gathering some
    // other dim leaves the shards of `gathered_dim` where they are, so either axis is honest then.
    const bool collapsed = is_collapsed(in) && mesh_shape.dims() > 1;
    const auto gathered = normalize_tensor_dim(gathered_dim, tensor_rank);
    if (collapsed && require_contiguous_gather && gathered.has_value() && mesh_shape[static_cast<int32_t>(axis)] > 1 &&
        is_shard_of((*expanded)[axis], *gathered, tensor_rank)) {
        for (size_t inner = axis + 1; inner < expanded->size(); ++inner) {
            if (mesh_shape[static_cast<int32_t>(inner)] > 1 && is_shard_of((*expanded)[inner], *gathered, tensor_rank)) {
                return fail(
                    fmt::format(
                        "all_gather of tensor dim {} along mesh axis {} would interleave the shards of a tensor "
                        "distributed with a 1-D mapper over mesh {} (label {}): the ring order is the row-major "
                        "device order, so gather along the innermost sharded mesh axis or create the tensor with an "
                        "N-D mesh mapper",
                        *gathered,
                        axis,
                        mesh_shape,
                        describe(in)),
                    in,
                    cluster_axis,
                    replicate);
            }
        }
    }

    auto out = std::move(*expanded);
    out[axis] = replicate;
    auto result = finalise(in, mesh_shape, std::move(out), tensor_rank, collapsed, &reason);
    if (!result.has_value()) {
        return fail(reason, in, cluster_axis, replicate);
    }
    return *result;
}

TensorTopology all_gather_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t gathered_dim, bool require_contiguous_gather) {
    return all_gather_output_topology(
        input.tensor_topology(),
        cluster_axis,
        mesh_shape_of(input, "all_gather"),
        rank_of(input),
        gathered_dim,
        require_contiguous_gather);
}

TensorTopology all_reduce_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank) {
    return all_gather_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, /*gathered_dim=*/0, /*require_contiguous_gather=*/false);
}

TensorTopology all_reduce_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis) {
    return all_reduce_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "all_reduce"), rank_of(input));
}

TensorTopology all_broadcast_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank) {
    return all_gather_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, /*gathered_dim=*/0, /*require_contiguous_gather=*/false);
}

TensorTopology all_broadcast_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis) {
    return all_broadcast_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "all_broadcast"), rank_of(input));
}

TensorTopology reduce_scatter_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t scatter_dim) {
    const auto normalized_dim = normalize_tensor_dim(scatter_dim, tensor_rank);
    if (!normalized_dim.has_value()) {
        return fail(
            fmt::format("reduce_scatter dim {} is out of range for a rank-{} tensor", scatter_dim, tensor_rank),
            in,
            cluster_axis,
            Shard{scatter_dim});
    }
    // Only normalised dims are ever written into a placement the helper creates.
    const TopologyPlacement shard = Shard{static_cast<int>(*normalized_dim)};
    const bool collapsed = is_collapsed(in) && mesh_shape.dims() > 1;
    std::string reason;

    if (!cluster_axis.has_value()) {
        if (collapsed) {
            // Whole-mesh scatter of a 1-D label: piece i lands on ring rank i, which is the label's own order.
            return TensorTopology(in.distribution_shape(), TopologyPlacements{shard}, in.mesh_coords());
        }
        auto expanded = uncollapse_placements(in, mesh_shape, &reason);
        if (!expanded.has_value()) {
            return fail(reason, in, cluster_axis, shard);
        }
        // Every axis with more than one device takes a piece; a size-1 axis holds the whole of its piece.
        const auto& axis_shape = in.distribution_shape();
        for (size_t axis = 0; axis < expanded->size(); ++axis) {
            (*expanded)[axis] = axis_shape[static_cast<int32_t>(axis)] > 1 ? shard : TopologyPlacement{Replicate{}};
        }
        auto result = finalise(in, mesh_shape, std::move(*expanded), tensor_rank, /*collapsed_input=*/false, &reason);
        if (!result.has_value()) {
            return fail(reason, in, cluster_axis, shard);
        }
        return *result;
    }

    const uint32_t axis = *cluster_axis;
    if (axis >= mesh_shape.dims()) {
        return fail(
            fmt::format("reduce_scatter cluster_axis {} is out of range for mesh {}", axis, mesh_shape),
            in,
            cluster_axis,
            shard);
    }
    auto expanded = uncollapse_placements(in, mesh_shape, &reason);
    if (!expanded.has_value()) {
        return fail(reason, in, cluster_axis, shard);
    }
    auto out = std::move(*expanded);
    out[axis] = shard;
    auto result = finalise(in, mesh_shape, std::move(out), tensor_rank, collapsed, &reason);
    if (!result.has_value()) {
        return fail(reason, in, cluster_axis, shard);
    }
    return *result;
}

TensorTopology reduce_scatter_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t scatter_dim) {
    return reduce_scatter_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "reduce_scatter"), rank_of(input), scatter_dim);
}

TensorTopology mesh_partition_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim) {
    return reduce_scatter_output_topology(in, cluster_axis, mesh_shape, tensor_rank, out_dim);
}

TensorTopology mesh_partition_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim) {
    return mesh_partition_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "mesh_partition"), rank_of(input), out_dim);
}

TensorTopology all_to_all_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim) {
    return reduce_scatter_output_topology(in, cluster_axis, mesh_shape, tensor_rank, out_dim);
}

TensorTopology all_to_all_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim) {
    return all_to_all_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "all_to_all"), rank_of(input), out_dim);
}

}  // namespace ttnn::operations::ccl::common
