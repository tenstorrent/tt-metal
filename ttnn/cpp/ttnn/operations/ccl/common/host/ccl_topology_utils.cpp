// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"

#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
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

// The one axis of size > 1 of a mesh that is a line (1xN, Nx1, ...); nullopt when the mesh has none or several.
std::optional<size_t> single_non_trivial_axis(const MeshShape& mesh_shape) {
    std::optional<size_t> found;
    for (size_t axis = 0; axis < mesh_shape.dims(); ++axis) {
        if (mesh_shape[static_cast<int32_t>(axis)] > 1) {
            if (found.has_value()) {
                return std::nullopt;
            }
            found = axis;
        }
    }
    return found;
}

// The label's coordinates are exactly MeshCoordinateRange(mesh_shape), in order. That is what makes the collapsed
// ring index equal the row-major device index, which the expansion in uncollapse_placements relies on.
bool covers_mesh_row_major(const std::vector<MeshCoordinate>& coords, const MeshShape& mesh_shape) {
    if (coords.size() != mesh_shape.mesh_size()) {
        return false;
    }
    // TensorTopology's own factories spell a 1-D mesh's coordinates as (0, i); accept that spelling too.
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

// Which family refused: the fallback (the input's label) is safe for a gather family -- it never adds a Replicate the
// output lacks -- but can over-claim Replicate for the scatter family, so the latter logs at error level.
enum class Family { Gather, Scatter };

// Each distinct message is logged once per process: a model that trips the same refusal in every layer would
// otherwise flood the log.
bool first_time(const std::string& message) {
    static std::mutex mutex;
    static std::unordered_set<std::string> seen;
    const std::lock_guard<std::mutex> lock(mutex);
    return seen.insert(message).second;
}

// No honest label: TT_FATAL under strict mode, otherwise a log line and nullopt so the caller leaves the union default
// (the input's label) in place, as mesh_partition does today.
std::optional<TensorTopology> fail(const std::string& what, Family family) {
    if (strict_ccl_topology()) {
        TT_THROW("{}", what);
    }
    if (family == Family::Scatter) {
        const std::string message = fmt::format(
            "{}; the output keeps the union-default TensorTopology (the input's label), which may claim Replicate on a "
            "mesh axis whose devices now hold different pieces and so lose data when serialised. Strict mode "
            "(ttnn.CONFIG.strict_ccl_topology) is the only safe mode for reduce_scatter / mesh_partition / all_to_all "
            "labels.",
            what);
        if (first_time(message)) {
            log_error(tt::LogOp, "{}", message);
        }
    } else {
        const std::string message = fmt::format(
            "{}; the output keeps the union-default TensorTopology (the input's label). Set "
            "ttnn.CONFIG.strict_ccl_topology to make this an error.",
            what);
        if (first_time(message)) {
            log_warning(tt::LogOp, "{}", message);
        }
    }
    return std::nullopt;
}

// Decides whether `out` (one placement per axis of `axis_shape`) is expressible. A size-1 axis that shards a dim
// another axis also shards holds the whole extent of that dim, so it reads Replicate. After that, no tensor dim
// sharded on more than one axis: it is the N-D label as is. A dim sharded on more than one axis is only expressible
// as the collapsed row-major label, and only when every non-trivial axis shards that dim and nothing else is sharded:
// that is exactly what a 1-D mapper produces, so the 1-D concat composer reassembles it. `new_shard_axis` is the axis
// the caller has just set to Shard (reduce_scatter along a cluster_axis); the collapse is only honest when it is
// inner (higher index) to every other axis sharding that dim, because the ring splits the pieces the outer axes
// already hold (device (r, c) holds chunk r * C + c). Scattering along an outer axis lays the pieces out column-major,
// which no label describes (plan 1a.2(c) as amended 2026-09-30).
std::optional<TensorTopology> finalise(
    const TensorTopology& in,
    const MeshShape& axis_shape,
    TopologyPlacements out,
    uint32_t rank,
    std::optional<uint32_t> new_shard_axis,
    std::string* reason) {
    const auto shards_same_dim_as = [&](size_t axis, const Shard& shard) {
        const auto dim = normalize_tensor_dim(shard.dim, rank);
        if (!dim.has_value()) {
            return false;
        }
        for (size_t other = 0; other < out.size(); ++other) {
            if (other != axis && placement_shards_tensor_dim(out[other], *dim, rank)) {
                return true;
            }
        }
        return false;
    };
    for (size_t axis = 0; axis < out.size(); ++axis) {
        const auto* shard = std::get_if<Shard>(&out[axis]);
        if (shard != nullptr && axis_shape[static_cast<int32_t>(axis)] == 1 && shards_same_dim_as(axis, *shard)) {
            out[axis] = Replicate{};
        }
    }

    std::optional<uint32_t> duplicated_dim;
    for (size_t axis = 0; axis < out.size() && !duplicated_dim.has_value(); ++axis) {
        const auto* shard = std::get_if<Shard>(&out[axis]);
        if (shard != nullptr && shards_same_dim_as(axis, *shard)) {
            duplicated_dim = normalize_tensor_dim(shard->dim, rank);
        }
    }

    if (!duplicated_dim.has_value()) {
        return TensorTopology(axis_shape, std::move(out), in.mesh_coords());
    }

    bool row_major_hierarchical = true;
    for (size_t axis = 0; axis < out.size(); ++axis) {
        const bool shards_duplicated_dim = placement_shards_tensor_dim(out[axis], *duplicated_dim, rank);
        if (axis_shape[static_cast<int32_t>(axis)] > 1) {
            row_major_hierarchical &= shards_duplicated_dim;
            if (new_shard_axis.has_value() && axis > *new_shard_axis) {
                row_major_hierarchical = false;
            }
        } else {
            row_major_hierarchical &= !std::holds_alternative<Shard>(out[axis]);
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

std::optional<TensorTopology> all_gather_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t gathered_dim,
    bool require_contiguous_gather) {
    const TopologyPlacement replicate = Replicate{};
    if (!cluster_axis.has_value()) {
        // Whole-mesh gather: every device ends up with every piece, whatever the label's rank. The ring order is the
        // label's own coordinate order, so the distribution shape and coordinates are kept.
        TopologyPlacements out(in.placements().size(), replicate);
        return TensorTopology(in.distribution_shape(), std::move(out), in.mesh_coords());
    }

    const uint32_t axis = *cluster_axis;
    if (axis >= mesh_shape.dims()) {
        return std::nullopt;  // the op's validation rejects this cluster_axis right after the hook
    }
    std::string reason;
    auto expanded = uncollapse_placements(in, mesh_shape, &reason);
    if (!expanded.has_value()) {
        return fail(reason, Family::Gather);
    }

    const bool collapsed = is_collapsed(in) && mesh_shape.dims() > 1;
    if (collapsed && single_non_trivial_axis(mesh_shape) == axis) {
        // The collapsed axis is the gathered axis (cluster_axis 1 on a 1xN line): keep the collapsed spelling.
        return TensorTopology(in.distribution_shape(), TopologyPlacements{replicate}, in.mesh_coords());
    }

    // A 1-D mapper shards in row-major device order, so after expanding, gathering the sharded dim along an outer
    // axis would interleave pieces that an inner axis still keeps apart. Only the sharded dim can interleave;
    // gathering some other dim leaves the shards where they are, so either axis is honest then.
    const auto gathered = normalize_tensor_dim(gathered_dim, tensor_rank);
    if (collapsed && require_contiguous_gather && gathered.has_value() &&
        placement_shards_tensor_dim((*expanded)[axis], *gathered, tensor_rank)) {
        for (size_t inner = axis + 1; inner < expanded->size(); ++inner) {
            if (mesh_shape[static_cast<int32_t>(inner)] > 1 &&
                placement_shards_tensor_dim((*expanded)[inner], *gathered, tensor_rank)) {
                return fail(
                    fmt::format(
                        "all_gather of tensor dim {} along mesh axis {} would interleave the shards of a tensor "
                        "distributed with a 1-D mapper over mesh {} (label {}): the ring order is the row-major device "
                        "order, so gather along the innermost sharded mesh axis or create the tensor with an N-D mesh "
                        "mapper",
                        *gathered,
                        axis,
                        mesh_shape,
                        describe(in)),
                    Family::Gather);
            }
        }
    }

    auto out = std::move(*expanded);
    out[axis] = replicate;
    const MeshShape& axis_shape = collapsed ? mesh_shape : in.distribution_shape();
    auto result = finalise(in, axis_shape, std::move(out), tensor_rank, std::nullopt, &reason);
    if (!result.has_value()) {
        return fail(reason, Family::Gather);
    }
    return result;
}

std::optional<TensorTopology> all_gather_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t gathered_dim, bool require_contiguous_gather) {
    return all_gather_output_topology(
        input.tensor_topology(),
        cluster_axis,
        mesh_shape_of(input, "all_gather"),
        rank_of(input),
        gathered_dim,
        require_contiguous_gather);
}

std::optional<TensorTopology> all_reduce_output_topology(
    const TensorTopology& in, std::optional<uint32_t> cluster_axis, const MeshShape& mesh_shape, uint32_t tensor_rank) {
    return all_gather_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, /*gathered_dim=*/0, /*require_contiguous_gather=*/false);
}

std::optional<TensorTopology> all_reduce_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis) {
    return all_reduce_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "all_reduce"), rank_of(input));
}

std::optional<TensorTopology> all_broadcast_output_topology(
    const TensorTopology& in, std::optional<uint32_t> cluster_axis, const MeshShape& mesh_shape, uint32_t tensor_rank) {
    return all_gather_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, /*gathered_dim=*/0, /*require_contiguous_gather=*/false);
}

std::optional<TensorTopology> all_broadcast_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis) {
    return all_broadcast_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "all_broadcast"), rank_of(input));
}

std::optional<TensorTopology> reduce_scatter_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t scatter_dim) {
    const auto normalized_dim = normalize_tensor_dim(scatter_dim, tensor_rank);
    if (!normalized_dim.has_value()) {
        return fail(
            fmt::format("reduce_scatter dim {} is out of range for a rank-{} tensor", scatter_dim, tensor_rank),
            Family::Scatter);
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
            return fail(reason, Family::Scatter);
        }
        // Every device holds a distinct piece of `scatter_dim`, in the label's coordinate order: the collapsed label
        // over those coordinates. A Shard of another dim on a non-trivial axis is still there next to the new piece,
        // and no label can state both.
        const auto& axis_shape = in.distribution_shape();
        for (size_t axis = 0; axis < expanded->size(); ++axis) {
            const auto* other_shard = std::get_if<Shard>(&(*expanded)[axis]);
            if (other_shard != nullptr && axis_shape[static_cast<int32_t>(axis)] > 1 &&
                !placement_shards_tensor_dim((*expanded)[axis], *normalized_dim, tensor_rank)) {
                return fail(
                    fmt::format(
                        "a whole-mesh reduce_scatter of tensor dim {} on a tensor still sharded on dim {} along mesh "
                        "axis {} (label {}) is not expressible",
                        *normalized_dim,
                        other_shard->dim,
                        axis,
                        describe(in)),
                    Family::Scatter);
            }
        }
        return TensorTopology(
            MeshShape(static_cast<uint32_t>(axis_shape.mesh_size())), TopologyPlacements{shard}, in.mesh_coords());
    }

    const uint32_t axis = *cluster_axis;
    if (axis >= mesh_shape.dims()) {
        return std::nullopt;  // the op's validation rejects this cluster_axis right after the hook
    }
    auto expanded = uncollapse_placements(in, mesh_shape, &reason);
    if (!expanded.has_value()) {
        return fail(reason, Family::Scatter);
    }
    if (collapsed && single_non_trivial_axis(mesh_shape) == axis) {
        // The collapsed axis is the scattered axis (cluster_axis 1 on a 1xN line): keep the collapsed spelling.
        return TensorTopology(in.distribution_shape(), TopologyPlacements{shard}, in.mesh_coords());
    }

    // An outer axis already sharding `scatter_dim` composes with the scattered axis into row-major hierarchical
    // sharding only if the scattered axis held the full extent of that dim (Replicate) or its own piece of it
    // (Shard{scatter_dim}); a different Shard there would need two dims sharded on one axis at once.
    const MeshShape& axis_shape = collapsed ? mesh_shape : in.distribution_shape();
    const TopologyPlacement& scattered_axis_held = (*expanded)[axis];
    const bool scattered_axis_composes = std::holds_alternative<Replicate>(scattered_axis_held) ||
                                         placement_shards_tensor_dim(scattered_axis_held, *normalized_dim, tensor_rank);
    for (size_t other = 0; other < expanded->size() && !scattered_axis_composes; ++other) {
        if (other != axis && axis_shape[static_cast<int32_t>(other)] > 1 &&
            placement_shards_tensor_dim((*expanded)[other], *normalized_dim, tensor_rank)) {
            return fail(
                fmt::format(
                    "reduce_scatter of tensor dim {} along mesh axis {}, which already shards dim {}, while mesh axis "
                    "{} shards dim {} too (label {}) is not expressible",
                    *normalized_dim,
                    axis,
                    std::get<Shard>(scattered_axis_held).dim,
                    other,
                    *normalized_dim,
                    describe(in)),
                Family::Scatter);
        }
    }

    auto out = std::move(*expanded);
    out[axis] = shard;
    auto result = finalise(in, axis_shape, std::move(out), tensor_rank, axis, &reason);
    if (!result.has_value()) {
        return fail(reason, Family::Scatter);
    }
    return result;
}

std::optional<TensorTopology> reduce_scatter_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t scatter_dim) {
    return reduce_scatter_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "reduce_scatter"), rank_of(input), scatter_dim);
}

std::optional<TensorTopology> mesh_partition_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim) {
    return reduce_scatter_output_topology(in, cluster_axis, mesh_shape, tensor_rank, out_dim);
}

std::optional<TensorTopology> mesh_partition_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim) {
    return mesh_partition_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "mesh_partition"), rank_of(input), out_dim);
}

std::optional<TensorTopology> all_to_all_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim) {
    return reduce_scatter_output_topology(in, cluster_axis, mesh_shape, tensor_rank, out_dim);
}

std::optional<TensorTopology> all_to_all_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim) {
    return all_to_all_output_topology(
        input.tensor_topology(), cluster_axis, mesh_shape_of(input, "all_to_all"), rank_of(input), out_dim);
}

}  // namespace ttnn::operations::ccl::common
