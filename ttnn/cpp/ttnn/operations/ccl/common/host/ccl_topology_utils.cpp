// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/config.hpp"
#include "ttnn/device_operation_detail.hpp"

namespace ttnn::operations::ccl::common {

using tt::tt_metal::TensorTopology;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshShape;
using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;

namespace {

std::string describe(const TensorTopology& topology) {
    std::ostringstream ss;
    ss << topology;
    return ss.str();
}

// The collapsed spelling: a 1-D label with a single placement for the whole of a multi-axis mesh. On a 1-D mesh the
// same form is the N-D label (one placement per axis), so it is not collapsed there.
bool is_collapsed(const TensorTopology& in, const MeshShape& mesh_shape) {
    return in.distribution_shape().dims() == 1 && in.placements().size() == 1 && mesh_shape.dims() > 1;
}

// The one axis of size > 1 of a mesh that is a line (1xN, Nx1, ...); nullopt when the mesh has none or several.
std::optional<size_t> single_non_trivial_axis(const MeshShape& mesh_shape) {
    std::optional<size_t> found;
    for (size_t axis = 0; axis < mesh_shape.dims(); ++axis) {
        if (mesh_shape[axis] > 1) {
            if (found.has_value()) {
                return std::nullopt;
            }
            found = axis;
        }
    }
    return found;
}

bool strict_ccl_topology() { return ttnn::CONFIG.get<"strict_ccl_topology">(); }

// Which family refused: the fallback (the input's label) is safe for a gather family -- it never adds a Replicate the
// output lacks -- but can over-claim Replicate for the scatter family, so the latter logs at error level.
using Family = RefusalFamily;

// SilenceIntermediateRefusals objects alive on this thread (see the header): while positive, refusals are silent.
thread_local int silence_intermediate_refusals_depth = 0;

// Each distinct message is logged once per process: a model that trips the same refusal in every layer would
// otherwise flood the log. The set is bounded; once it holds kMaxRememberedMessages distinct messages it is cleared,
// so a process that keeps producing new refusals repeats an old one at most once per cycle instead of growing without
// bound.
bool first_time(const std::string& message) {
    constexpr size_t kMaxRememberedMessages = 256;
    static std::mutex mutex;
    static std::unordered_set<std::string> seen;
    const std::lock_guard<std::mutex> lock(mutex);
    if (seen.size() >= kMaxRememberedMessages) {
        seen.clear();
    }
    return seen.insert(message).second;
}

}  // namespace

SilenceIntermediateRefusals::SilenceIntermediateRefusals() { ++silence_intermediate_refusals_depth; }

SilenceIntermediateRefusals::~SilenceIntermediateRefusals() { --silence_intermediate_refusals_depth; }

// No honest label: nullopt without a word inside a SilenceIntermediateRefusals scope (the label is an intermediate's
// that the caller replaces), TT_FATAL under strict mode, otherwise a log line and nullopt so the device op returns the
// input's label and a host relabel site skips the relabel.
std::optional<TensorTopology> refuse(const std::string& what, RefusalFamily family) {
    if (silence_intermediate_refusals_depth > 0) {
        return std::nullopt;
    }
    if (strict_ccl_topology()) {
        TT_THROW("{}", what);
    }
    if (family == Family::Scatter) {
        const std::string message = fmt::format(
            "{}; the output keeps the input's TensorTopology, which may claim Replicate on a mesh axis whose devices "
            "now hold different pieces and so lose data when serialised. Strict mode (ttnn.CONFIG.strict_ccl_topology) "
            "is the only safe mode for reduce_scatter / mesh_partition / all_to_all labels.",
            what);
        if (first_time(message)) {
            log_error(tt::LogOp, "{}", message);
        }
    } else {
        const std::string message = fmt::format(
            "{}; the output keeps the input's TensorTopology. Set ttnn.CONFIG.strict_ccl_topology to make this an "
            "error.",
            what);
        if (first_time(message)) {
            log_warning(tt::LogOp, "{}", message);
        }
    }
    return std::nullopt;
}

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
        // One placement per mesh axis only means something when axis i of the label is axis i of the mesh: the
        // distribution shape fits the mesh per axis and the coordinates walk a distribution-shaped block of it in
        // row-major order. A mapper whose mesh_shape_override does not fit distributes in row-major order over all
        // devices instead (DistributionMode::ROW_MAJOR), and per physical axis that layout has no placement.
        //
        // The coordinates can only be consulted when they are the distribution's. A device tensor's label carries
        // its storage's coordinates -- every device of the mesh, since an op output's buffer spans the mesh whatever
        // block its input occupied (update_output_tensor_topologies keeps each output's own coordinates) -- so for a
        // sub-range distribution such as the (1, C) row that from_torch(ShardTensor2dMesh(mesh_shape=(1, C))) builds
        // on a 2-D mesh the coordinate list says nothing about where the block lies. The shape alone decides then:
        // a shape that fits the mesh per axis was mapped as a block (DistributionMode::SUBMESH, with or without an
        // offset), so its axes are the mesh's; one that does not fit was laid out row-major over the mesh.
        bool fits = true;
        for (size_t axis = 0; axis < mesh_dims; ++axis) {
            fits &= distribution_shape[axis] <= mesh_shape[axis];
        }
        const bool coordinates_are_the_distributions = in.mesh_coords().size() == distribution_shape.mesh_size();
        if (!fits || (coordinates_are_the_distributions &&
                      !has_row_major_mesh_coordinates(in.mesh_coords(), distribution_shape))) {
            set_reason(fmt::format(
                "tensor topology {} does not align with the axes of mesh {}: its coordinates are not a {}-shaped "
                "block of the mesh in row-major order (a mapper whose mesh_shape_override does not fit the mesh "
                "distributes in row-major order over all devices); create the tensor with a mapper whose shape fits "
                "the mesh",
                describe(in),
                mesh_shape,
                distribution_shape));
            return std::nullopt;
        }
        return placements;
    }
    if (!is_collapsed(in, mesh_shape)) {
        set_reason(fmt::format(
            "tensor topology {} is neither a collapsed 1-D label nor one placement per axis of mesh {}",
            describe(in),
            mesh_shape));
        return std::nullopt;
    }
    if (distribution_shape.mesh_size() != mesh_shape.mesh_size() ||
        !has_row_major_mesh_coordinates(in.mesh_coords(), mesh_shape)) {
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
            if (mesh_shape[axis] > 1) {
                out[axis] = *shard;
            }
        }
    }
    return out;
}

// Shared by both families: all_gather_output_topology below is its first caller.
namespace {

// Coverage. The devices an op reaches are the mesh's, not the label's:
// a cluster-axis op spans the mesh's full extent along that axis and a whole-mesh op every device, so a label that
// covers fewer devices along the axis addressed -- a (1, 4) row block on a 2x4 along axis 0, or over the whole
// mesh -- is refused: the devices the label does not describe take part (a sum with them, a gather of them, a
// partition that numbers them) and no label over the block can say what they hold. A block is gathered or
// scattered within itself only along an axis it spans fully (the row along axis 1). The reason, or nullopt when the
// label covers what the op reaches. A collapsed label has no axis to compare; uncollapse_placements already
// requires it to cover the whole mesh, so only N-D labels are judged per axis here.
std::optional<std::string> coverage_gap(
    const TensorTopology& in, const MeshShape& mesh_shape, std::optional<uint32_t> axis, std::string_view what) {
    const auto& label_shape = in.distribution_shape();
    if (!axis.has_value()) {
        if (label_shape.mesh_size() != mesh_shape.mesh_size()) {
            return fmt::format(
                "a whole-mesh {} spans the {} devices of mesh {} but the label {} covers {}: the devices the label "
                "does "
                "not describe take part, and no label over the block can say what they hold",
                what,
                mesh_shape.mesh_size(),
                mesh_shape,
                describe(in),
                label_shape.mesh_size());
        }
        return std::nullopt;
    }
    if (label_shape.dims() != mesh_shape.dims()) {
        return std::nullopt;
    }
    if (label_shape[*axis] != mesh_shape[*axis]) {
        return fmt::format(
            "{} along mesh axis {} spans {} devices but the label {} covers {} along that axis: the groups reach "
            "devices the label does not describe",
            what,
            *axis,
            mesh_shape[*axis],
            describe(in),
            label_shape[*axis]);
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
// already hold (device (r, c) holds chunk r * C + c). Scattering along an outer axis lays the pieces out column-major
// (device (r, c) holds chunk c * R + r), which no label describes.
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
        if (shard != nullptr && axis_shape[axis] == 1 && shards_same_dim_as(axis, *shard)) {
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
        if (axis_shape[axis] > 1) {
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

}  // namespace

std::optional<TensorTopology> all_gather_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t gathered_dim,
    bool require_contiguous_gather) {  // refuse gathers that would interleave a dim sharded on more than one axis
    const TopologyPlacement replicate = Replicate{};
    if (!cluster_axis.has_value()) {
        // Whole-mesh gather: every device ends up with every piece, whatever the label's rank. The ring order is the
        // label's own coordinate order, so the distribution shape and coordinates are kept -- provided the label
        // covers every device the ring reaches.
        if (const auto gap = coverage_gap(in, mesh_shape, std::nullopt, "collective"); gap.has_value()) {
            return refuse(*gap, Family::Gather);
        }
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
        return refuse(reason, Family::Gather);
    }

    if (const auto gap = coverage_gap(in, mesh_shape, axis, "a collective"); gap.has_value()) {
        return refuse(*gap, Family::Gather);
    }

    const bool collapsed = is_collapsed(in, mesh_shape);
    if (collapsed && single_non_trivial_axis(mesh_shape) == axis) {
        // The collapsed axis is the gathered axis (cluster_axis 1 on a 1xN line): keep the collapsed spelling.
        return TensorTopology(in.distribution_shape(), TopologyPlacements{replicate}, in.mesh_coords());
    }

    // A dim sharded on more than one mesh axis is read as row-major hierarchical sharding (what a 1-D mapper
    // produces, and the only reading `finalise` gives an N-D label that spells the same dim twice), so gathering it
    // along an outer axis would interleave pieces that an inner axis still keeps apart. Only the sharded dim can
    // interleave; gathering some other dim leaves the shards where they are, so either axis is honest then.
    const MeshShape& axis_shape = collapsed ? mesh_shape : in.distribution_shape();
    const auto gathered = normalize_tensor_dim(gathered_dim, tensor_rank);
    if (require_contiguous_gather && gathered.has_value() &&
        placement_shards_tensor_dim((*expanded)[axis], *gathered, tensor_rank)) {
        for (size_t inner = axis + 1; inner < expanded->size(); ++inner) {
            if (axis_shape[inner] > 1 && placement_shards_tensor_dim((*expanded)[inner], *gathered, tensor_rank)) {
                return refuse(
                    fmt::format(
                        "all_gather of tensor dim {} along mesh axis {} would interleave the shards of a tensor that "
                        "shards dim {} on more than one axis of mesh {} (label {}): the ring order is the row-major "
                        "device order, so gather along the innermost mesh axis sharding that dim, or create the tensor "
                        "with an N-D mesh mapper that shards each dim once",
                        *gathered,
                        axis,
                        *gathered,
                        mesh_shape,
                        describe(in)),
                    Family::Gather);
            }
        }
    }

    auto out = std::move(*expanded);
    out[axis] = replicate;
    auto result = finalise(in, axis_shape, std::move(out), tensor_rank, std::nullopt, &reason);
    if (!result.has_value()) {
        return refuse(reason, Family::Gather);
    }
    return result;
}

std::optional<TensorTopology> all_gather_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t gathered_dim, bool require_contiguous_gather) {
    return all_gather_output_topology(
        input.tensor_topology(),
        cluster_axis,
        input.device()->shape(),
        input.logical_shape().rank(),
        gathered_dim,
        require_contiguous_gather);
}

// all_reduce reuses the all_gather rule on purpose. A label records where bytes live, not the arithmetic that
// produced them: an all_gather along `cluster_axis` leaves every device on that axis holding the same bytes (the
// concatenated pieces) and so does an all_reduce (the sum), while neither touches the other axes. Both therefore read
// as Replicate on `cluster_axis` with every other placement kept (whole mesh: Replicate everywhere), which is what
// all_gather_output_topology computes, including the collapsed-label reading and the coverage check. The contiguity
// requirement is off because it guards the *order* in which a gather concatenates a dim sharded on more than one mesh
// axis (gathering along an outer axis would interleave pieces an inner axis keeps apart); a sum has no concatenation
// order, so a dim sharded twice reduces along either axis. `gathered_dim` is consulted only by that check, so the
// value passed here is irrelevant.
std::optional<TensorTopology> all_reduce_output_topology(
    const TensorTopology& in, std::optional<uint32_t> cluster_axis, const MeshShape& mesh_shape, uint32_t tensor_rank) {
    return all_gather_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, /*gathered_dim=*/0, /*require_contiguous_gather=*/false);
}

std::optional<TensorTopology> all_reduce_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis) {
    return all_reduce_output_topology(
        input.tensor_topology(), cluster_axis, input.device()->shape(), input.logical_shape().rank());
}

// all_broadcast reuses the all_gather rule on purpose, once per output. The op returns one tensor per device on
// `cluster_axis`; output i holds device i's input on every device of that axis, so each output has the same bytes
// on every device there (Replicate on `cluster_axis`, every other placement kept; whole mesh: Replicate everywhere),
// which is the all_gather label; the hook applies it to all `ring_size` outputs. Nothing is concatenated, so the
// contiguity check (the order a gather concatenates a dim sharded on more than one mesh axis) does not apply, and
// `gathered_dim`, consulted only by that check, is irrelevant.
std::optional<TensorTopology> all_broadcast_output_topology(
    const TensorTopology& in, std::optional<uint32_t> cluster_axis, const MeshShape& mesh_shape, uint32_t tensor_rank) {
    return all_gather_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, /*gathered_dim=*/0, /*require_contiguous_gather=*/false);
}

std::optional<TensorTopology> all_broadcast_output_topology(const Tensor& input, std::optional<uint32_t> cluster_axis) {
    return all_broadcast_output_topology(
        input.tensor_topology(), cluster_axis, input.device()->shape(), input.logical_shape().rank());
}

// The scatter family's shared rule, next to its first caller.
namespace {

// Whether a scatter sums over the devices it spans (reduce_scatter: everything those devices held is consumed) or
// leaves each device its own piece (mesh_partition, all_to_all: a Shard of another dim stays next to the new piece).
enum class ScatterKind { Reduction, Partition };

// The reduce_scatter / mesh_partition / all_to_all label; see the header for the rules and where the two kinds differ.
std::optional<TensorTopology> scatter_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t scatter_dim,
    ScatterKind kind,
    std::string_view op) {
    const auto normalized_dim = normalize_tensor_dim(scatter_dim, tensor_rank);
    if (!normalized_dim.has_value()) {
        return refuse(
            fmt::format("{} dim {} is out of range for a rank-{} tensor", op, scatter_dim, tensor_rank),
            Family::Scatter);
    }
    // Only normalised dims are ever written into a placement the helper creates.
    const TopologyPlacement shard = Shard{static_cast<int>(*normalized_dim)};
    const bool collapsed = is_collapsed(in, mesh_shape);
    std::string reason;

    if (!cluster_axis.has_value()) {
        // A whole-mesh scatter gives ring rank i piece i of `scatter_dim`, the ring being the label's coordinate
        // order (what the pure overload has to assume; the Tensor overloads respell over the storage), so the result
        // is the collapsed {N}, [Shard{scatter_dim}] over those coordinates whatever the label's rank. The label's
        // own axes are read as they are: no axis is addressed, so the per-axis alignment uncollapse_placements
        // insists on is not needed, and a row-major reshape of the whole mesh ({1,8} over a 2x4, a 2-D mapper whose
        // mesh_shape_override did not fit) is as exact as the collapsed label that enumerates the same devices.
        const auto& label_shape = in.distribution_shape();
        const auto& placements = in.placements();
        if (placements.size() != label_shape.dims()) {
            return refuse(
                fmt::format(
                    "tensor topology {} has {} placements for a {}-D distribution shape",
                    describe(in),
                    placements.size(),
                    label_shape.dims()),
                Family::Scatter);
        }
        if (const auto gap = coverage_gap(in, mesh_shape, std::nullopt, op); gap.has_value()) {
            return refuse(*gap, Family::Scatter);
        }
        // Every device holds a distinct piece of `scatter_dim`, in the label's coordinate order: the collapsed label
        // over those coordinates. A reduction summed every device's tensor into that piece, so nothing stays sharded
        // on another dim; a partition keeps each device's other-dim piece next to the new one, and no label can state
        // both.
        //
        // The collapsed label has one placement for the whole mesh, so it can only say "scatter_dim is split N ways
        // in ring order, every other dim is whole". A partition of a tensor still sharded on another dim along a
        // non-trivial axis produces outputs that label cannot describe. 2x4 mesh, input [Shard{2}, Replicate],
        // whole-mesh mesh_partition of dim 3 (ring rank i = r * 4 + c keeps piece i of 8 of its own tensor):
        //
        //         c=0          c=1          c=2          c=3
        //   r=0   d2 half 0    d2 half 0    d2 half 0    d2 half 0
        //         d3 piece 0   d3 piece 1   d3 piece 2   d3 piece 3
        //   r=1   d2 half 1    d2 half 1    d2 half 1    d2 half 1
        //         d3 piece 4   d3 piece 5   d3 piece 6   d3 piece 7
        //
        //   {8},[Shard{3}]        says dim 2 is whole on every device: false, the rows hold different halves
        //   [Shard{2}, Shard{3}]  says device (r, c) holds dim-3 piece c of 4, the same on both rows: false
        //
        // No label describes it, so it is refused. A Shard of scatter_dim itself passes: each device keeps a
        // sub-piece of the piece it already held, and the outputs are N distinct pieces of that dim with every other
        // dim whole, which is what the collapsed label states (a label says how the outputs assemble, not that they
        // tile the input). A Shard on a size-1 axis holds the whole extent and passes too. A collapsed
        // {N}, [Shard{other}] is the same case with one placement. A reduction never reaches this check: the sum is
        // one tensor whose other dims are whole on every device. With a cluster_axis the output stays N-D and the
        // other dim's Shard keeps its own axis, so only the whole-mesh collapse loses it.
        if (kind == ScatterKind::Partition) {
            for (size_t axis = 0; axis < placements.size(); ++axis) {
                const auto* other_shard = std::get_if<Shard>(&placements[axis]);
                if (other_shard != nullptr && label_shape[axis] > 1 &&
                    !placement_shards_tensor_dim(placements[axis], *normalized_dim, tensor_rank)) {
                    return refuse(
                        fmt::format(
                            "a whole-mesh {} of tensor dim {} on a tensor still sharded on dim {} along axis {} of its "
                            "label {} is not expressible",
                            op,
                            *normalized_dim,
                            other_shard->dim,
                            axis,
                            describe(in)),
                        Family::Scatter);
                }
            }
        }
        return TensorTopology(
            MeshShape(static_cast<uint32_t>(label_shape.mesh_size())), TopologyPlacements{shard}, in.mesh_coords());
    }

    const uint32_t axis = *cluster_axis;
    if (axis >= mesh_shape.dims()) {
        return std::nullopt;  // the op's validation rejects this cluster_axis right after the hook
    }
    auto expanded = uncollapse_placements(in, mesh_shape, &reason);
    if (!expanded.has_value()) {
        return refuse(reason, Family::Scatter);
    }
    if (const auto gap = coverage_gap(in, mesh_shape, axis, op); gap.has_value()) {
        return refuse(*gap, Family::Scatter);
    }
    if (collapsed && single_non_trivial_axis(mesh_shape) == axis) {
        // The collapsed axis is the scattered axis (cluster_axis 1 on a 1xN line): keep the collapsed spelling.
        return TensorTopology(in.distribution_shape(), TopologyPlacements{shard}, in.mesh_coords());
    }

    // A reduction consumes whatever the scattered axis held. A partition does not: a Shard of another dim there
    // stays next to each device's new piece (device c holds chunk c of that dim and piece c of `scatter_dim`), which
    // is two dims sharded on one axis at once -- no label says that, so it is refused whatever the other axes hold.
    // A stale Shard dim (#52331) is a Shard of another dim here too.
    const MeshShape& axis_shape = collapsed ? mesh_shape : in.distribution_shape();
    const TopologyPlacement& scattered_axis_held = (*expanded)[axis];
    const bool scattered_axis_composes = kind == ScatterKind::Reduction ||
                                         std::holds_alternative<Replicate>(scattered_axis_held) ||
                                         placement_shards_tensor_dim(scattered_axis_held, *normalized_dim, tensor_rank);
    if (!scattered_axis_composes) {
        return refuse(
            fmt::format(
                "{} of tensor dim {} along mesh axis {}, which already shards dim {} (label {}), is not expressible: "
                "a partition keeps each device's piece of dim {} next to its new piece of dim {}",
                op,
                *normalized_dim,
                axis,
                std::get<Shard>(scattered_axis_held).dim,
                describe(in),
                std::get<Shard>(scattered_axis_held).dim,
                *normalized_dim),
            Family::Scatter);
    }

    auto out = std::move(*expanded);
    out[axis] = shard;
    auto result = finalise(in, axis_shape, std::move(out), tensor_rank, axis, &reason);
    if (!result.has_value()) {
        return refuse(reason, Family::Scatter);
    }
    return result;
}

}  // namespace

std::optional<TensorTopology> reduce_scatter_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t scatter_dim) {
    return scatter_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, scatter_dim, ScatterKind::Reduction, "reduce_scatter");
}

// A whole-mesh ring whose rank is the index into a tensor's device-storage coordinates
// (get_linearized_index_from_physical_coord without a cluster_axis: reduce_scatter, the fused matmul + reduce_scatter
// ops and all_to_all_async_generic) puts piece i on storage coordinate i, not on the i-th coordinate of the label the
// pure overloads have to assume is the ring order: spell `label` over the storage coordinates so piece i sits on ring
// rank i whatever coordinates the input label carried. Only for a tensor stored on every device of the mesh: on a
// multi-host mesh the storage holds this host's shard while the label is global, and a single-host sub-mesh tensor's
// label already carries its own device list, so both keep the label's coordinates. With a cluster_axis the label is
// returned as is. mesh_partition and all_to_all_async do not rank by storage order (row * num_cols + col over the mesh
// view, and MeshDeviceView::get_ring_devices), so their labels do not come through here.
std::optional<TensorTopology> over_storage_ring_order(
    std::optional<TensorTopology> label,
    const Tensor& ring_tensor,
    std::optional<uint32_t> cluster_axis,
    std::string_view op) {
    // Returned as is: no label (the pure helper already refused); a cluster_axis (the ring rank is then the coordinate
    // value along that axis, not a position in the storage list, so the label's coordinates are already right); a
    // tensor not stored on every device (on a multi-host mesh this host's storage lists only its own devices while the
    // label is global, so respelling over it would shrink the label; a single-host sub-mesh label already lists its
    // own devices, so there is nothing to respell).
    if (!label.has_value() || cluster_axis.has_value() || !ring_tensor.device_storage().is_uniform_storage()) {
        return label;
    }
    const auto storage_coords = ring_tensor.device_storage().get_coords();
    if (storage_coords.size() != label->mesh_coords().size()) {
        return refuse(
            fmt::format(
                "whole-mesh {} label {} has {} coordinates for a tensor stored on {} devices",
                op,
                describe(*label),
                label->mesh_coords().size(),
                storage_coords.size()),
            Family::Scatter);
    }
    return TensorTopology(
        label->distribution_shape(),
        label->placements(),
        std::vector<MeshCoordinate>(storage_coords.begin(), storage_coords.end()));
}

std::optional<TensorTopology> reduce_scatter_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t scatter_dim) {
    return over_storage_ring_order(
        reduce_scatter_output_topology(
            input.tensor_topology(), cluster_axis, input.device()->shape(), input.logical_shape().rank(), scatter_dim),
        input,
        cluster_axis,
        "reduce_scatter");
}

std::optional<TensorTopology> mesh_partition_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim) {
    return scatter_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, out_dim, ScatterKind::Partition, "mesh_partition");
}

std::optional<TensorTopology> mesh_partition_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim) {
    // mesh_partition's program factory numbers a whole-mesh partition by the row-major linearised index of each
    // device's own coordinate (get_linearized_index over the mesh view), which for a tensor stored on every device is
    // its storage order: the same respelling as reduce_scatter's. With a cluster_axis the rank is the coordinate
    // value along that axis and the label's coordinates, a block uncollapse_placements has checked, are kept.
    return over_storage_ring_order(
        mesh_partition_output_topology(
            input.tensor_topology(), cluster_axis, input.device()->shape(), input.logical_shape().rank(), out_dim),
        input,
        cluster_axis,
        "mesh_partition");
}

std::optional<TensorTopology> all_to_all_output_topology(
    const TensorTopology& in,
    std::optional<uint32_t> cluster_axis,
    const MeshShape& mesh_shape,
    uint32_t tensor_rank,
    int32_t out_dim) {
    return scatter_output_topology(
        in, cluster_axis, mesh_shape, tensor_rank, out_dim, ScatterKind::Partition, "all_to_all");
}

std::optional<TensorTopology> all_to_all_output_topology(
    const Tensor& input, std::optional<uint32_t> cluster_axis, int32_t out_dim) {
    return all_to_all_output_topology(
        input.tensor_topology(), cluster_axis, input.device()->shape(), input.logical_shape().rank(), out_dim);
}

tt::tt_metal::TensorTopology union_topology(
    const std::vector<std::reference_wrapper<const tt::tt_metal::TensorTopology>>& operands,
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords) {
    TT_FATAL(!operands.empty(), "Cannot compute the union label of no operands");
    const tt::tt_metal::TensorTopology& primary = operands.front().get();
    const auto& primary_placements = primary.placements();
    if (std::all_of(primary_placements.begin(), primary_placements.end(), [](const auto& placement) {
            return std::holds_alternative<Replicate>(placement);
        })) {
        auto [placements, shape] = ttnn::device_operation::detail::compute_output_placements_and_shape(operands);
        return tt::tt_metal::TensorTopology(std::move(shape), std::move(placements), std::move(mesh_coords));
    }
    // The first operand's rank is the result's; a sharded operand of another rank would have one side of the union
    // dropped (the union keeps the highest rank), so it is left out. Fully replicated operands never decide the rank
    // while something is sharded and are kept as they are.
    const size_t rank = primary.distribution_shape().dims();
    std::vector<std::reference_wrapper<const tt::tt_metal::TensorTopology>> same_rank;
    same_rank.reserve(operands.size());
    for (const auto& operand : operands) {
        const auto& placements = operand.get().placements();
        if (operand.get().distribution_shape().dims() == rank ||
            std::all_of(placements.begin(), placements.end(), [](const auto& placement) {
                return std::holds_alternative<Replicate>(placement);
            })) {
            same_rank.push_back(operand);
            continue;
        }
        // The labels are formatted only when debug logging is on: this runs on every launch of the fused ops.
        if (tt::LoggerRegistry::instance().get(tt::LogOp)->should_log(spdlog::level::debug)) {
            log_debug(
                tt::LogOp,
                "union_topology leaves out the operand label {}: it is sharded with distribution rank {}, and the "
                "result keeps the first operand's rank {} ({})",
                describe(operand.get()),
                operand.get().distribution_shape().dims(),
                rank,
                describe(primary));
        }
    }
    auto [placements, shape] = ttnn::device_operation::detail::compute_output_placements_and_shape(same_rank);
    return tt::tt_metal::TensorTopology(std::move(shape), std::move(placements), std::move(mesh_coords));
}

}  // namespace ttnn::operations::ccl::common
