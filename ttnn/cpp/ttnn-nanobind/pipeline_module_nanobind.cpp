// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_module_nanobind.hpp"

#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "tt-metalium/experimental/blitz_decode_pipeline.hpp"
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>

namespace ttnn::pipeline_module {

void bind_blitz_decode_pipeline(nb::module_& mod) {
    using tt::tt_metal::experimental::blitz::BlitzDecodePipelineStage;

    nb::class_<BlitzDecodePipelineStage>(mod, "BlitzDecodePipelineStage")
        .def_ro("stage_index", &BlitzDecodePipelineStage::stage_index)
        .def_ro("entry_node_coord", &BlitzDecodePipelineStage::entry_node_coord)
        .def_ro("exit_node_coord", &BlitzDecodePipelineStage::exit_node_coord)
        .def("__repr__", [](const BlitzDecodePipelineStage& stage) {
            std::ostringstream repr;
            repr << "BlitzDecodePipelineStage(stage_index=" << stage.stage_index
                 << ", entry_node_coord=" << stage.entry_node_coord << ", exit_node_coord=" << stage.exit_node_coord
                 << ")";
            return repr.str();
        });

    mod.def(
        "generate_blitz_decode_pipeline",
        [](bool initialize_loopback) {
            return tt::tt_metal::experimental::blitz::generate_blitz_decode_pipeline(initialize_loopback);
        },
        nb::arg("initialize_loopback") = true,
        R"doc(
            Generate the Blitz decode pipeline stages.

            Pipeline topology is derived from the control plane's inter-mesh connectivity.

            Args:
                initialize_loopback: When True (default), includes the loopback stage from the
                    last mesh back to mesh 0. When False, generates a linear pipeline with no
                    return path.

            Returns:
                List[BlitzDecodePipelineStage]: Ordered pipeline stages for Blitz decode.
        )doc");
}

void bind_pipeline_builder(nb::module_& mod) {
    using tt::tt_fabric::FabricNodeId;
    using tt::tt_fabric::RoutingDirection;

    nb::enum_<RoutingDirection>(mod, "RoutingDirection", R"(
        Cardinal routing direction used by the fabric control plane.

        Values:
            N: North
            E: East
            S: South
            W: West
            Z: Z-dimension (inter-galaxy)
            C: Centre — source and destination are the same chip
            NONE: No route exists between source and destination
    )")
        .value("N", RoutingDirection::N)
        .value("E", RoutingDirection::E)
        .value("S", RoutingDirection::S)
        .value("W", RoutingDirection::W)
        .value("Z", RoutingDirection::Z)
        .value("C", RoutingDirection::C)
        .value("NONE", RoutingDirection::NONE);

    mod.def(
        "get_forwarding_direction",
        [](const FabricNodeId& src, const FabricNodeId& dst) -> std::optional<RoutingDirection> {
            return tt::tt_fabric::pipeline_get_forwarding_direction(src, dst);
        },
        nb::arg("src"),
        nb::arg("dst"),
        R"(
            Return the direction in which data should be forwarded from *src* to reach *dst*.

            Returns ``None`` if *dst* is not reachable from *src*.  When *src* and *dst* are
            directly connected by a single ethernet cable, this is the direction of that cable.
            For multi-hop paths the first-hop direction is returned; callers that need to verify
            a *direct* (single-hop) connection should follow up with ``get_chip_neighbors``.

            Args:
                src: FabricNodeId of the source chip.
                dst: FabricNodeId of the destination chip.

            Returns:
                RoutingDirection or None.
        )");

    // ------------------------------------------------------------------
    // Graph layout resolution
    // ------------------------------------------------------------------

    nb::class_<tt::tt_fabric::ResolvedEdge>(mod, "ResolvedEdge", R"(
        Physical coordinates discovered for one directed edge.

        Attributes:
            src:        Source node name.
            dst:        Destination node name.
            is_loopback: True for the return edge (last stage → stage 0).
            exit_row:   Row of the exit chip in *src*'s submesh.
            exit_col:   Column of the exit chip in *src*'s submesh.
            entry_row:  Row of the entry chip in *dst*'s submesh.
            entry_col:  Column of the entry chip in *dst*'s submesh.
    )")
        .def_ro("src", &tt::tt_fabric::ResolvedEdge::src)
        .def_ro("dst", &tt::tt_fabric::ResolvedEdge::dst)
        .def_ro("is_loopback", &tt::tt_fabric::ResolvedEdge::is_loopback)
        .def_ro("exit_row", &tt::tt_fabric::ResolvedEdge::exit_row)
        .def_ro("exit_col", &tt::tt_fabric::ResolvedEdge::exit_col)
        .def_ro("entry_row", &tt::tt_fabric::ResolvedEdge::entry_row)
        .def_ro("entry_col", &tt::tt_fabric::ResolvedEdge::entry_col)
        .def("__repr__", [](const tt::tt_fabric::ResolvedEdge& e) {
            return std::string("ResolvedEdge(") + e.src + " -> " + e.dst + (e.is_loopback ? " [loopback]" : "") +
                   " exit=(" + std::to_string(e.exit_row) + "," + std::to_string(e.exit_col) + ")" + " entry=(" +
                   std::to_string(e.entry_row) + "," + std::to_string(e.entry_col) + "))";
        });

    nb::class_<tt::tt_fabric::GraphLayoutResult>(mod, "GraphLayoutResult", R"(
        Result of topology-based graph layout resolution.

        Attributes:
            stage_order:     Node names in topological pipeline stage order (index == stage_idx).
            node_to_submesh: Maps each node name to its submesh index.
            resolved_edges:  One ResolvedEdge per input edge, with discovered physical coords.
            h2d_entry_row:   Row of the H2D entry chip in stage-0's submesh.
            h2d_entry_col:   Column of the H2D entry chip in stage-0's submesh.
            d2h_exit_row:    Row of the D2H exit chip in stage-0's submesh.
            d2h_exit_col:    Column of the D2H exit chip in stage-0's submesh.
    )")
        .def_ro("stage_order", &tt::tt_fabric::GraphLayoutResult::stage_order)
        .def_ro("node_to_submesh", &tt::tt_fabric::GraphLayoutResult::node_to_submesh)
        .def_ro("resolved_edges", &tt::tt_fabric::GraphLayoutResult::resolved_edges)
        .def_ro("h2d_entry_row", &tt::tt_fabric::GraphLayoutResult::h2d_entry_row)
        .def_ro("h2d_entry_col", &tt::tt_fabric::GraphLayoutResult::h2d_entry_col)
        .def_ro("d2h_exit_row", &tt::tt_fabric::GraphLayoutResult::d2h_exit_row)
        .def_ro("d2h_exit_col", &tt::tt_fabric::GraphLayoutResult::d2h_exit_col);

    mod.def(
        "resolve_graph_layout",
        [](const std::vector<tt::tt_fabric::EdgeInputTuple>& edges,
           const std::vector<std::vector<tt::tt_fabric::ChipTuple>>& submesh_chips)
            -> tt::tt_fabric::GraphLayoutResult { return tt::tt_fabric::resolve_graph_layout(edges, submesh_chips); },
        nb::arg("edges"),
        nb::arg("submesh_chips"),
        R"(
            Auto-discover the physical layout of a pipeline graph.

            Uses ``get_forwarding_direction`` and ``get_chip_neighbors`` to find the
            direct ethernet link for each logical edge in the graph, then performs a
            topological sort and backtracking submesh assignment.

            Args:
                edges:         List of (src_name, dst_name, is_loopback) tuples describing
                               the pipeline graph.  Set is_loopback=True for the return
                               edge from the last stage back to stage 0.
                submesh_chips: For each submesh, a list of (mesh_id, chip_id, row, col)
                               tuples (obtained from submesh.get_fabric_node_id()).

            Returns:
                GraphLayoutResult with physical coords for every edge and the H2D/D2H
                chip coords in stage-0's submesh.
        )");

    mod.def(
        "get_chip_neighbors",
        [](const FabricNodeId& src, RoutingDirection direction) -> std::map<uint32_t, std::vector<uint32_t>> {
            return tt::tt_fabric::pipeline_get_chip_neighbors(src, direction);
        },
        nb::arg("src"),
        nb::arg("direction"),
        R"(
            Return the chips directly connected to *src* via an ethernet cable in *direction*.

            Includes both intra-mesh and inter-mesh neighbors.  The result is a ``dict``
            mapping ``mesh_id`` (int) to a list of ``chip_id`` (int) values for chips that
            share a direct ethernet link with *src* in the given direction.

            Use together with ``get_forwarding_direction`` to test direct (single-hop)
            adjacency between two chips:

                direction = ttnn.fabric.get_forwarding_direction(fid_a, fid_b)
                if direction is not None:
                    neighbors = ttnn.fabric.get_chip_neighbors(fid_a, direction)
                    if int(fid_b.mesh_id) in neighbors and fid_b.chip_id in neighbors[int(fid_b.mesh_id)]:
                        # fid_a and fid_b are directly connected

            Args:
                src:       FabricNodeId of the chip to query.
                direction: RoutingDirection to look in.

            Returns:
                dict[int, list[int]] — {mesh_id: [chip_id, ...]} of direct neighbors.
        )");
}

}  // namespace ttnn::pipeline_module
