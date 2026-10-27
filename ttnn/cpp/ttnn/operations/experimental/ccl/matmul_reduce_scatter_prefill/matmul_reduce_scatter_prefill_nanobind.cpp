// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_prefill_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/matmul_reduce_scatter_prefill.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill {

namespace {

const char* schedule_name(detail::ScheduleKind kind) {
    switch (kind) {
        case detail::ScheduleKind::Line: return "line";
        case detail::ScheduleKind::RingUnbalanced: return "ring_unbalanced";
        case detail::ScheduleKind::RingBalanced: return "ring_balanced";
    }
    return "line";
}

const char* half_split_name(detail::HalfSplit split) {
    switch (split) {
        case detail::HalfSplit::K: return "k";
        case detail::HalfSplit::N: return "n";
        case detail::HalfSplit::M: return "m";
    }
    return "k";
}

// The regime / schedule / grid the last call used ({} before any call); a debugging aid, not part of the op's API.
nb::dict last_plan_dict() {
    nb::dict out;
    const auto& recorded = last_plan();
    if (!recorded.has_value()) {
        return out;
    }
    const detail::Plan& plan = recorded->plan;
    out["xblock"] = plan.resident ? "resident" : "stream";
    out["schedule"] = schedule_name(recorded->kind);
    out["kb"] = plan.kb;
    out["depths"] = nb::make_tuple(plan.depths[0], plan.depths[1]);
    out["super_block"] = plan.super_block;
    out["hold_depth"] = plan.depth_hold;
    out["grid"] = nb::make_tuple(plan.grid.grid_m, plan.grid.grid_n);
    out["strip"] = plan.grid.strip == detail::Strip::Row ? "row" : "col";
    out["half_split"] = half_split_name(plan.half_split);
    out["waves"] = plan.waves.n;
    out["chunks"] = nb::make_tuple(plan.n_m, plan.n_n);
    out["chunk_tiles"] = nb::make_tuple(plan.chunk_m, plan.chunk_n);
    out["zone_sets"] = plan.zone_sets;
    out["P"] = plan.P;
    out["footprint_kib"] = plan.footprint / 1024;
    return out;
}

}  // namespace

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill

namespace ttnn::operations::experimental::ccl {

void bind_matmul_reduce_scatter_prefill(nb::module_& mod) {
    ttnn::bind_function<"matmul_reduce_scatter_prefill", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused multi-device matmul + reduce-scatter (SUM) over one mesh axis.

        Every device of a group along ``cluster_axis`` holds a K-slice A[g] (..., M, K) and W[g] (K, N). The op
        computes each device's partial A[g] @ W[g] on a 2D multicast matmul grid and reduce-scatters the partials
        along ``scatter_dim``: the partial of each output block travels to its owner (the shorter way on a ring, along
        the line otherwise), each hop adding its own partial in fp32.

        Args:
            input_tensor (ttnn.Tensor): A, rank 2-4 with leading dims 1, bfloat16 or bfloat8_b, TILE, DRAM interleaved.
            weight (ttnn.Tensor): W (K, N), bfloat16, bfloat8_b or bfloat4_b, TILE, DRAM interleaved.

        Keyword Args:
            cluster_axis (int): the mesh axis of the group (0 or 1).
            scatter_dim (int, optional): -2 (M) or -1 (N). Defaults to -2.
            topology (ttnn.Topology, optional): Linear or Ring. Defaults to Linear.
            num_links (int, optional): fabric links per hop (1 or 2); None uses every usable link.
            compute_kernel_config (ttnn.ComputeConfigDescriptor, optional): math fidelity / approx mode; fp32 DEST
                and full sync are always on. Defaults to HiFi2.
            memory_config (ttnn.MemoryConfig, optional): interleaved output placement. Defaults to DRAM interleaved.
            output_tensor (ttnn.Tensor, optional): preallocated output matching the spec the op would allocate.

        Returns:
            ttnn.Tensor: bfloat16 TILE output, the scattered dim divided by the group size.

        Raises:
            RuntimeError: an unsupported input (dtype, layout, shape, placement, links, sharded output) or a shape
                for which no plan fits L1.
        )doc",
        &ttnn::experimental::matmul_reduce_scatter_prefill,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("scatter_dim") = -2,
        nb::arg("topology") = ttnn::ccl::Topology::Linear,
        nb::arg("num_links") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());

    mod.def(
        "_matmul_reduce_scatter_prefill_last_plan",
        &matmul_reduce_scatter_prefill::last_plan_dict,
        "The plan the last matmul_reduce_scatter_prefill call used (debugging aid).");
}

}  // namespace ttnn::operations::experimental::ccl
