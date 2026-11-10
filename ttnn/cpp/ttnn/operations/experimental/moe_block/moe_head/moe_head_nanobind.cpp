// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_head_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

#include "device/head_plan.hpp"
#include "moe_head.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

namespace moe_head_nanobind_detail {

nb::dict regions_dict(const std::vector<std::pair<std::string, int64_t>>& regions) {
    nb::dict dictionary;
    for (const auto& [name, bytes] : regions) {
        dictionary[nb::str(name.c_str())] = bytes;
    }
    return dictionary;
}

void bind_head_plan(nb::module_& mod) {
    nb::class_<HeadLayout> layout_class(mod, "HeadLayout", "A layout of the MoE head.");
    layout_class.def_ro("engine_rows", &HeadLayout::engine_rows)
        .def_ro("row_tiles_per_core", &HeadLayout::row_tiles_per_core)
        .def_ro("row_chunks", &HeadLayout::row_chunks)
        .def_ro("k_step_tiles", &HeadLayout::k_step_tiles)
        .def_ro("engine_columns", &HeadLayout::engine_columns)
        .def_ro("compact_reduce_scatter_zones", &HeadLayout::compact_reduce_scatter_zones)
        .def_ro("router_iterations", &HeadLayout::router_iterations)
        .def(nb::self == nb::self)
        .def("__repr__", [](const HeadLayout& layout) { return describe_layout(layout); });

    nb::class_<HeadPlanned> planned(mod, "HeadPlanned", "A sized layout and its predicted time (us).");
    planned.def_ro("layout", &HeadPlanned::layout)
        .def_ro("arena", &HeadPlanned::arena)
        .def_prop_ro("sizes", [](const HeadPlanned& head_planned) { return regions_dict(head_planned.sizes); })
        .def_ro("predicted_us", &HeadPlanned::predicted_us)
        .def_prop_ro("parts", [](const HeadPlanned& head_planned) { return regions_dict(head_planned.parts); })
        .def_ro("note", &HeadPlanned::note);

    nb::class_<HeadFit> fit(mod, "HeadFit", R"doc(
        plan_moe_head's answer: ok, planned (best first), reason (the requirement that failed, with its numbers and
        what would fit), unconstrained (the cost model's pick with unlimited L1), note (what L1 forced).)doc");
    fit.def_ro("ok", &HeadFit::ok)
        .def_ro("planned", &HeadFit::planned)
        .def_ro("reason", &HeadFit::reason)
        .def_ro("unconstrained", &HeadFit::unconstrained)
        .def_ro("note", &HeadFit::note)
        .def_ro("row_blocks", &HeadFit::row_blocks)
        .def("__bool__", [](const HeadFit& head_fit) { return head_fit.ok; });
}

}  // namespace moe_head_nanobind_detail

void bind_moe_head(nb::module_& mod) {
    moe_head_nanobind_detail::bind_head_plan(mod);
    mod.def(
        "plan_moe_head",
        [](MoEBlockResources& resources,
           const MoEDtypes& dtypes,
           uint32_t accumulators,
           std::optional<int64_t> l1_free) { return plan_moe_head(resources, dtypes, accumulators, l1_free); },
        nb::arg("resources"),
        nb::arg("dtypes"),
        nb::arg("accumulators"),
        nb::arg("l1_free").none() = nb::none(),
        R"doc(
        The MoE head's whole host plan on a block's resources: the model's requirements, then the head planner on
        the mesh's compute grid and fabric payload. Computed once and kept in the
        resources. l1_free: free L1 a core the head may take (default: the figure measured at the block's first head,
        as moe_head plans with - the plan the op runs).)doc");

    nb::class_<MoEHeadWeights>(
        mod, "MoEHeadWeights", "The shared expert's and the router's weights in the head's layouts.")
        .def(
            nb::init<Tensor, Tensor, Tensor, Tensor>(),
            nb::arg("w"),
            nb::arg("w_router"),
            nb::arg("router_bias"),
            nb::arg("router_index"))
        .def_rw("w", &MoEHeadWeights::w)
        .def_rw("w_router", &MoEHeadWeights::w_router)
        .def_rw("router_bias", &MoEHeadWeights::router_bias)
        .def_rw("router_index", &MoEHeadWeights::router_index);
    nb::class_<MoEHeadOutputs>(mod, "MoEHeadOutputs", "The MoE head's outputs (the resources' tensors).")
        .def_ro("shared_expert_output", &MoEHeadOutputs::shared_expert_output)
        .def_ro("routing_rows", &MoEHeadOutputs::routing_rows)
        .def_ro("dispatch_buffer", &MoEHeadOutputs::dispatch_buffer)
        .def_ro("route_words", &MoEHeadOutputs::route_words)
        .def_ro("counts", &MoEHeadOutputs::counts)
        .def_ro("regions", &MoEHeadOutputs::regions)
        .def_ro("token_index", &MoEHeadOutputs::token_index)
        .def_ro("table", &MoEHeadOutputs::table)
        .def_ro("plan", &MoEHeadOutputs::plan)
        .def_ro("row_blocks", &MoEHeadOutputs::row_blocks);

    ttnn::bind_function<"prepare_moe_head_weights", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        The shared expert's and the router's weights in the MoE head's layouts (its planned engine), once a layer.

        Args:
            gate, up: the shared expert's [Is, H] (host, float32 or bfloat16).
            down: [H, Is].
            latent_down: a latent model's [Hr, H] (None without).
            router_weight: [E, H] (E: the router's candidates, num_experts x fold); router_bias: [E].

        Keyword args:
            resources, dtypes, accumulators, l1_free: as moe_head.

        Returns:
            MoEHeadWeights (device tensors).
        )doc",
        &prepare_moe_head_weights,
        nb::arg("gate"),
        nb::arg("up"),
        nb::arg("down"),
        nb::arg("latent_down"),
        nb::arg("router_weight"),
        nb::arg("router_bias"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("dtypes"),
        nb::arg("accumulators"),
        nb::arg("l1_free") = nb::none());

    ttnn::bind_function<"moe_head", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        The MoE block's head as one program a chip: the x all-gather over TP into the shared expert's matmul cores,
        its down + reduce-scatter, the router (logits, top-K, the owners' joint routing rows), the routing exchange and
        the combine planner (the combine's plan tensors), the dispatch prep (this chip's owned rows into the dispatch
        buffer, their route words) and, with a latent, the latent down + all-to-all.

        Args:
            x: this chip's [1, 1, rows, H / TP] (rows: a mesh row's tokens), bfloat16 TILE, DRAM.
            weights: prepare_moe_head_weights'.

        Keyword args:
            resources: the block's MoEBlockResources.
            dtypes: MoEDtypes (the shared expert's weights / fidelity).
            accumulators: the combine's accumulator count (the routing planner's instances at most).
            l1_free: the free L1 a core the head may take (default: measured at the block's first head).
            arena: an L1 tensor over the head's arena cores the head runs in (default one the resources keep, which
                holds that L1 for good: a block shares one arena between its ops).
            actual_start, actual_end: a padded chunk - main's chunked-prefill metadata scalars as they are (the
                chunk's first real token's absolute position, one past its last), one-element uint32 tensors in DRAM,
                replicated, read on device every call (a trace replays with their values of the moment). The chunk is
                in main's KV-pad-aware rotated layout over the SP rows (moe_padding_config): an SP row's real rows are
                a prefix, TP chip p owns the row's tokens p S ... Padding tokens get no routes: not dispatched, not
                computed by the routed experts, not combined (their combined rows are zero). None: every row is real.
            split: this call's piece of the chunk when MoEBlock runs it as MoEBlockConfig.chunk_splits pieces (x:
                the piece's rows of every chip; a padded chunk's real rows offset by it).
            metadata: a padded forward as the forward's metadata words (common/prefill/prefill_chunk.py: at
                MoEBlockConfig.forward_chunks 1 (slot, start, end), above {num_chunks, 0} then forward_chunks
                (slot, start, end) records, unused ones (INVALID, 0, 0)), one uint32 DRAM tensor of one page,
                replicated, read on device every call - in place of actual_start / actual_end; a packed forward
                (forward_chunks > 1) takes it. Each SP row holds every chunk's rows back to back, each chunk in main's
                rotated layout; padding rows are not routed.

        The head runs the call's rows as the row blocks its planner picked from its L1 fit (1 wherever the rows fit at
        once): one program a chip whose kernels block over the blocks.

        Returns:
            MoEHeadOutputs: shared_expert_output, routing_rows, dispatch_buffer, route_words and the combine
            plan (counts, regions, token_index, table, plan) - the resources' tensors, rewritten each call - and
            row_blocks (the tail's moe_tail(row_blocks=)): the owner-order rows are in that block layout.

        The plan, the host model and the persistent words are made on the first call and kept in the resources; a
        repeated call rewrites only the arguments of the tensors whose address moved.
        )doc",
        &moe_head,
        nb::arg("x"),
        nb::arg("weights"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("dtypes"),
        nb::arg("accumulators"),
        nb::arg("l1_free") = nb::none(),
        nb::arg("arena") = nb::none(),
        nb::arg("actual_start") = nb::none(),
        nb::arg("actual_end") = nb::none(),
        nb::arg("split") = 0,
        nb::arg("metadata") = nb::none());
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
