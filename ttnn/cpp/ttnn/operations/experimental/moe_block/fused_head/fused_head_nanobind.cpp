// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_head_nanobind.hpp"

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
#include "fused_head.hpp"

namespace ttnn::operations::experimental::moe_block::fused_head {

namespace fused_head_nanobind_detail {

nb::dict regions_dict(const std::vector<std::pair<std::string, int64_t>>& r) {
    nb::dict d;
    for (const auto& [k, v] : r) {
        d[nb::str(k.c_str())] = v;
    }
    return d;
}

void bind_head_plan(nb::module_& mod) {
    nb::class_<HeadLayout> lay(mod, "HeadLayout", "A layout of the fused head (head_plan.HeadLayout).");
    lay.def_ro("GY", &HeadLayout::GY)
        .def_ro("MT", &HeadLayout::MT)
        .def_ro("NCH", &HeadLayout::NCH)
        .def_ro("KB", &HeadLayout::KB)
        .def_ro("GX", &HeadLayout::GX)
        .def_ro("compact", &HeadLayout::compact)
        .def_ro("xdram", &HeadLayout::xdram)
        .def_ro("NIT", &HeadLayout::NIT)
        .def(nb::self == nb::self)
        .def("__repr__", [](const HeadLayout& l) { return describe_layout(l); });

    nb::class_<HeadPlanned> planned(mod, "HeadPlanned", "A sized layout and its predicted us (head_plan.Planned).");
    planned.def_ro("lay", &HeadPlanned::lay)
        .def_ro("arena", &HeadPlanned::arena)
        .def_prop_ro("sizes", [](const HeadPlanned& p) { return regions_dict(p.sizes); })
        .def_ro("t", &HeadPlanned::t)
        .def_prop_ro("parts", [](const HeadPlanned& p) { return regions_dict(p.parts); })
        .def_ro("note", &HeadPlanned::note);

    nb::class_<HeadFit> fit(mod, "HeadFit", R"doc(
        plan_fused_head's answer: ok, planned (best first), why (the requirement that failed, with its numbers and what
        would fit), free (the cost model's pick with unlimited L1), note (what L1 forced, or the DRAM fallback's cost).)doc");
    fit.def_ro("ok", &HeadFit::ok)
        .def_ro("planned", &HeadFit::planned)
        .def_ro("why", &HeadFit::why)
        .def_ro("free", &HeadFit::free)
        .def_ro("note", &HeadFit::note)
        .def("__bool__", [](const HeadFit& f) { return f.ok; });
}

}  // namespace fused_head_nanobind_detail

void bind_fused_head(nb::module_& mod) {
    fused_head_nanobind_detail::bind_head_plan(mod);
    mod.def(
        "plan_fused_head",
        [](MoEBlockResources& resources,
           const MoEDtypes& dtypes,
           uint32_t accumulators,
           std::optional<int64_t> l1_free) { return plan_fused_head(resources, dtypes, accumulators, l1_free); },
        nb::arg("resources"),
        nb::arg("dtypes"),
        nb::arg("accumulators"),
        nb::arg("l1_free").none() = nb::none(),
        R"doc(
        The fused head's whole host plan on a block's resources: the model's requirements, then the head planner on
        the mesh's compute grid, fabric payload and the config's DRAM fallback. Computed once and kept in the
        resources. l1_free: free L1 a core the head may take (default: the figure measured at the block's first head,
        as fused_head plans with - the plan the op runs).)doc");

    nb::class_<FusedHeadWeights>(
        mod, "FusedHeadWeights", "The shared expert's and the router's weights in the head's layouts.")
        .def(
            nb::init<Tensor, Tensor, Tensor, Tensor>(),
            nb::arg("w"),
            nb::arg("w_router"),
            nb::arg("router_bias"),
            nb::arg("router_index"))
        .def_rw("w", &FusedHeadWeights::w)
        .def_rw("w_router", &FusedHeadWeights::w_router)
        .def_rw("router_bias", &FusedHeadWeights::router_bias)
        .def_rw("router_index", &FusedHeadWeights::router_index);
    nb::class_<FusedHeadOutputs>(mod, "FusedHeadOutputs", "The fused head's outputs (the resources' tensors).")
        .def_ro("se", &FusedHeadOutputs::se)
        .def_ro("rows", &FusedHeadOutputs::rows)
        .def_ro("xbuf", &FusedHeadOutputs::xbuf)
        .def_ro("route", &FusedHeadOutputs::route)
        .def_ro("counts", &FusedHeadOutputs::counts)
        .def_ro("regions", &FusedHeadOutputs::regions)
        .def_ro("token_index", &FusedHeadOutputs::token_index)
        .def_ro("table", &FusedHeadOutputs::table)
        .def_ro("plan", &FusedHeadOutputs::plan);

    ttnn::bind_function<"prepare_fused_head_weights", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        The shared expert's and the router's weights in the fused head's layouts (its planned engine), once a layer.

        Args:
            gate, up: the shared expert's [Is, H] (host, float32 or bfloat16).
            down: [H, Is].
            latent_down: a latent model's [Hr, H] (None without).
            router_weight: [E, H] (E: the router's candidates, num_experts x fold); router_bias: [E].

        Keyword args:
            resources, dtypes, accumulators, l1_free: as fused_head.

        Returns:
            FusedHeadWeights (device tensors).
        )doc",
        &prepare_fused_head_weights,
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

    ttnn::bind_function<"fused_head", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        The MoE block's head as one program a chip: the x all-gather over TP into the shared expert's matmul cores,
        its down + reduce-scatter, the router (logits, top-K, the owners' joint routing rows), the routing exchange and
        the combine planner (the combine's plan tensors), the dispatch prep (this chip's owned rows into the dispatch
        buffer, their route words) and, with a latent, the latent down + all-to-all.

        Args:
            x: this chip's [1, 1, rows, H / TP] (rows: a mesh row's tokens), bfloat16 TILE, DRAM.
            weights: prepare_fused_head_weights'.

        Keyword args:
            resources: the block's MoEBlockResources.
            dtypes: MoEDtypes (the shared expert's weights / fidelity).
            accumulators: the combine's accumulator count (the routing planner's instances at most).
            x_full: the DRAM fallback's gathered x [1, 1, rows, H] (only when the plan took the fallback).
            l1_free: the free L1 a core the head may take (default: measured at the block's first head).
            arena: an L1 tensor over the head's arena cores the head runs in (default one the resources keep, which
                holds that L1 for good: a block shares one arena between its ops).
            actual_start, actual_end: a padded chunk - main's chunked-prefill metadata scalars as they are (the
                chunk's first real token's absolute position, one past its last), one-element uint32 tensors in DRAM,
                replicated, read on device every call (a trace replays with their values of the moment). The chunk is
                in main's KV-pad-aware rotated layout over the SP rows (moe_padding_config): an SP row's real rows are
                a prefix, TP chip p owns the row's tokens p S ... Padding tokens get no routes: not dispatched, not
                computed by the routed experts, not combined (their combined rows are zero). None: every row is real.
            loop: this call's loop over the chunk when it runs as MoEBlockConfig.chunk_splits x token_loops head
                calls (split x token_loops + its loop in the split; x: that call's rows of every chip). A split's
                last loop plans every loop's tokens of the split.

        Returns:
            FusedHeadOutputs: se, rows, xbuf, route and the combine plan (counts, regions, token_index, table,
            plan) - the resources' tensors, rewritten each call.

        The plan, the host model and the persistent words are made on the first call and kept in the resources; a
        repeated call rewrites only the arguments of the tensors whose address moved.
        )doc",
        &fused_head,
        nb::arg("x"),
        nb::arg("weights"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("dtypes"),
        nb::arg("accumulators"),
        nb::arg("x_full") = nb::none(),
        nb::arg("l1_free") = nb::none(),
        nb::arg("arena") = nb::none(),
        nb::arg("actual_start") = nb::none(),
        nb::arg("actual_end") = nb::none(),
        nb::arg("loop") = 0);
}

}  // namespace ttnn::operations::experimental::moe_block::fused_head
