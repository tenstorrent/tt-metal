// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_head_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"

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
    export_enum<WeightDtype>(mod, "HeadWeightDtype");
    export_enum<RouterPlacementMode>(mod, "HeadRouterPlacement");

    nb::class_<HeadShape> shape(mod, "HeadShape", R"doc(
        What the fused head computes, a chip (head_plan.HeadShape): rows (a mesh row's tokens), H, I (the shared
        expert's intermediate a TP chip), G (TP), E (router candidates), K, latent, fold, links, ring (the TP axis
        wraps), grid, R (mesh rows), sp_ring, payload (fabric packet payload bytes), A (combine accumulators), w_dtype.
        Any field can be given as a keyword.)doc");
    shape.def("__init__", [](HeadShape* self, const nb::kwargs& kwargs) {
        HeadShape value{};
        nb::object proxy = nb::cast(&value, nb::rv_policy::reference);
        for (auto [k, v] : kwargs) {
            if (!nb::hasattr(proxy, k)) {
                throw nb::attribute_error(("HeadShape has no field '" + nb::cast<std::string>(k) + "'").c_str());
            }
            nb::setattr(proxy, k, v);
        }
        new (self) HeadShape(value);
    });
    shape.def_rw("rows", &HeadShape::rows)
        .def_rw("H", &HeadShape::H)
        .def_rw("I", &HeadShape::I)
        .def_rw("G", &HeadShape::G)
        .def_rw("E", &HeadShape::E)
        .def_rw("K", &HeadShape::K)
        .def_rw("latent", &HeadShape::latent)
        .def_rw("fold", &HeadShape::fold)
        .def_rw("links", &HeadShape::links)
        .def_rw("ring", &HeadShape::ring)
        .def_rw("grid", &HeadShape::grid)
        .def_rw("R", &HeadShape::R)
        .def_rw("sp_ring", &HeadShape::sp_ring)
        .def_rw("payload", &HeadShape::payload)
        .def_rw("A", &HeadShape::A)
        .def_rw("w_dtype", &HeadShape::w_dtype)
        .def(nb::self == nb::self);

    nb::class_<HeadLayout> lay(mod, "HeadLayout", "A layout of the fused head (head_plan.HeadLayout).");
    lay.def_ro("GY", &HeadLayout::GY)
        .def_ro("MT", &HeadLayout::MT)
        .def_ro("NCH", &HeadLayout::NCH)
        .def_ro("KB", &HeadLayout::KB)
        .def_ro("on_engine", &HeadLayout::on_engine)
        .def_ro("GX", &HeadLayout::GX)
        .def_ro("compact", &HeadLayout::compact)
        .def_ro("split", &HeadLayout::split)
        .def_ro("xdram", &HeadLayout::xdram)
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
        head_plan_fit's answer: ok, planned (best first), why (the requirement that failed, with its numbers and what
        would fit), free (the cost model's pick with unlimited L1), note (what L1 forced, or the DRAM fallback's cost).)doc");
    fit.def_ro("ok", &HeadFit::ok)
        .def_ro("planned", &HeadFit::planned)
        .def_ro("why", &HeadFit::why)
        .def_ro("free", &HeadFit::free)
        .def_ro("note", &HeadFit::note)
        .def("__bool__", [](const HeadFit& f) { return f.ok; });

    mod.def(
        "head_plan_fit",
        [](const HeadShape& shape,
           std::optional<int64_t> l1_free,
           bool split,
           std::optional<bool> on_engine,
           std::optional<uint32_t> KB,
           RouterPlacementMode placement,
           bool largest,
           bool explain,
           bool fallback) {
            HeadFitOptions o;
            o.l1_free = l1_free;
            o.split = split;
            o.on_engine = on_engine;
            o.KB = KB.value_or(0);
            o.placement = placement;
            o.largest = largest;
            o.explain = explain;
            o.fallback = fallback;
            return head_plan_fit(shape, o);
        },
        nb::arg("shape"),
        nb::kw_only(),
        nb::arg("l1_free").none() = nb::none(),
        nb::arg("split") = false,
        nb::arg("on_engine").none() = nb::none(),
        nb::arg("KB").none() = nb::none(),
        nb::arg("placement") = RouterPlacementMode::Measured,
        nb::arg("largest") = true,
        nb::arg("explain") = true,
        nb::arg("fallback") = true,
        R"doc(
        The fused head's whole host plan (head_plan.fit), no device: every candidate layout of `shape` whose L1 and
        per-core requirements hold, ranked by the cost model. l1_free: free L1 a core at the head's build (default
        the planner's host figure); split / on_engine / KB: restrict the layouts; fallback: the DRAM fallback where no
        resident layout fits; largest: on an L1 refusal, also the largest rows that fit; explain: the unconstrained
        pick and the L1 note.)doc");
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
        The fused head's whole host plan on a block's resources (MoEBlock's FusedHead.plan): the model's
        requirements, then head_plan_fit on the mesh's compute grid, fabric payload and the config's DRAM fallback.
        Computed once and kept in the resources. l1_free: free L1 a core the head may take (default the planner's
        host figure).)doc");

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
        .def_ro("latent", &FusedHeadOutputs::latent)
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
            l1_free: the free L1 a core the head may take (default the planner's host figure).

        Returns:
            FusedHeadOutputs: se, rows, xbuf, route, latent, and the combine plan (counts, regions, token_index, table,
            plan) - the resources' tensors, rewritten each call.

        The plan, the host model and the persistent words are made on the first call and kept in the resources; a
        repeated call patches only the tensors' addresses.
        )doc",
        &fused_head,
        nb::arg("x"),
        nb::arg("weights"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("dtypes"),
        nb::arg("accumulators"),
        nb::arg("x_full") = nb::none(),
        nb::arg("l1_free") = nb::none());

    nb::class_<FusedHeadPrograms>(mod, "FusedHeadPrograms", "fused_head_programs' answer (tests).")
        .def_prop_ro(
            "programs",
            [](const FusedHeadPrograms& p) {
                nb::list out;
                for (const auto& [c, d] : p.programs) {
                    out.append(nb::make_tuple(nb::make_tuple(c[0], c[1]), nb::cast(d, nb::rv_policy::copy)));
                }
                return out;
            })
        .def_ro("addresses", &FusedHeadPrograms::addresses)
        .def_ro("fabric_starts", &FusedHeadPrograms::fabric_starts);
    mod.def(
        "fused_head_programs",
        &fused_head_programs,
        nb::arg("x"),
        nb::arg("weights"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("dtypes"),
        nb::arg("accumulators"),
        nb::arg("x_full") = nb::none(),
        nb::arg("l1_free") = nb::none(),
        R"doc(
        The per-chip program descriptors fused_head would launch, without launching (tests: compared with the
        reference's on a mock mesh).)doc");
}

}  // namespace ttnn::operations::experimental::moe_block::fused_head
