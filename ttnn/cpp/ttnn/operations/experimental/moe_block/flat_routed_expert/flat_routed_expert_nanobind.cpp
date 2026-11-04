// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_nanobind.hpp"

#include <map>
#include <memory>
#include <string>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/mesh_device.hpp>

#include "device/flat_program.hpp"
#include "device/flat_routed_expert_plan.hpp"
#include "flat_routed_expert.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

namespace nanobind_detail {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::distributed::MeshDevice;

// A plan as Python holds it (plans are shared and immutable).
struct PlanHandle {
    std::shared_ptr<const FlatRoutedExpertPlan> plan;
};

nb::list cores(const std::vector<CoreCoord>& cs) {
    nb::list l;
    for (const auto& c : cs) {
        l.append(nb::make_tuple(c.x, c.y));
    }
    return l;
}

// The plan's layout facts the weight preparation and callers' checks need.
nb::dict plan_dict(const FlatRoutedExpertPlan& p) {
    nb::dict d;
    d["n_rd"] = p.readers.size();
    d["n_rd_sg"] = p.n_rd_sg;
    d["np"] = p.np;
    d["mt"] = p.mt;
    d["rg"] = p.rg;
    d["nu"] = p.nu;
    d["nk_gu"] = p.nk_gu;
    d["nd"] = p.down.size();
    d["pcds"] = p.pcds;
    d["col0s"] = p.col0s;
    d["banks"] = p.banks;
    d["wd_rot_tiles"] = p.w_tile ? p.wd_rot_region / p.w_tile : 0;
    d["coords"] = cores(p.coords);
    d["gu"] = cores(p.gu);
    d["relays"] = cores(p.relays);
    d["down"] = cores(p.down);
    return d;
}

}  // namespace nanobind_detail

void bind_flat_routed_expert(nb::module_& mod) {
    using namespace nanobind_detail;

    nb::class_<FlatLayoutOptions>(mod, "FlatLayoutOptions", R"doc(
        The flat routed expert's layout / program choices (part of the plan key and of the program hash). Unset
        fields keep the plan's own choice.

        Keyword Args:
            pin_ratio (int): pin an expert only at this many times the next one's rows (0: always pin).
            rows_hint (int): expected mean rows per local expert (-1: unknown); <= 320 takes 64-row sub-blocks on
                plans with gate/up row passes.
            gu_read_pieces (int): each gate/up weight block read as 1 / 2 / 4 pieces (-1: 1).
    )doc")
        .def(
            "__init__",
            [](FlatLayoutOptions* self, uint32_t pin_ratio, int32_t rows_hint, int32_t gu_read_pieces) {
                new (self)
                    FlatLayoutOptions{.pin_ratio = pin_ratio, .rows_hint = rows_hint, .gu_read_pieces = gu_read_pieces};
            },
            nb::kw_only(),
            nb::arg("pin_ratio") = 8,
            nb::arg("rows_hint") = -1,
            nb::arg("gu_read_pieces") = -1)
        .def_rw("pin_ratio", &FlatLayoutOptions::pin_ratio)
        .def_rw("rows_hint", &FlatLayoutOptions::rows_hint)
        .def_rw("gu_read_pieces", &FlatLayoutOptions::gu_read_pieces)
        .def("key", &FlatLayoutOptions::key)
        .def("__repr__", [](const FlatLayoutOptions& o) { return "FlatLayoutOptions(" + o.str() + ")"; })
        .def("__eq__", [](const FlatLayoutOptions& a, const FlatLayoutOptions& b) { return a == b; });

    nb::class_<PlanHandle>(mod, "FlatRoutedExpertPlan", R"doc(
        The flat routed expert's plan for a mesh device and shape: role cores, work split, arena layout, weight bank
        layout. ``as_dict()`` gives its layout facts (what the weight preparation needs).
    )doc")
        .def("as_dict", [](const PlanHandle& h) { return plan_dict(*h.plan); });

    mod.def(
        "flat_routed_expert_plan",
        [](MeshDevice& mesh_device,
           uint32_t hidden,
           uint32_t intermediate,
           uint32_t experts_per_chip,
           uint32_t num_global_experts,
           uint32_t max_tokens_per_expert,
           bool weights_bf8,
           uint32_t pin,
           const FlatLayoutOptions& layout) {
            return PlanHandle{flat_routed_expert_plan(
                mesh_device,
                make_flat_routed_expert_config(
                    hidden,
                    intermediate,
                    experts_per_chip,
                    num_global_experts,
                    max_tokens_per_expert,
                    weights_bf8,
                    pin,
                    layout))};
        },
        nb::arg("mesh_device"),
        nb::arg("hidden"),
        nb::arg("intermediate"),
        nb::arg("experts_per_chip"),
        nb::arg("num_global_experts"),
        nb::arg("max_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("weights_bf8") = false,
        nb::arg("pin") = 1,
        nb::arg("layout") = FlatLayoutOptions{},
        R"doc(
        The flat routed expert's plan (cached per mesh device and shape) for hidden H, intermediate I (per device),
        experts_per_chip local experts, num_global_experts (the counts' length) and the per-expert capacity.
        )doc");

    mod.def(
        "create_flat_routed_expert_done_words",
        [](MeshDevice& mesh_device, const PlanHandle& plan, std::optional<tt::tt_metal::BufferType> buffer_type) {
            return create_flat_routed_expert_done_words(
                mesh_device, *plan.plan, buffer_type.value_or(tt::tt_metal::BufferType::L1));
        },
        nb::arg("mesh_device"),
        nb::arg("plan"),
        nb::arg("buffer_type") = nb::none(),
        R"doc(
        The down coordinators' done words for ``plan``: a zeroed persistent uint32 tensor sharded over the plan's
        coordinator cores in L1 (or L1_SMALL). Keep it for every call of the op; the op leaves it zero.
        )doc");

    ttnn::bind_function<"flat_routed_expert", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        Flat spatially pipelined routed experts (Blackhole): every local expert's GLU FFN of a chip in ONE program
        that streams each expert's weights once through a spatial pipeline (the gate/up cores read their weight
        blocks from DRAM, x relays read / tilize the row-major dispatch buffer and feed it along unicast snakes, down
        cores write y at each expert's region). The routing's counts / regions rows are read on device, so one cached
        program serves every routing.

            y_e = act(x_e @ Wg_e, x_e @ Wu_e) @ Wd_e   for every local expert e

        Args:
            dispatched_buffer (ttnn.Tensor): [rows * P, H / P] bf16 ROW_MAJOR, DRAM interleaved (P: x_pages_per_row).
            expert_token_counts (ttnn.Tensor): [1, num_global_experts] uint32 ROW_MAJOR, DRAM interleaved.
            expert_region_offsets (ttnn.Tensor): [1, num_global_experts] uint32 ROW_MAJOR: each expert's first row.
            global_expert_ids (ttnn.Tensor): [experts_per_chip] uint32 ROW_MAJOR, DRAM (per device).
            gate_up_weights (ttnn.Tensor): the plan's gate/up bank layout (bfloat4_b or bfloat8_b).
            down_weights (ttnn.Tensor): the plan's down bank layout.
            done_words (ttnn.Tensor): create_flat_routed_expert_done_words(mesh_device, plan).
            intermediate (int): the expert intermediate size (per device).
            max_tokens_per_expert (int): the dispatch capacity per expert (>= 256).

        Keyword Args:
            activation (int): 0 SiLU, 1 SwiGLU-OAI, 2 SiTU, 3 clamped SiLU, 4 GELU-tanh.
            pin (int): pin the largest expert's weights in chunks of >= pin sub-blocks (0: off).
            token_index (ttnn.Tensor, optional): [1, rows] uint32 ROW_MAJOR DRAM; flat row r reads x row
                token_index[r] (x is then e.g. the all-gathered tokens); y has token_index's rows.
            x_pages_per_row (int): x stores each token row as this many consecutive pages.
            y_row_major (bool): y as row-major bf16 [rows, H], pack-untilized on the down cores.
            down_fp32 (bool): accumulate the down projection in fp32 DEST.
            pack_stochastic_rounding (bool): stochastic rounding in the compute kernels' packers.
            y_bf16 (bool): y as bf16 tiles (tiled y only).
            math_fidelity (int): the expert matmuls' fidelity (0 LoFi: exact for bfp4 weights; 2 HiFi2 for bfp8).
            layout (FlatLayoutOptions): the layout / program choices; must match the plan the weights were laid out
                with.
            output (ttnn.Tensor, optional): a preallocated y.

        Returns:
            ttnn.Tensor: y [rows, H] bfp8 TILE (bf16 TILE with y_bf16, bf16 ROW_MAJOR with y_row_major), written at
            the active experts' rows.
        )doc",
        &flat_routed_expert,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("global_expert_ids").noconvert(),
        nb::arg("gate_up_weights").noconvert(),
        nb::arg("down_weights").noconvert(),
        nb::arg("done_words").noconvert(),
        nb::arg("intermediate"),
        nb::arg("max_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("activation") = 0,
        nb::arg("pin") = 1,
        nb::arg("token_index") = nb::none(),
        nb::arg("x_pages_per_row") = 1,
        nb::arg("y_row_major") = false,
        nb::arg("down_fp32") = false,
        nb::arg("pack_stochastic_rounding") = false,
        nb::arg("y_bf16") = false,
        nb::arg("math_fidelity") = 0,
        nb::arg("layout") = FlatLayoutOptions{},
        nb::arg("output") = nb::none());
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
