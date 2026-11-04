// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"

#include "device/combine_host.hpp"
#include "routed_experts_combine.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace routed_experts_combine_nanobind_detail {

nb::dict layout_dict(const CombineLayout& l) {
    nb::dict d;
    d["slot"] = l.slot;
    d["acc_fp32"] = l.acc_fp32;
    d["acc_cbs"] = l.acc_cbs;
    d["rings"] = l.rings;
    d["fits"] = l.fits;
    d["why"] = l.why;
    if (l.has_PSL) {
        d["PSL"] = l.PSL;
        d["D"] = l.D;
    }
    if (l.has_rings) {
        d["QP"] = l.QP;
        d["QL"] = l.QL;
        d["acc_used"] = l.acc_used;
        d["port_used"] = l.port_used;
    }
    return d;
}

nb::dict choice_dict(const CombineAutoChoice& c) {
    nb::dict d;
    d["acc_on_ports"] = c.acc_on_ports;
    d["n_acc"] = c.n_acc;
    d["inj_slots"] = c.inj_slots;
    d["depth"] = c.depth;
    d["local_keep"] = c.local_keep;
    d["layout"] = layout_dict(c.layout);
    return d;
}

}  // namespace routed_experts_combine_nanobind_detail

void bind_routed_experts_combine(nb::module_& mod) {
    using namespace routed_experts_combine_nanobind_detail;

    export_enum<CombineMode>(mod, "CombineMode");

    nb::class_<CombineConfig> cfg(mod, "CombineConfig", R"doc(
        The combine's op-specific configuration (combine_snap comes from MoEBlockConfig). n_acc / mode pin the
        accumulator count and mode the fused head planned; arena_bytes is the combine's L1 budget a core (0: the
        exchange's sizing from the largest free L1 block at the first call, less the flat program's static CBs and the
        dispatch's); depth is the landing pool's slots per source chip. A row travels as one fabric packet with its
        32 B trailer.
    )doc");
    cfg.def(nb::init<>())
        .def_rw("n_acc", &CombineConfig::n_acc)
        .def_rw("mode", &CombineConfig::mode)
        .def_rw("arena_bytes", &CombineConfig::arena_bytes)
        .def_rw("depth", &CombineConfig::depth)
        .def(nb::self == nb::self);

    ttnn::bind_function<"routed_experts_combine", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        The flat routed experts and the sparse combine in one program per chip: every expert output row goes from the
        flat expert's y writers through L1 into a fabric port's injection ring (or a local accumulator's ring), home
        over the 2D fabric (Y first, then X) into an accumulator's landing ring, and is summed there in fp32 (gate
        weight x row, snapped to multiples of combine_snap when set: bit-identical in any landing order).

        Args:
            dispatched: the gathered rows [N S, Hr] bf16 row-major (the dispatch's output).
            counts, regions, token_index, table, plan: the fused head's planner outputs (per-expert counts and first
                rows, the flat rows' token index, the L1 hand-off table, the combine's per-call plan).
            global_expert_ids, gate_up_weights, down_weights: the layer's flat expert, laid out for the flat plan
                (flat_routed_expert_plan with this op's capacity and layout).
            resources: the block's MoEBlockResources (config, placement, persistent words).
            max_tokens_per_expert: the flat expert's capacity m.
            activation: the experts' activation (gate_up_compute SE_ACT: 0 SiLU, 1 SwiGLU-OAI, 2 SiTU, 3 clamped SiLU).
            arena: the block's shared L1 arena (routed_experts_combine_arena_spec; the fused head ran in it first);
                None: the op allocates one for the call.
            layout: FlatLayoutOptions.
            combine: CombineConfig.

        Returns:
            y_own [1, 1, S, Hr] bf16 row-major (persistent: one buffer for every call).
        )doc",
        &routed_experts_combine,
        nb::arg("dispatched").noconvert(),
        nb::arg("counts").noconvert(),
        nb::arg("regions").noconvert(),
        nb::arg("token_index").noconvert(),
        nb::arg("table").noconvert(),
        nb::arg("plan").noconvert(),
        nb::arg("global_expert_ids").noconvert(),
        nb::arg("gate_up_weights").noconvert(),
        nb::arg("down_weights").noconvert(),
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("max_tokens_per_expert"),
        nb::arg("activation"),
        nb::arg("arena") = nb::none(),
        nb::arg("layout") = flat_routed_expert::FlatLayoutOptions{},
        nb::arg("combine") = CombineConfig{});

    mod.def(
        "routed_experts_combine_arena_spec",
        &routed_experts_combine_arena_spec,
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("rows_cap"),
        nb::arg("max_tokens_per_expert"),
        nb::arg("activation"),
        nb::arg("layout") = flat_routed_expert::FlatLayoutOptions{},
        nb::arg("combine") = CombineConfig{},
        nb::arg("other_bytes") = 0,
        R"doc(
        The TensorSpec of the L1 arena the block allocates once a call (ttnn.allocate_tensor_on_device) and passes to
        fused_head(..., arena=) and routed_experts_combine(..., arena=): height-sharded over every worker core,
        max(this op's need, other_bytes) a core - other_bytes: the head's arena bytes. rows_cap: the hand-off table's
        rows. Plans the op on first use (the same arguments as the op's).
        )doc");

    // ---- host math (no device)
    mod.def(
        "combine_state_layout",
        [](uint32_t N, uint32_t A) {
            const auto s = state_layout(N, A);
            return nb::make_tuple(s.CRED, s.USED, s.RET, s.LAYR, s.LAYO, s.U0, s.CNT, s.MISC, s.HDRS, s.RINGS);
        },
        nb::arg("N"),
        nb::arg("A"),
        "The per-core state layout: (CRED, USED, RET, LAYR, LAYO, U0, CNT, MISC, HDRS, RINGS).");
    mod.def(
        "combine_auto",
        [](uint32_t N,
           uint32_t S,
           uint32_t W,
           uint32_t n_cells,
           uint32_t port_cells,
           uint32_t arena_bytes,
           uint32_t depth,
           uint32_t a_max,
           std::optional<CombineMode> mode) {
            const auto r = combine_auto(N, S, W, n_cells, port_cells, arena_bytes, depth, a_max, mode);
            nb::dict modes;
            if (r.ports_apart) {
                modes["ports apart"] = choice_dict(*r.ports_apart);
            }
            if (r.acc_on_ports) {
                modes["acc_on_ports"] = choice_dict(*r.acc_on_ports);
            }
            nb::dict d = r.best ? choice_dict(*r.best) : nb::dict();
            d["fits"] = r.fits;
            if (!r.fits) {
                d["why"] = r.why;
            }
            d["modes"] = modes;
            return d;
        },
        nb::arg("N"),
        nb::arg("S"),
        nb::arg("W"),
        nb::arg("n_cells"),
        nb::arg("port_cells"),
        nb::kw_only(),
        nb::arg("arena_bytes"),
        nb::arg("depth") = 8,
        nb::arg("a_max") = kAccMax,
        nb::arg("mode").none() = nb::none(),
        "The combine's automatic configuration as a dict (mode: a CombineMode, or none for the best).");
    mod.def(
        "combine_credit_policy",
        [] {
            const auto p = credit_policy();
            return nb::make_tuple(p.cred_thr, p.cred_idle, p.cred_age);
        },
        "The combine's credit policy: (CRED_THR, CRED_IDLE, CRED_AGE).");
    mod.def(
        "routed_experts_combine_l1_plan",
        [](MoEBlockResources& resources,
           uint32_t max_tokens_per_expert,
           uint32_t activation,
           const flat_routed_expert::FlatLayoutOptions& layout,
           const CombineConfig& combine,
           uint32_t dispatch_group,
           uint32_t head_port_cb) {
            const auto l1 = plan_routed_experts_combine_l1(
                resources, max_tokens_per_expert, activation, layout, combine, dispatch_group, head_port_cb);
            nb::dict d;
            d["layout"] = l1.flat.layout;
            d["static_bytes"] = l1.static_bytes;
            d["arena_bytes"] = l1.arena_bytes;
            d["A"] = l1.plan->A;
            d["D"] = l1.plan->D;
            d["QP"] = l1.plan->QP;
            d["QL"] = l1.plan->QL;
            d["acc_on_ports"] = l1.plan->fused;
            return d;
        },
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("max_tokens_per_expert"),
        nb::arg("activation"),
        nb::arg("layout") = flat_routed_expert::FlatLayoutOptions{},
        nb::arg("combine") = CombineConfig{},
        nb::arg("dispatch_group"),
        nb::arg("head_port_cb") = 0,
        "Host-only what-if: the combine's L1 plan at a dispatch group as a dict (the flat layout fitted into its "
        "arena, "
        "static CB bytes, arena bytes, accumulators A, landing depth D, QP, QL, acc_on_ports).");
    mod.def(
        "routed_experts_combine_static_cb_bytes",
        &routed_experts_combine_static_cb_bytes,
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("max_tokens_per_expert"),
        nb::arg("activation"),
        nb::arg("layout") = flat_routed_expert::FlatLayoutOptions{},
        R"doc(
        L1 bytes a core the block's own programs that run while its arena is allocated (the dispatch, the flat experts
        + the combine) take for their static circular buffers: what the arena must stay above.
        )doc");
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
