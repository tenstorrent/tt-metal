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
    d["slot_bytes"] = l.slot_bytes;
    d["accumulator_fp32_bytes"] = l.accumulator_fp32_bytes;
    d["accumulator_cb_bytes"] = l.accumulator_cb_bytes;
    d["state_bytes"] = l.state_bytes;
    d["fits"] = l.fits;
    d["why"] = l.why;
    if (l.has_landing_slots) {
        d["landing_slots_per_accumulator"] = l.landing_slots_per_accumulator;
        d["landing_slots_per_source"] = l.landing_slots_per_source;
    }
    if (l.has_ring_sizes) {
        d["injection_ring_slots"] = l.injection_ring_slots;
        d["local_ring_slots"] = l.local_ring_slots;
        d["accumulator_core_bytes"] = l.accumulator_core_bytes;
        d["port_core_bytes"] = l.port_core_bytes;
    }
    return d;
}

nb::dict choice_dict(const CombineAutoChoice& c) {
    nb::dict d;
    d["accumulators_on_ports"] = c.accumulators_on_ports;
    d["accumulator_count"] = c.accumulator_count;
    d["injection_ring_slots"] = c.injection_ring_slots;
    d["depth"] = c.depth;
    d["local_ring_slots_reserved"] = c.local_ring_slots_reserved;
    d["shared_landing"] = c.shared_landing;
    d["layout"] = layout_dict(c.layout);
    return d;
}

}  // namespace routed_experts_combine_nanobind_detail

void bind_routed_experts_combine(nb::module_& mod) {
    using namespace routed_experts_combine_nanobind_detail;

    export_enum<CombineMode>(mod, "CombineMode");
    export_enum<CombineLanding>(mod, "CombineLanding");

    nb::class_<CombineConfig> cfg(mod, "CombineConfig", R"doc(
        The combine's op-specific configuration (combine_snap comes from MoEBlockConfig). accumulator_count / mode pin the
        accumulator count and mode the MoE head planned; arena_bytes is the combine's L1 budget a core (0: the
        exchange's sizing from the largest free L1 block at the first call, less the flat program's static CBs and the
        dispatch's); depth is the landing pool's slots per source chip; landing: a CombineLanding (Rings: one ring a
        source chip; Shared: slot grants from one free list; FlatCores: rings in the flat expert cores' free L1;
        SharedFlatCores: Shared's grants on FlatCores' slots), None the planner's (rings in the accumulators' L1 or on
        the flat expert cores, ranked by combine_choice_key). A row travels as one fabric packet with its 32 B trailer.
    )doc");
    cfg.def(nb::init<>())
        .def_rw("accumulator_count", &CombineConfig::accumulator_count)
        .def_rw("mode", &CombineConfig::mode)
        .def_rw("arena_bytes", &CombineConfig::arena_bytes)
        .def_rw("depth", &CombineConfig::depth)
        .def_rw("landing", &CombineConfig::landing)
        .def_rw("credit_batch", &CombineConfig::credit_batch)
        .def(nb::self == nb::self);  // NOLINT(misc-redundant-expression): binds operator==

    ttnn::bind_function<"routed_experts_combine", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        The flat routed experts and the sparse combine in one program per chip: every expert output row goes from the
        flat expert's y writers through L1 into a fabric port's injection ring (or a local accumulator's ring), home
        over the 2D fabric (Y first, then X) into an accumulator's landing ring, and is summed there in fp32 (gate
        weight x row, snapped to multiples of combine_snap when set: bit-identical in any landing order).

        Args:
            dispatched: the gathered rows [N S, Hr] bf16 row-major (the dispatch's output).
            counts, regions, token_index, table, plan: the MoE head's planner outputs (per-expert counts and first
                rows, the flat rows' token index, the L1 hand-off table, the combine's per-call plan).
            global_expert_ids, gate_up_weights, down_weights: the layer's flat expert, laid out for the flat plan
                (flat_routed_expert_plan with this op's capacity and layout).
            resources: the block's MoEBlockResources (config, placement, persistent words).
            max_tokens_per_expert: the flat expert's capacity m.
            arena: the block's shared L1 arena (routed_experts_combine_arena_spec; the MoE head ran in it first);
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
        nb::arg("layout") = flat_routed_expert::FlatLayoutOptions{},
        nb::arg("combine") = CombineConfig{},
        nb::arg("other_bytes") = 0,
        R"doc(
        The TensorSpec of the L1 arena the block allocates once a call (ttnn.allocate_tensor_on_device) and passes to
        moe_head(..., arena=) and routed_experts_combine(..., arena=): height-sharded over every worker core,
        max(this op's need, other_bytes) a core - other_bytes: the head's arena bytes. rows_cap: the hand-off table's
        rows. Plans the op on first use (the same arguments as the op's).
        )doc");

    // ---- host math (no device)
    mod.def(
        "combine_state_layout",
        [](uint32_t N, uint32_t A) {
            const auto s = state_layout(N, A);
            return nb::make_tuple(
                s.credits_received_offset,
                s.rows_sent_offset,
                s.credits_to_return_offset,
                s.ring_layouts_received_offset,
                s.ring_layouts_to_forward_offset,
                s.rows_sent_at_call_start_offset,
                s.counts_offset,
                s.misc_words_offset,
                s.row_headers_offset,
                s.rings_offset);
        },
        nb::arg("N"),
        nb::arg("A"),
        "The per-core state layout's byte offsets (StateLayout): (credits_received, rows_sent, credits_to_return, "
        "ring_layouts_received, ring_layouts_to_forward, rows_sent_at_call_start, counts, misc_words, row_headers, "
        "rings).");
    mod.def(
        "combine_auto",
        [](uint32_t N,
           uint32_t S,
           uint32_t W,
           uint32_t free_core_count,
           uint32_t port_core_count,
           uint32_t arena_bytes,
           uint32_t depth,
           uint32_t a_max,
           std::optional<CombineMode> mode,
           CombineLanding landing,
           uint32_t pool_slots) {
            const auto r = combine_auto(
                N, S, W, free_core_count, port_core_count, arena_bytes, depth, a_max, mode, landing, pool_slots);
            nb::dict modes;
            if (r.ports_apart) {
                modes["ports apart"] = choice_dict(*r.ports_apart);
            }
            if (r.accumulators_on_ports) {
                modes["accumulators_on_ports"] = choice_dict(*r.accumulators_on_ports);
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
        nb::arg("free_core_count"),
        nb::arg("port_core_count"),
        nb::kw_only(),
        nb::arg("arena_bytes"),
        nb::arg("depth") = 8,
        nb::arg("a_max") = kAccMax,
        nb::arg("mode").none() = nb::none(),
        nb::arg("landing") = CombineLanding::Rings,
        nb::arg("pool_slots") = 0,
        "The combine's automatic configuration as a dict (mode: a CombineMode, or none for the best; pool_slots: "
        "CombineLanding.FlatCores' slots on the flat expert cores a chip).");
    mod.def(
        "combine_credit_policy",
        [](uint32_t N, uint32_t landing_slots_per_source, bool shared_landing, std::optional<uint32_t> batch) {
            const auto p = credit_policy(N, landing_slots_per_source, shared_landing, batch);
            return nb::make_tuple(p.return_credits_after_rows, p.idle_snapshot_rows, p.idle_snapshot_age_cycles);
        },
        nb::arg("N") = 32,
        nb::arg("landing_slots_per_source") = 16,
        nb::arg("shared_landing") = false,
        nb::arg("batch").none() = nb::none(),
        "The combine's credit policy (CreditPolicy) at landing_slots_per_source a source chip (and the shared "
        "landing): "
        "(return_credits_after_rows, idle_snapshot_rows, idle_snapshot_age_cycles).");
    mod.def(
        "routed_experts_combine_l1_plan",
        [](MoEBlockResources& resources,
           uint32_t max_tokens_per_expert,
           const flat_routed_expert::FlatLayoutOptions& layout,
           const CombineConfig& combine,
           uint32_t dispatch_group,
           uint32_t head_port_cb) {
            const auto l1 = plan_routed_experts_combine_l1(
                resources, max_tokens_per_expert, layout, combine, dispatch_group, head_port_cb);
            nb::dict d;
            d["layout"] = l1.flat.layout;
            d["static_bytes"] = l1.static_bytes;
            d["arena_bytes"] = l1.arena_bytes;
            d["accumulator_count"] = l1.plan->accumulator_count;
            d["landing_slots_per_source"] = l1.plan->landing_slots_per_source;
            d["injection_ring_slots"] = l1.plan->injection_ring_slots;
            d["local_ring_slots"] = l1.plan->local_ring_slots;
            d["accumulators_on_ports"] = l1.plan->accumulators_on_ports;
            d["shared_landing"] = l1.plan->shared_landing;
            d["landing_slots_per_accumulator"] = l1.plan->landing_slots_per_accumulator;
            d["landing_on_flat_expert_cores"] = l1.plan->landing_on_flat_expert_cores;
            d["pool_slots"] = l1.plan->pool_slots;
            return d;
        },
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("max_tokens_per_expert"),
        nb::arg("layout") = flat_routed_expert::FlatLayoutOptions{},
        nb::arg("combine") = CombineConfig{},
        nb::arg("dispatch_group"),
        nb::arg("head_port_cb") = 0,
        "Host-only what-if: the combine's L1 plan at a dispatch group as a dict (the flat layout fitted into its "
        "arena, "
        "static CB bytes, arena bytes, accumulator_count, landing_slots_per_source, injection_ring_slots, "
        "local_ring_slots, "
        "accumulators_on_ports, shared_landing, landing_slots_per_accumulator, landing_on_flat_expert_cores, "
        "pool_slots).");
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
